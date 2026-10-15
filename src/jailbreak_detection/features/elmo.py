
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def calculate_elmo_cosine_similarity(df_pares, elmo): #para prompts completos
    # Obtener los 60 prompts tokenizados
    prompts = (
        df_pares["tokens_a"].tolist() +
        df_pares["tokens_b"].tolist()
    )

    # Generar los embeddings por lotes
    vectores = []

    batch_size = 4

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        embeddings = elmo.get_elmo_vector_average(batch)
        vectores.extend(embeddings)

    vectores = np.array(vectores)
    normas = np.linalg.norm(vectores, axis=1)
    malos = ~np.isfinite(vectores).all(axis=1) | ~np.isfinite(normas) | (normas == 0)

    for i in np.flatnonzero(malos):
        lado = "a" if i < len(df_pares) else "b"
        par = df_pares.iloc[i % len(df_pares)]["id_par"]
        print(f"Vector inválido: par {par}, lado {lado}")

    # Separar los embeddings A y B
    n = len(df_pares)

    A = vectores[:n]
    B = vectores[n:]

    # Calcular similitud coseno de cada pareja
    result= (
        np.sum(A * B, axis=1) /
        (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
    )

    return result







def calculate_elmo_word_similarity(palabras, docs_train, elmo, max_prompts=5):

    resultados = []

    for palabra in palabras:

        # 1. Buscar prompts que contengan la palabra
        candidatos = [
            prompt for prompt in docs_train
            if palabra in prompt
        ]

        # Elegir hasta 5 prompts distintos, priorizando los más cortos
        prompts = []
        vistos = set()

        for prompt in sorted(candidatos, key=len):
            clave = tuple(prompt)

            if clave not in vistos:
                prompts.append(prompt)
                vistos.add(clave)

            if len(prompts) == max_prompts:
                break

        if len(prompts) < 2:
            print(f"{palabra}: no hay suficientes prompts")
            continue

        # 2. Generar embeddings contextuales con ELMo
        embeddings = elmo.get_elmo_vectors(prompts)

        # 3. Extraer el embedding de la palabra en cada prompt
        vectores = []

        for i, prompt in enumerate(prompts):
            posicion = prompt.index(palabra)
            vectores.append(embeddings[i, posicion, :])

        vectores = np.array(vectores)

        # 4. Calcular similitud coseno entre todos los vectores
        similitudes = cosine_similarity(vectores)

        # 5. Calcular media, excluyendo diagonal y duplicados
        indices = np.triu_indices_from(similitudes, k=1)
        media = similitudes[indices].mean()

        resultados.append({
            "palabra": palabra,
            "n_prompts": len(prompts),
            "cosine_similarity_avg": media
        })

    return pd.DataFrame(resultados)



import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def calculate_elmo_similar_words(palabras, docs_train, domain_w2v, elmo):

    # 1. Obtener las 10 palabras similares según Word2Vec
    vecinos = {
        palabra: domain_w2v.wv.most_similar(palabra, topn=10)
        for palabra in palabras
    }

    # Reunir todas las palabras que necesitamos representar
    todas_palabras = set(palabras)

    for lista in vecinos.values():
        todas_palabras.update(p for p, similitud in lista)

    # 2. Seleccionar máximo 5 prompts por palabra
    contextos = {}

    for palabra in todas_palabras:

        prompts = [
            tuple(prompt) for prompt in docs_train
            if palabra in prompt and len(prompt) <= 150
        ]

        # Priorizar prompts cortos y sin duplicados
        contextos[palabra] = sorted(set(prompts), key=len)[:5]

    # 3. Agrupar prompts para evitar calcularlos varias veces
    prompt_palabras = {}

    for palabra, prompts in contextos.items():
        for prompt in prompts:
            prompt_palabras.setdefault(prompt, []).append(palabra)

    prompts_unicos = list(prompt_palabras)

    embeddings_palabras = {p: [] for p in todas_palabras}

    # 4. Generar embeddings contextuales con ELMo por lotes
    batch_size = 4

    for i in range(0, len(prompts_unicos), batch_size):

        batch = prompts_unicos[i:i + batch_size]

        embeddings = elmo.get_elmo_vectors(
            [list(prompt) for prompt in batch]
        )

        for j, prompt in enumerate(batch):
            for palabra in prompt_palabras[prompt]:

                posicion = prompt.index(palabra)
                vector = embeddings[j, posicion, :]

                if np.isfinite(vector).all():
                    embeddings_palabras[palabra].append(vector)

    # 5. Calcular el embedding medio de cada palabra
    medias = {}

    for palabra, vectores in embeddings_palabras.items():
        if len(vectores) > 0:
            medias[palabra] = np.mean(vectores, axis=0)

    # 6. Calcular similitud coseno entre palabra y candidatas
    resultados = []

    for palabra, candidatas in vecinos.items():

        for candidata, similitud_w2v in candidatas:

            similitud_elmo = np.nan

            if palabra in medias and candidata in medias:
                similitud_elmo = cosine_similarity(
                    medias[palabra].reshape(1, -1),
                    medias[candidata].reshape(1, -1)
                )[0, 0]

            resultados.append({
                "palabra": palabra,
                "candidata": candidata,
                "W2V_cosine": similitud_w2v,
                "ELMO_cosine": similitud_elmo,
                "n_prompts": len(embeddings_palabras[candidata])
            })

    return pd.DataFrame(resultados)