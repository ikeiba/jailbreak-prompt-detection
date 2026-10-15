
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