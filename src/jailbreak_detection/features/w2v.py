


import numpy as np
from gensim.models import Word2Vec

def calculate_oov(prompts, model):
    """Devuelve los tokens fuera del vocabulario y su porcentaje de apariciones."""
    not_represented_words = []
    total_tokens = 0
    for prompt in prompts:
        for token in prompt:
            total_tokens += 1
            if token not in model.key_to_index:
                not_represented_words.append(token)


    percentage = 100 * len(not_represented_words) / total_tokens if total_tokens else 0.0
    return not_represented_words, percentage




def train_domain_w2v(docs, model):


    # train_tokens es una lista de listas de tokens
    # Ejemplo: [["ChatGPT", "ignore", "all"], [...], ...]

    # PASO 1: crear un Word2Vec nuevo
    domain_w2v = Word2Vec(
        vector_size=300,
        window=5,
        min_count=1,
        sg=1,
        workers=1,
        seed=42
    )

    google_w2v = model
    # PASO 2: construir el vocabulario SOLO con train
    domain_w2v.build_vocab(docs["train"])

    # PASO 3: crear un control individual para cada embedding
    domain_w2v.wv.vectors_lockf = np.ones(
        len(domain_w2v.wv),
        dtype=np.float32
    )

    # PASO 4: copiar y congelar los embeddings conocidos
    n_conocidas = 0
    n_oov = 0

    for word in domain_w2v.wv.index_to_key:

        idx = domain_w2v.wv.key_to_index[word]

        if word in google_w2v.key_to_index:

            # Copiar el vector de Google News
            domain_w2v.wv.vectors[idx] = google_w2v[word].copy()

            # 0.0 = NO entrenar ese embedding
            domain_w2v.wv.vectors_lockf[idx] = 0.0

            n_conocidas += 1

        else:
            # 1.0 = SÍ entrenar (inicialización aleatoria)
            domain_w2v.wv.vectors_lockf[idx] = 1.0
            n_oov += 1

    print("Embeddings congelados:", n_conocidas)
    print("Embeddings nuevos entrenables:", n_oov)

    # PASO 5: entrenar con TODOS los prompts de train
    domain_w2v.train(
        docs["train"],
        total_examples=domain_w2v.corpus_count,
        epochs=10
    )
    return domain_w2v
