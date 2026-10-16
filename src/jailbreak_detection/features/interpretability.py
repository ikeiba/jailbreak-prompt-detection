import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np






def crear_pares_prompt(docs, splits):
    
    # ==========================================================
    # 1. CREAR DATAFRAME CON LOS PROMPTS REALES DE TRAIN
    # ==========================================================

    df_train = pd.DataFrame({
        "tokens": docs["train"].to_list(),
        "prompt_raw": splits["train"]["prompt_raw"].to_list(),
    })

    # Tokens OOV que queremos estudiar
    tokens_objetivo = {
        "ChatGPT", "OpenAI", "JailBreak", "alphabreak",
        "BetterDAN", "BasedGPT", "AntiGPT", "DMDAN",
        "UnleashedGPT", "RAYX", "role-play", "in-character",
        "step-by-step", "nsfw", "FILTERING", "<user>",
        "<prompt>", "<placeholder>", "<targetlanguage>", "🔓",
    }


    # ==========================================================
    # 2. FILTRAR PROMPTS CON ALGUNO DE LOS 20 OOV
    # ==========================================================

    df_oov = (
        df_train[
            df_train["tokens"].apply(
                lambda ts: bool(tokens_objetivo.intersection(ts))
            )
        ]
        .reset_index()
        .rename(columns={"index": "id_train"})
    )

    print("Prompts con OOV objetivo:", len(df_oov))


    # ==========================================================
    # 3. PAREJAS OOV SELECCIONADAS
    # ==========================================================

    # IMPORTANTE:
    # Las posiciones son filas del DataFrame filtrado df_oov.
    # Las etiquetas propuestas deben revisarse semánticamente.

    pares_oov = [
        # Similitud alta
        (5, 9, "alta"),
        (14, 67, "alta"),
        (75, 79, "alta"),
        (89, 223, "alta"),
        (209, 273, "alta"),
        (371, 189, "alta"),
        (595, 607, "alta"),
        (609, 610, "alta"),
        (1033, 3018, "alta"),
        (381, 382, "alta"),

        # Similitud baja
        (68, 2955, "baja"),
        (70, 1053, "baja"),
        (139, 733, "baja"),
        (299, 3153, "baja"),
        (3888, 2048, "baja"),
        (13, 2460, "baja"),
        (2498, 2345, "baja"),
        (3449, 3577, "baja"),
        (4340, 3201, "baja"),
        (1573, 1386, "baja"),
    ]

    posiciones = [
        pos
        for a, b, _ in pares_oov
        for pos in (a, b)
    ]

    if max(posiciones) >= len(df_oov):
        raise ValueError(
            "Las posiciones seleccionadas no coinciden "
            "con el DataFrame OOV."
        )


    # ==========================================================
    # 4. FUNCIÓN PARA REGISTRAR CADA PAREJA
    # ==========================================================

    def registrar_par(
        id_a, id_b, grupo, etiqueta,
        revision, similitud_tfidf=None
    ):

        fila_a = df_train.iloc[id_a]
        fila_b = df_train.iloc[id_b]

        return {
            "id_train_a": id_a,
            "id_train_b": id_b,

            "prompt_a": fila_a["prompt_raw"],
            "prompt_b": fila_b["prompt_raw"],

            "tokens_a": fila_a["tokens"],
            "tokens_b": fila_b["tokens"],

            "grupo": grupo,
            "similitud_esperada": (
                etiqueta if revision == "revisada" else pd.NA
            ),
            "etiqueta_provisional": etiqueta,

            "similitud_tfidf": similitud_tfidf,
            "revision": revision,
        }


    # ==========================================================
    # 5. RECUPERAR LOS 20 PARES OOV DESDE TRAIN
    # ==========================================================

    resultados = []
    utilizados = set()

    for pos_a, pos_b, etiqueta in pares_oov:

        id_a = int(df_oov.iloc[pos_a]["id_train"])
        id_b = int(df_oov.iloc[pos_b]["id_train"])

        if id_a == id_b:
            raise ValueError("Una pareja contiene el mismo prompt.")

        if id_a in utilizados or id_b in utilizados:
            raise ValueError("Un prompt aparece en varias parejas.")

        utilizados.update([id_a, id_b])

        resultados.append(
            registrar_par(
                id_a,
                id_b,
                "con OOV",
                etiqueta,
                "pendiente de revisión semántica",
            )
        )


    # ==========================================================
    # 6. BUSCAR PROMPTS SIN LOS 20 OOV OBJETIVO
    # ==========================================================

    # CORRECCIÓN PRINCIPAL:
    # No exigimos ausencia total de OOV de Google News.
    # Solo que NO contengan los 20 tokens seleccionados.

    sin_oov = df_train[
        df_train["tokens"].apply(
            lambda ts:
                15 <= len(ts) <= 140
                and not bool(tokens_objetivo.intersection(ts))
        )
    ].copy()

    print("Prompts sin OOV objetivo:", len(sin_oov))

    if len(sin_oov) < 20:
        raise ValueError(
            f"Solo existen {len(sin_oov)} prompts candidatos."
        )


    # ==========================================================
    # 7. CALCULAR SIMILITUD TF-IDF PARA PROPONER PAREJAS no oov
    # ==========================================================

    muestra = sin_oov.sample(
        n=min(750, len(sin_oov)),
        random_state=42
    )

    ids = muestra.index.to_list()
    listas = muestra["tokens"].to_list()

    textos = [
        " ".join(tokens)
        for tokens in listas
    ]

    vectorizador = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    X = vectorizador.fit_transform(textos)

    matriz = cosine_similarity(X)


    # ==========================================================
    # 8. GENERAR CANDIDATOS
    # ==========================================================

    candidatos = []

    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):

            # Evitar prompts idénticos
            if listas[a] == listas[b]:
                continue

            largo_a = len(listas[a])
            largo_b = len(listas[b])

            # Evitar grandes diferencias de longitud
            if max(largo_a, largo_b) / min(largo_a, largo_b) > 1.6:
                continue

            candidatos.append((
                float(matriz[a, b]),
                ids[a],
                ids[b]
            ))


    # ==========================================================
    # 9. SELECCIONAR 5 PARES ALTOS Y 5 BAJOS
    # ==========================================================

    for etiqueta, descendente in [
        ("alta", True),
        ("baja", False)
    ]:

        elegidos = 0

        candidatos_ordenados = sorted(
            candidatos,
            key=lambda x: x[0],
            reverse=descendente
        )

        for puntuacion, id_a, id_b in candidatos_ordenados:

            if id_a in utilizados or id_b in utilizados:
                continue

            utilizados.update([id_a, id_b])

            resultados.append(
                registrar_par(
                    id_a,
                    id_b,
                    "sin OOV objetivo",
                    etiqueta,
                    "pendiente de revisión semántica",
                    puntuacion
                )
            )

            elegidos += 1

            if elegidos == 5:
                break

        if elegidos < 5:
            raise ValueError(
                f"Solo se encontraron {elegidos} parejas {etiqueta}."
            )


    # ==========================================================
    # 10. DATAFRAME FINAL CON LOS 30 PARES
    # ==========================================================

    df_pares = pd.DataFrame(resultados)

    df_pares.insert(
        0,
        "id_par",
        range(1, len(df_pares) + 1)
    )

    print("Total de parejas:", len(df_pares))

    print(
        df_pares.groupby(
            ["grupo", "etiqueta_provisional"]
        ).size()
    )

    return(df_pares)
