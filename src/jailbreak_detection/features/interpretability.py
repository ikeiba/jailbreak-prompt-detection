import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


import pandas as pd
import random

import pandas as pd


import pandas as pd
from pathlib import Path


import pandas as pd
from pathlib import Path


def crear_pares_prompt(docs_train):
    """Reconstruye las 30 parejas seleccionadas usando solo docs['train'].

    docs_train debe contener los prompts tokenizados en el MISMO ORDEN
    utilizado para seleccionar los id_train_a/id_train_b.
    """
    train = list(docs_train)

    # (id_train_a, id_train_b, etiqueta, similitud_W2V, similitud_ELMo)
    # Las filas 1-20 contienen alguno de los OOV objetivo.
    # Las filas 21-30 no contienen ninguno de esos 20 OOV objetivo.
    pares = [
        # 10 parejas con OOV: similitud alta
        (11, 19, "alta", 0.9982642531394958, 0.9921318933795478),
        (26, 142, "alta", 0.9980965852737427, 0.9949642383974677),
        (158, 162, "alta", 0.9647342562675476, 0.9499964950198982),
        (177, 394, "alta", 0.9612483978271484, 0.9765525967892652),
        (374, 499, "alta", 0.9954546093940735, 0.8451978036481178),
        (666, 340, "alta", 0.9036819338798523, 0.7987054126754269),
        (1212, 1389, "alta", 0.9722398519515991, 0.7492026938097371),
        (1462, 1464, "alta", 0.93635094165802, 0.7846243829806672),
        (1894, 4934, "alta", 0.9915081262588501, 0.9935806353218961),
        (683, 684, "alta", 0.9948294162750244, 0.9854527513465675),
        # 10 parejas con OOV: similitud baja
        (143, 4871, "baja", 0.9354937076568604, 0.7868261980997815),
        (145, 1914, "baja", 0.8723940253257751, 0.7250920278304076),
        (254, 1594, "baja", 0.7953108549118042, 0.7132826909000225),
        (546, 5069, "baja", 0.8700703978538513, 0.6290854219949832),
        (6598, 2910, "baja", 0.807483971118927, 0.8567845563349973),
        (23, 3322, "baja", 0.937744677066803, 0.8180758682741378),
        (3360, 3207, "baja", 0.7987642884254456, 0.6023822650617873),
        (5743, 6012, "baja", 0.8351823687553406, 0.7061965159254051),
        (7483, 5179, "baja", 0.7985346913337708, 0.5180294092699328),
        (2434, 2247, "baja", 0.6194079518318176, 0.6997382869402341),
        # 5 parejas sin OOV objetivo: similitud baja
        (320, 8171, "baja", 0.708224356174469, 0.5894235802017901),
        (3897, 3746, "baja", 0.600167989730835, 0.5504223177838745),
        (3650, 1307, "baja", 0.9289341568946838, 0.7033822912247293),
        (5667, 1146, "baja", 0.7801335453987122, 0.6545956240836864),
        (5852, 1440, "baja", 0.7326743602752686, 0.5038083292690936),
        # 5 parejas sin OOV objetivo: similitud alta provisional
        (393, 358, "alta", 0.8863129019737244, 0.6509849516098267),
        (3981, 1076, "alta", 0.8925866484642029, 0.7694087821333596),
        (1288, 4360, "alta", 0.8464011549949646, 0.8085160740424491),
        (7377, 8355, "alta", 0.9250538349151611, 0.7825116228450655),
        (7613, 7636, "alta", 0.8606777787208557, 0.7863625658700377),
    ]

    revisiones = {
        26: "alta propuesta tras revisión semántica",
        27: "alta propuesta, caso dudoso pendiente de confirmar",
        28: "alta propuesta, revisar diferencias de requisitos",
        29: "alta propuesta tras revisión semántica",
        30: "alta provisional, similitud en intención de jailbreak",
    }

    max_id = max(max(a, b) for a, b, *_ in pares)
    if max_id >= len(train):
        raise ValueError(
            f"docs_train tiene {len(train)} prompts, pero se necesita "
            f"el índice {max_id}. Comprueba que estás usando el train original."
        )

    resultados = []
    for id_par, (id_a, id_b, etiqueta, sim_w2v, sim_elmo) in enumerate(
        pares, start=1
    ):
        tokens_a = list(train[id_a])
        tokens_b = list(train[id_b])

        if not all(isinstance(token, str) for token in tokens_a + tokens_b):
            raise TypeError("docs_train debe contener listas de tokens (strings).")

        if id_par <= 20:
            revision = "pendiente de revisión semántica"
        elif id_par <= 25:
            revision = "etiqueta propuesta tras revisión semántica"
        else:
            revision = revisiones[id_par]

        resultados.append({
            "id_par": id_par,
            "id_train_a": id_a,
            "id_train_b": id_b,
            "prompt_a": " ".join(tokens_a),
            "prompt_b": " ".join(tokens_b),
            "tokens_a": tokens_a,
            "tokens_b": tokens_b,
            "grupo": "con OOV" if id_par <= 20 else "sin OOV objetivo",
            "similitud_esperada": pd.NA,
            "etiqueta_provisional": etiqueta,
            "revision": revision,
            "W2V_cosine_similarity": sim_w2v,
            "ELMO_cosine_similarity": sim_elmo,
        })

    df_pares_finales = pd.DataFrame(resultados)
    if (
        len(df_pares_finales) != 30
        or (df_pares_finales["etiqueta_provisional"] == "alta").sum() != 15
        or (df_pares_finales["etiqueta_provisional"] == "baja").sum() != 15
    ):
        raise ValueError("La distribución final no es 15 altas y 15 bajas.")

    ruta_salida = Path("../data/processed/df_pares_finales.csv")
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_pares_finales.to_csv(ruta_salida, index=False)

    print(f"Guardadas {len(df_pares_finales)} parejas en {ruta_salida}")
    return df_pares_finales
