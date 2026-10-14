import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def find_similar_pairs(df: pd.DataFrame, min_jaccard: float = 0.7, chunk: int = 1000) -> pd.DataFrame:
    """Devuelve todos los pares de prompts con Jaccard de conjuntos de palabras >= min_jaccard.
    Jaccard = palabras compartidas / palabras totales entre los dos.
    Columnas: id_a, id_b, jaccard (cada par aparece una sola vez)."""
    B = (CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b")
         .fit_transform(df["prompt"]).tocsr().astype(np.float32))
    sizes = np.asarray(B.sum(axis=1)).ravel()   # nº de palabras distintas por prompt
    ids = df["id"].to_numpy()

    out = []
    for start in range(0, B.shape[0], chunk):
        inter = (B[start:start + chunk] @ B.T).tocoo()   # palabras compartidas
        i, j = inter.row + start, inter.col
        jac = inter.data / (sizes[i] + sizes[j] - inter.data)
        m = (i < j) & (jac >= min_jaccard)
        out.append(pd.DataFrame({"id_a": ids[i[m]], "id_b": ids[j[m]], "jaccard": jac[m]}))
    return pd.concat(out, ignore_index=True)

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def assign_groups(df: pd.DataFrame, pairs: pd.DataFrame, threshold: float) -> pd.Series:
    """Asigna un id de grupo a cada prompt: dos prompts van al mismo grupo si están
    enlazados (directa o indirectamente) por pares con Jaccard >= threshold.
    Los prompts sin pares reciben un grupo propio. `pairs` debe haberse calculado
    sobre este mismo df."""
    pos = pd.Series(np.arange(len(df)), index=df["id"].to_numpy())
    p = pairs[pairs["jaccard"] >= threshold]
    a = pos.loc[p["id_a"]].to_numpy()
    b = pos.loc[p["id_b"]].to_numpy()
    n = len(df)
    graph = coo_matrix((np.ones(len(p)), (a, b)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return pd.Series(labels, index=df.index)