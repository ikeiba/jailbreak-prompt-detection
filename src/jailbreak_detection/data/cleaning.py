import pandas as pd


def normalize_key(text: str) -> str:
    """Clave solo para comparar: minúsculas y espacios colapsados.
    El texto original no se modifica."""
    return " ".join(text.lower().split())


def remove_contradictory(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Elimina todas las filas cuyo prompt (normalizado) aparece con ambas etiquetas.
    Devuelve (df_limpio, filas_eliminadas)."""
    key = df["prompt"].map(normalize_key)
    n_labels = df.groupby(key)["jailbreak"].transform("nunique")
    mask = n_labels > 1
    return df[~mask].copy(), df[mask].copy()


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conserva la primera aparición de cada prompt (normalizado) y añade
    `n_copies` con el nº de veces que aparecía. Devuelve (df_limpio, filas_eliminadas).
    Debe ejecutarse después de remove_contradictory."""
    key = df["prompt"].map(normalize_key)
    n_copies = key.map(key.value_counts())
    dup_mask = key.duplicated(keep="first")

    out = df[~dup_mask].copy()
    out["n_copies"] = n_copies[~dup_mask]
    return out, df[dup_mask].copy()

def remove_mixed_near_duplicates(df: pd.DataFrame, near_dup_col: str = "near_dup_id"
                                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Elimina todas las filas de los grupos de casi-idénticos (Jaccard >= 0,95)
    que contienen ambas etiquetas. Devuelve (df_limpio, filas_eliminadas)."""
    mask = df.groupby(near_dup_col)["jailbreak"].transform("nunique") > 1
    return df[~mask].copy(), df[mask].copy()


def flag_mixed_groups(df: pd.DataFrame, group_col: str = "group_id") -> pd.DataFrame:
    """Añade `group_mixed`: True si el grupo de plantilla (Jaccard >= 0,80) contiene
    ambas etiquetas. No elimina nada."""
    out = df.copy()
    out["group_mixed"] = out.groupby(group_col)["jailbreak"].transform("nunique") > 1
    return out