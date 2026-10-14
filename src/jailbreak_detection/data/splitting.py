import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def split_by_groups(df: pd.DataFrame, group_col: str = "group_id", n_folds: int = 20,
                    n_train: int = 14, n_val: int = 3, seed: int = 42) -> pd.DataFrame:
    """Split 70/15/15 estratificado por etiqueta. Todos los prompts de un grupo van a la
    misma partición. Se reparten 20 folds de ~5 %: 14 a train, 3 a val y 3 a test."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=int)
    for f, (_, idx) in enumerate(sgkf.split(df, df["jailbreak"], df[group_col])):
        fold[idx] = f
    out = df.copy()
    out["split"] = np.select([fold < n_train, fold < n_train + n_val],
                             ["train", "val"], default="test")
    return out


def dedup_eval_near_duplicates(df: pd.DataFrame, near_dup_col: str = "near_dup_id"
                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """En val y test conserva un representante (el primero) por grupo de casi idénticos.
    Train se conserva entero. Devuelve (df_final, filas_descartadas)."""
    drop = (df["split"] != "train") & df.duplicated(["split", near_dup_col], keep="first")
    return df[~drop].copy(), df[drop].copy()