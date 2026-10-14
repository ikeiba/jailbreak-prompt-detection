from pathlib import Path

import pandas as pd
from datasets import load_dataset

REPO = "TrustAIRLab/in-the-wild-jailbreak-prompts"
CONFIGS = {"jailbreak": "jailbreak_2023_12_25", "regular": "regular_2023_12_25"}
KEEP_COLS = ["id", "prompt", "jailbreak", "platform", "source", "date"]


def load_raw(save_to: str | Path | None = None) -> pd.DataFrame:
    jb = load_dataset(REPO, CONFIGS["jailbreak"], split="train").to_pandas()
    reg = load_dataset(REPO, CONFIGS["regular"], split="train").to_pandas()

    assert jb["jailbreak"].all() and not reg["jailbreak"].any()

    df = pd.concat([jb, reg], ignore_index=True)
    df.insert(0, "id", range(len(df)))

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_to, index=False)
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[KEEP_COLS].copy()