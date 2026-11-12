"""Análisis del modelo final: métricas por tramos de longitud y visualización de la atención."""

import html

import numpy as np
import pandas as pd

from jailbreak_detection.models.training import evaluate

LENGTH_BINS = [0, 64, 256, 512, 1024, np.inf]
LENGTH_LABELS = ["1-64", "65-256", "257-512", "513-1024", ">1024"]


def metrics_by_length(n_tokens, y_true, y_pred, bins=LENGTH_BINS, labels=LENGTH_LABELS) -> pd.DataFrame:
    """Métricas por tramo de longitud original (en tokens, antes de truncar).
    Recall = jailbreaks detectados en el tramo; FPR = normales marcados como jailbreak."""
    df = pd.DataFrame({"n_tokens": np.asarray(n_tokens), "y": np.asarray(y_true).astype(int),
                       "pred": np.asarray(y_pred).astype(int)})
    df["Tramo"] = pd.cut(df["n_tokens"], bins=bins, labels=labels, right=True)
    rows = []
    for label, g in df.groupby("Tramo", observed=False):
        if g.empty:
            continue
        neg = g[g["y"] == 0]
        m = evaluate(g["y"], g["pred"]) if g["y"].nunique() > 1 else {
            k: np.nan for k in ("F1", "Accuracy", "Precision", "Recall", "MCC")}
        if g["y"].nunique() == 1 and g["y"].iloc[0] == 1:
            m["Recall"] = (g["pred"] == 1).mean()
        rows.append({"Tramo": label, "n": len(g), "Jailbreaks": int(g["y"].sum()),
                     "MCC": m["MCC"], "Precision": m["Precision"], "Recall": m["Recall"],
                     "F1": m["F1"], "FPR normales": (neg["pred"] == 1).mean() if len(neg) else np.nan})
    return pd.DataFrame(rows)


def attention_html(tokens, weights, unk_mask=None, title: str = "", max_tokens: int = 400,
                   gap_after: int | None = None, top_k: int = 10) -> str:
    """Texto con fondo proporcional al peso de atención (normalizado al máximo del prompt).
    unk_mask marca (subrayado) los tokens que el modelo vio como <unk>.
    gap_after indica dónde se cortó el prompt por truncado principio + final."""
    weights = np.asarray(weights, dtype=float)
    scale = weights.max() if weights.max() > 0 else 1.0
    spans = []
    shown = range(len(tokens)) if len(tokens) <= max_tokens else \
        list(range(max_tokens // 2)) + [None] + list(range(len(tokens) - max_tokens // 2, len(tokens)))
    for i in shown:
        if i is None:
            spans.append("<b>[… omitido en la visualización …]</b>")
            continue
        alpha = weights[i] / scale
        style = f"background: rgba(220, 50, 47, {alpha:.2f}); padding: 1px 2px; border-radius: 2px;"
        if unk_mask is not None and unk_mask[i]:
            style += " text-decoration: underline dotted;"
        spans.append(f'<span title="{weights[i]:.4f}" style="{style}">{html.escape(tokens[i])}</span>')
        if gap_after is not None and i == gap_after - 1:
            spans.append("<b>[… truncado …]</b>")
    top = np.argsort(-weights)[:top_k]
    top_txt = ", ".join(f"{html.escape(tokens[i])} ({weights[i]:.3f})" for i in top)
    return (f'<div style="margin: 12px 0; font-family: monospace; font-size: 12px; line-height: 1.7;">'
            f"<div><b>{html.escape(title)}</b></div><div><i>Top {top_k}:</i> {top_txt}</div>"
            f'<div style="border: 1px solid #ccc; padding: 6px;">{" ".join(spans)}</div></div>')
