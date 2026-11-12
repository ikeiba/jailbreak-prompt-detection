"""Entrenamiento con early stopping sobre MCC de validación, ajuste de umbral, checkpoints
reanudables y registro en TensorBoard."""

import json
import random
import shutil
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, precision_score,
                             recall_score)
from torch import nn

from jailbreak_detection.models.data import PromptDataset, collate, make_loader
from jailbreak_detection.models.embeddings import EMBEDDINGS, InputPipeline
from jailbreak_detection.models.encoder import GRUEncoderClassifier

PROB_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


@dataclass(frozen=True)
class ExperimentConfig:
    embedding: str
    use_attention: bool = True
    truncation: str = "head_tail"
    max_len: int = 1024
    hidden_size: int = 128
    dropout: float = 0.3
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    clip_norm: float = 1.0
    max_epochs: int = 30
    patience: int = 5
    seed: int = 42

    @property
    def name(self) -> str:
        attn = "attn" if self.use_attention else "noattn"
        return f"{self.embedding}__{attn}__{self.truncation}__L{self.max_len}"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(y_true, y_pred) -> dict:
    """Mismas métricas que el notebook 04 (clase positiva = jailbreak)."""
    return {
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def tune_threshold(y_true, scores, grid=PROB_GRID, default: float = 0.5) -> tuple[float, float]:
    """Umbral que maximiza MCC; en caso de empate, el más cercano a default."""
    y_true = np.asarray(y_true)
    best = max(grid, key=lambda t: (round(matthews_corrcoef(y_true, scores >= t), 12),
                                    -abs(t - default)))
    return float(best), float(matthews_corrcoef(y_true, scores >= best))


def build_model(cfg: ExperimentConfig, pipeline: InputPipeline) -> GRUEncoderClassifier:
    return GRUEncoderClassifier(
        input_size=pipeline.input_size, hidden_size=cfg.hidden_size,
        use_attention=cfg.use_attention, dropout=cfg.dropout,
        embedding_matrix=pipeline.matrix, freeze_embeddings=pipeline.freeze)


@torch.no_grad()
def predict(model, pipeline: InputPipeline, dataset: PromptDataset, device, batch_size: int = 32,
            criterion=None, what: str = "probs"):
    """Recorre dataset en orden de longitud y devuelve los resultados en el orden original.
    what: "probs" -> (probabilidades, pérdida media); "vectors" -> vectores (N, 256)."""
    model.eval()
    loader = make_loader(dataset, batch_size, shuffle=False)
    out = np.zeros((len(dataset), model.output_dim) if what == "vectors" else len(dataset),
                   dtype=np.float32)
    loss_sum = 0.0
    for batch in loader:
        inputs = pipeline.to_device(batch["inputs"], device)
        lengths = batch["lengths"]
        if what == "vectors":
            out[batch["index"].numpy()] = model.encode(inputs, lengths).cpu().numpy()
            continue
        logits = model(inputs, lengths)
        out[batch["index"].numpy()] = torch.sigmoid(logits).cpu().numpy()
        if criterion is not None:
            loss_sum += criterion(logits, batch["labels"].to(device)).item() * len(lengths)
    if what == "vectors":
        return out
    return out, loss_sum / len(dataset)


@torch.no_grad()
def attention_weights(model, pipeline: InputPipeline, dataset: PromptDataset, indices, device):
    """Pesos de atención (uno por token real) y probabilidad de los items indicados."""
    model.eval()
    results = []
    for i in indices:
        batch = collate([dataset[i]])
        n = int(batch["lengths"][0])
        logits, weights = model(pipeline.to_device(batch["inputs"], device), batch["lengths"],
                                return_attention=True)
        results.append((weights[0, :n].cpu().numpy(), torch.sigmoid(logits)[0].item()))
    return results


def load_model(exp_dir: str | Path, pipeline: InputPipeline, device) -> GRUEncoderClassifier:
    ckpt = torch.load(Path(exp_dir) / "model.pt", map_location="cpu", weights_only=False)
    cfg = ExperimentConfig(**ckpt["config"])
    model = build_model(cfg, pipeline)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def load_result(exp_dir: str | Path) -> dict | None:
    path = Path(exp_dir) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def best_by_val_mcc(results: dict[str, dict]) -> str:
    """Nombre del experimento con mayor MCC en validación completa (umbral ajustado).
    Con empate gana el primero en el orden de results."""
    return max(results, key=lambda name: results[name]["val_tuned"]["MCC"])


STEP1_EMBEDDINGS = tuple(EMBEDDINGS)  # paso 1: las cuatro variantes con la configuración base


def step1_winner(out_root: str | Path) -> str | None:
    """Embedding ganador del paso 1 a partir de los resultados guardados (None si falta alguno)."""
    results = {}
    for emb in STEP1_EMBEDDINGS:
        result = load_result(Path(out_root) / ExperimentConfig(embedding=emb).name)
        if result is None:
            return None
        results[emb] = result
    return best_by_val_mcc(results)


def summary_writer(log_dir: str | Path):
    """SummaryWriter de PyTorch sin importar TensorFlow.

    tensorboard.compat importa TensorFlow si está instalado (lo está, por simple_elmo), salvo que
    exista el módulo tensorboard.compat.notf: en ese caso usa su sustituto interno. Así el
    entrenamiento no carga TensorFlow (ni lo pone a competir por la GPU)."""
    sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=str(log_dir))


def run_experiment(cfg: ExperimentConfig, pipeline: InputPipeline, train_df, val_df,
                   pos_weight: float, out_root: str | Path, tb_root: str | Path, device,
                   log=print) -> dict:
    """Entrena una configuración. Si ya tiene metrics.json en disco no hace nada y lo devuelve.

    metrics.json se escribe al final, después del checkpoint: si un entrenamiento se corta,
    la siguiente ejecución lo repite desde el principio (y borra sus logs incompletos)."""
    exp_dir = Path(out_root) / cfg.name
    done = load_result(exp_dir)
    if done is not None:
        log(f"[{cfg.name}] ya entrenado (mejor época {done['best_epoch']}, "
            f"MCC val {done['val_tuned']['MCC']:.4f}); se salta.")
        return done

    set_seed(cfg.seed)
    exp_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(tb_root) / cfg.name
    shutil.rmtree(tb_dir, ignore_errors=True)
    writer = summary_writer(tb_dir)

    train_ds = pipeline.dataset(train_df, cfg.max_len, cfg.truncation)
    val_ds = pipeline.dataset(val_df, cfg.max_len, cfg.truncation)
    y_val = val_ds.labels.astype(int)
    train_loader = make_loader(train_ds, cfg.batch_size, shuffle=True, seed=cfg.seed)

    model = build_model(cfg, pipeline).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_trainable = sum(p.numel() for p in params)
    log(f"[{cfg.name}] train={len(train_ds)} val={len(val_ds)} | parámetros entrenables: "
        f"{n_trainable:,} | dispositivo: {device}")

    history, best_mcc, best_epoch, best_state, waited = [], -np.inf, 0, None, 0
    start = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            inputs = pipeline.to_device(batch["inputs"], device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs, batch["lengths"]), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.clip_norm)
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        train_loss = loss_sum / len(train_ds)

        val_probs, val_loss = predict(model, pipeline, val_ds, device, cfg.batch_size, criterion)
        m = evaluate(y_val, val_probs >= 0.5)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **m})
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        for k in ("MCC", "Precision", "Recall", "F1"):
            writer.add_scalar(f"Val@0.5/{k}", m[k], epoch)
        writer.flush()

        improved = m["MCC"] > best_mcc
        if improved:
            best_mcc, best_epoch, waited = m["MCC"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
        log(f"[{cfg.name}] época {epoch:02d} | loss train {train_loss:.4f} val {val_loss:.4f} | "
            f"MCC {m['MCC']:.4f} P {m['Precision']:.3f} R {m['Recall']:.3f} F1 {m['F1']:.3f} | "
            f"{time.time() - t0:.0f}s{' *' if improved else ''}")
        if waited >= cfg.patience:
            log(f"[{cfg.name}] early stopping: {cfg.patience} épocas sin mejorar.")
            break

    model.load_state_dict(best_state)
    val_probs, _ = predict(model, pipeline, val_ds, device, cfg.batch_size)
    threshold, _ = tune_threshold(y_val, val_probs)
    result = {
        "name": cfg.name,
        "config": asdict(cfg),
        "pos_weight": pos_weight,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "threshold": threshold,
        "val_at_05": evaluate(y_val, val_probs >= 0.5),
        "val_tuned": evaluate(y_val, val_probs >= threshold),
        "history": history,
        "train_seconds": time.time() - start,
        "device": str(device),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    torch.save({"state_dict": best_state, "config": asdict(cfg),
                "input_size": pipeline.input_size}, exp_dir / "model.pt")
    np.save(exp_dir / "val_probs.npy", val_probs)

    tuned = result["val_tuned"]
    writer.add_hparams(
        {"embedding": cfg.embedding, "attention": cfg.use_attention,
         "truncation": cfg.truncation, "max_len": cfg.max_len, "lr": cfg.lr,
         "weight_decay": cfg.weight_decay, "batch_size": cfg.batch_size,
         "dropout": cfg.dropout, "hidden_size": cfg.hidden_size, "pos_weight": pos_weight},
        {"hparam/val_MCC": tuned["MCC"], "hparam/val_Precision": tuned["Precision"],
         "hparam/val_Recall": tuned["Recall"], "hparam/val_F1": tuned["F1"],
         "hparam/val_MCC_at_0.5": result["val_at_05"]["MCC"],
         "hparam/best_epoch": best_epoch, "hparam/threshold": threshold},
        run_name=".",
    )
    writer.close()

    tmp = exp_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    tmp.replace(exp_dir / "metrics.json")
    log(f"[{cfg.name}] mejor época {best_epoch} | umbral {threshold:.2f} | MCC val "
        f"{tuned['MCC']:.4f} (a 0.5: {result['val_at_05']['MCC']:.4f})")
    return result
