"""Evaluasi model pada data validasi: confusion matrix, laporan per kelas, pasangan kelas tertukar.

Contoh:
    python -m scripts.evaluate --checkpoint checkpoint/resnet18_160_best.pt
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from scripts.config import CLASS_NAMES, NUM_CLASSES, Config
from scripts.dataset import make_eval_loader
from scripts.model import load_model_from_checkpoint
from scripts.utils import get_device


@torch.no_grad()
def collect_val_predictions(ckpt_path: Path, cfg: Config,
                            device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Prediksi seluruh data validasi yang dipakai saat training checkpoint tersebut.

    Indeks validasi dibaca dari metadata checkpoint, sehingga evaluasi selalu memakai
    split yang sama (termasuk jika training memakai --subset).

    Args:
        ckpt_path: Path checkpoint.
        cfg: Objek Config.
        device: Device komputasi.

    Returns:
        Tuple (prediksi, target, val_idx, meta).
    """
    model, meta = load_model_from_checkpoint(ckpt_path, device)
    cfg_local = replace(cfg, image_size=meta["image_size"])
    labels = pd.read_csv(cfg.train_csv)["label"].to_numpy()
    val_idx = np.asarray(meta["val_idx"])
    loader = make_eval_loader(cfg_local, val_idx, labels)
    preds, targets = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(dim=1).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets), val_idx, meta


def top_confusions(cm: np.ndarray, names: Sequence[str], k: int = 10) -> List[Dict]:
    """Daftar pasangan (kelas benar -> kelas prediksi) yang paling sering salah.

    Args:
        cm: Confusion matrix (baris = kelas benar, kolom = prediksi).
        names: Nama kelas berurutan sesuai indeks.
        k: Jumlah pasangan teratas.

    Returns:
        List dict berisi true, pred, count, rate (proporsi dari seluruh sampel kelas benar).
    """
    off = cm.copy()
    np.fill_diagonal(off, 0)  # abaikan prediksi benar
    order = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0][:k]
    row_sums = cm.sum(axis=1)
    return [{"true": names[i], "pred": names[j], "count": int(off[i, j]),
             "rate": float(off[i, j] / row_sums[i])} for i, j in order]


def plot_confusion_matrix(cm: np.ndarray, names: Sequence[str], out_path: Path) -> None:
    """Gambar confusion matrix ternormalisasi per baris (recall per kelas pada diagonal).

    Args:
        cm: Confusion matrix mentah.
        names: Nama kelas.
        out_path: Lokasi file gambar.
    """
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if norm[i, j] >= 0.05:  # hanya anotasi sel yang bermakna agar tidak ramai
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Kelas sebenarnya")
    ax.set_title("Confusion matrix (ternormalisasi per baris)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_history(history: Dict[str, List[float]], out_path: Path) -> None:
    """Gambar kurva loss dan akurasi train vs validasi.

    Args:
        history: Dict riwayat dari train.fit.
        out_path: Lokasi file gambar.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, title in zip(axes, ("loss", "acc"), ("Loss", "Akurasi")):
        ax.plot(epochs, history[f"train_{key}"], marker="o", label="train", color="#4C72B0")
        ax.plot(epochs, history[f"val_{key}"], marker="o", label="validasi", color="#DD8452")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_summary(path: Path, model_name: str, meta: Dict, report: Dict, confusions: List[Dict]) -> None:
    """Tulis ringkasan hasil dalam Markdown (siap ditempel ke README).

    Args:
        path: Lokasi file .md.
        model_name: Nama checkpoint/arsitektur.
        meta: Metadata checkpoint.
        report: Output classification_report(output_dict=True).
        confusions: Output top_confusions.
    """
    lines = [f"Model: {model_name}, ukuran input {meta['image_size']} piksel, "
             f"epoch terbaik {meta['epoch']}.", "",
             f"Akurasi validasi: {report['accuracy']:.4f}", "",
             "| Kelas | Precision | Recall | F1 | Jumlah |", "|---|---|---|---|---|"]
    for name in [CLASS_NAMES[i] for i in range(NUM_CLASSES)]:
        r = report[name]
        lines.append(f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |")
    lines += ["", "Pasangan kelas yang paling sering tertukar:", "",
              "| Benar | Diprediksi | Jumlah | Proporsi kelas benar |", "|---|---|---|---|"]
    for c in confusions:
        lines.append(f"| {c['true']} | {c['pred']} | {c['count']} | {c['rate']:.3f} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Entry point CLI evaluasi."""
    ap = argparse.ArgumentParser(description="Evaluasi checkpoint pada data validasi.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    args = ap.parse_args()

    cfg = Config()
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    preds, targets, _, meta = collect_val_predictions(args.checkpoint, cfg, device)
    names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    stem = args.checkpoint.stem.replace("_best", "")

    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))
    plot_confusion_matrix(cm, names, cfg.figures_dir / f"confusion_matrix_{stem}.png")
    report = classification_report(targets, preds, labels=list(range(NUM_CLASSES)),
                                   target_names=names, output_dict=True, zero_division=0)
    confusions = top_confusions(cm, names, k=10)
    write_summary(cfg.figures_dir.parent / f"results_{stem}.md", stem, meta, report, confusions)

    hist_path = args.checkpoint.with_name(f"{stem}_history.json")
    if hist_path.exists():
        plot_history(json.loads(hist_path.read_text()), cfg.figures_dir / f"history_{stem}.png")
    print(f"Akurasi validasi: {report['accuracy']:.4f}. Output di {cfg.figures_dir.parent}")


if __name__ == "__main__":
    main()
