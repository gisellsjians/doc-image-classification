"""Inferensi pada data test dan pembuatan file submission Kaggle.

Contoh:
    python -m scripts.predict --checkpoints checkpoint/resnet18_160_best.pt
    python -m scripts.predict --checkpoints checkpoint/a.pt checkpoint/b.pt   # ensemble
"""

import argparse
from dataclasses import replace
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from scripts.config import NUM_CLASSES, Config
from scripts.dataset import make_test_loader
from scripts.model import load_model_from_checkpoint
from scripts.utils import get_device


@torch.no_grad()
def predict_probs(model: torch.nn.Module, loader, device: torch.device) -> np.ndarray:
    """Hitung probabilitas softmax untuk seluruh data pada loader.

    Args:
        model: Model dalam mode eval.
        loader: DataLoader yang menghasilkan tensor gambar saja.
        device: Device komputasi.

    Returns:
        Array (N, NUM_CLASSES) berisi probabilitas.
    """
    out = []
    for x in loader:
        out.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
    return np.concatenate(out)


def ensemble_probs(checkpoints: List[Path], cfg: Config, device: torch.device) -> np.ndarray:
    """Rata-ratakan probabilitas dari satu atau lebih checkpoint.

    Setiap checkpoint memakai ukuran cache miliknya sendiri (dari metadata).

    Args:
        checkpoints: Daftar path checkpoint.
        cfg: Objek Config.
        device: Device komputasi.

    Returns:
        Array (N, NUM_CLASSES) probabilitas rata-rata.
    """
    probs = []
    for path in checkpoints:
        model, meta = load_model_from_checkpoint(path, device)
        cfg_local = replace(cfg, image_size=meta["image_size"])
        if not cfg_local.cache_path("test").exists():
            raise FileNotFoundError(f"Cache test tidak ada: {cfg_local.cache_path('test')}. "
                                    f"Jalankan: python -m scripts.preprocess --size {meta['image_size']}")
        print(f"Prediksi dengan {path.name} (val acc {meta['val_acc']:.4f})")
        probs.append(predict_probs(model, make_test_loader(cfg_local), device))
    return np.mean(probs, axis=0)


def write_submission(ids: pd.Series, preds: np.ndarray, out_path: Path, sample_path: Path) -> pd.DataFrame:
    """Tulis submission.csv setelah memvalidasi formatnya terhadap sample_submission.

    Args:
        ids: Kolom id dari test.csv (urutan sama dengan cache test).
        preds: Prediksi label integer.
        out_path: Lokasi file keluaran.
        sample_path: Path sample_submission.csv.

    Returns:
        DataFrame submission.

    Raises:
        AssertionError: Jika format tidak sesuai sample_submission.
    """
    sub = pd.DataFrame({"id": ids.to_numpy(), "label": np.asarray(preds, dtype=int)})
    sample = pd.read_csv(sample_path)
    assert list(sub.columns) == list(sample.columns), "Nama kolom berbeda dari sample_submission"
    assert len(sub) == len(sample), f"Jumlah baris {len(sub)} != {len(sample)}"
    assert sub["id"].equals(sample["id"]), "Urutan atau isi id berbeda dari sample_submission"
    assert sub["label"].between(0, NUM_CLASSES - 1).all(), "Ada label di luar 0..15"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    return sub


def main() -> None:
    """Entry point CLI prediksi."""
    ap = argparse.ArgumentParser(description="Buat submission.csv dari satu atau lebih checkpoint.")
    ap.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = Config()
    out = args.out or cfg.submission_dir / "submission.csv"
    probs = ensemble_probs(args.checkpoints, cfg, get_device())
    test = pd.read_csv(cfg.test_csv)
    sub = write_submission(test["id"], probs.argmax(axis=1), out, cfg.sample_submission_csv)
    print(f"Submission disimpan: {out} ({len(sub)} baris)")
    print(sub["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
