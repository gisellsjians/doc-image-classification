"""Dataset PyTorch berbasis cache .npy, pembagian train/val, dan DataLoader."""

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from scripts.config import Config
from scripts.preprocess import get_eval_transform, get_train_transform


class CachedDocDataset(Dataset):
    """Dataset yang membaca gambar dari cache .npy (memory-mapped).

    Args:
        cache_path: Path ke file .npy berbentuk (N, H, W) uint8.
        indices: Indeks baris (terhadap cache dan CSV) yang dipakai subset ini.
        labels: Label seluruh baris CSV (panjang N), atau None untuk data test.
        transform: Transform pada tensor 1xHxW float [0, 1].
    """

    def __init__(self, cache_path: Path, indices: Sequence[int],
                 labels: Optional[Sequence[int]] = None,
                 transform: Optional[Callable] = None) -> None:
        """Buka cache sebagai memmap dan simpan indeks, label, dan transform."""
        self.images = np.load(cache_path, mmap_mode="r")  # tidak dimuat penuh ke RAM
        self.indices = np.asarray(indices)
        self.labels = None if labels is None else np.asarray(labels)
        self.transform = transform

    def __len__(self) -> int:
        """Jumlah sampel pada subset."""
        return len(self.indices)

    def __getitem__(self, i: int):
        """Ambil satu sampel.

        Args:
            i: Posisi pada subset.

        Returns:
            (x, y) dengan x tensor 3xHxW (grayscale diulang 3 channel agar cocok
            dengan model pretrained) dan y label int; untuk test hanya x.
        """
        row = self.indices[i]
        arr = np.array(self.images[row])                      # salin dari memmap
        x = torch.from_numpy(arr).float().div_(255.0).unsqueeze(0)  # 1xHxW di [0,1]
        if self.transform is not None:
            x = self.transform(x)
        x = x.repeat(3, 1, 1)                                 # 1 -> 3 channel
        if self.labels is None:
            return x
        return x, int(self.labels[row])


def make_splits(labels: Sequence[int], val_ratio: float,
                seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bagi indeks menjadi train dan validasi secara stratified.

    Stratifikasi menjaga proporsi kelas sama di kedua subset, sehingga
    akurasi validasi tidak bias oleh komposisi kelas.

    Args:
        labels: Label seluruh data latih.
        val_ratio: Proporsi validasi, misalnya 0.15.
        seed: Random seed.

    Returns:
        Tuple (train_idx, val_idx).
    """
    idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_ratio, stratify=np.asarray(labels), random_state=seed)
    return np.sort(train_idx), np.sort(val_idx)


def make_loaders(cfg: Config, train_idx: Sequence[int], val_idx: Sequence[int],
                 labels: Sequence[int]) -> Tuple[DataLoader, DataLoader]:
    """Buat DataLoader train (dengan augmentasi) dan validasi.

    Args:
        cfg: Objek Config.
        train_idx: Indeks data latih.
        val_idx: Indeks data validasi.
        labels: Label seluruh data latih.

    Returns:
        Tuple (train_loader, val_loader).
    """
    path = cfg.cache_path("train")
    train_ds = CachedDocDataset(path, train_idx, labels, get_train_transform(cfg))
    val_ds = CachedDocDataset(path, val_idx, labels, get_eval_transform(cfg))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)
    return train_loader, val_loader


def make_test_loader(cfg: Config) -> DataLoader:
    """Buat DataLoader untuk test (urutan sama dengan test.csv, tanpa shuffle).

    Args:
        cfg: Objek Config.

    Returns:
        DataLoader yang menghasilkan tensor gambar saja.
    """
    n = len(pd.read_csv(cfg.test_csv))
    ds = CachedDocDataset(cfg.cache_path("test"), np.arange(n), None, get_eval_transform(cfg))
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)


def make_eval_loader(cfg: Config, indices: Sequence[int], labels: Sequence[int]) -> DataLoader:
    """Buat DataLoader evaluasi (tanpa augmentasi, tanpa shuffle) untuk subset data latih.

    Args:
        cfg: Objek Config (image_size menentukan file cache yang dibaca).
        indices: Indeks baris data latih yang dievaluasi, misalnya indeks validasi.
        labels: Label seluruh data latih.

    Returns:
        DataLoader yang menghasilkan pasangan (x, y).
    """
    ds = CachedDocDataset(cfg.cache_path("train"), indices, labels, get_eval_transform(cfg))
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
