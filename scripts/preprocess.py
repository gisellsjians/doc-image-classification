"""Preprocessing gambar: resize ke grayscale uint8 dan simpan sebagai cache .npy.

Alasan cache: decoding ribuan file TIF besar di CPU adalah bottleneck utama.
Kita decode dan resize SEKALI, lalu training membaca array kecil dari disk.
Modul ini sengaja tidak mengimpor torch di level atas agar ringan dipakai
untuk pembuatan cache; transform torchvision diimpor secara lazy.
"""

import os

# Batasi thread BLAS SEBELUM numpy diimpor. Di Windows setiap proses worker
# mengimpor ulang modul ini; tanpa batas ini tiap worker memesan buffer
# OpenBLAS sebesar jumlah core dan memori cepat habis.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from scripts.config import Config

# pandas sengaja TIDAK diimpor di level atas: worker tidak membutuhkannya dan
# impor pandas di setiap proses anak adalah penyebab utama MemoryError.


def load_image(path: Path, size: int) -> np.ndarray:
    """Baca satu gambar, ubah ke grayscale, resize ke (size, size).

    Resize PIL dengan BILINEAR sudah anti-aliasing saat downscale, sehingga
    garis teks tipis tidak menjadi noise aliasing.

    Args:
        path: Path file gambar.
        size: Sisi gambar target (piksel).

    Returns:
        Array uint8 berukuran (size, size).
    """
    with Image.open(path) as im:
        im = im.convert("L")  # RVL-CDIP pada dasarnya grayscale; paksa 1 channel
        im = im.resize((size, size), resample=Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def _load_one(args: Tuple[Path, int]) -> Optional[np.ndarray]:
    """Wrapper aman untuk Pool: kembalikan None jika file gagal dibaca.

    Args:
        args: Tuple (path, size).

    Returns:
        Array hasil load_image, atau None jika terjadi error.
    """
    path, size = args
    try:
        return load_image(path, size)
    except Exception:
        return None


def build_cache(rel_paths: Sequence[str], root: Path, out_path: Path, size: int,
                n_workers: Optional[int] = None) -> List[str]:
    """Bangun cache .npy berurutan sama dengan rel_paths (urutan = urutan baris CSV).

    Urutan sangat penting: indeks i pada cache harus sesuai baris i pada CSV.
    Pool.imap menjaga urutan hasil.

    Args:
        rel_paths: Daftar path relatif gambar (kolom filename pada CSV).
        root: Folder dasar (data_dir) tempat path relatif dirujuk.
        out_path: Lokasi file .npy keluaran.
        size: Sisi gambar target.
        n_workers: Jumlah proses paralel. Default: min(4, core - 1), dibatasi agar RAM aman.

    Returns:
        Daftar path relatif yang gagal dibaca (diisi halaman putih di cache).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_workers = n_workers or max(1, min(4, (os.cpu_count() or 2) - 1))
    n = len(rel_paths)

    # open_memmap menulis langsung ke disk, jadi RAM tidak membengkak.
    cache = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8,
                                      shape=(n, size, size))
    tasks = [(root / p, size) for p in rel_paths]
    failed: List[str] = []

    with mp.get_context("spawn").Pool(n_workers) as pool:  # spawn: perilaku sama di semua OS
        results = pool.imap(_load_one, tasks, chunksize=32)
        for i, arr in enumerate(tqdm(results, total=n, desc=out_path.name)):
            if arr is None:
                failed.append(rel_paths[i])
                cache[i] = 255  # halaman putih sebagai pengganti
            else:
                cache[i] = arr
    cache.flush()
    return failed


def get_train_transform(cfg: Config):
    """Augmentasi ringan untuk data latih (input: tensor 1xHxW dalam [0, 1]).

    Tanpa horizontal flip karena dokumen bukan objek simetris (teks akan
    terbalik dan tidak pernah muncul di data uji). Hanya geser, rotasi, dan
    skala kecil untuk meniru variasi hasil scan.

    Args:
        cfg: Objek Config.

    Returns:
        Objek transform torchvision.
    """
    from torchvision import transforms as T
    return T.Compose([
        T.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.97, 1.03), fill=1.0),
        T.Normalize(mean=[cfg.pixel_mean], std=[cfg.pixel_std]),
    ])


def get_eval_transform(cfg: Config):
    """Transform untuk validasi dan test: hanya normalisasi.

    Args:
        cfg: Objek Config.

    Returns:
        Objek transform torchvision.
    """
    from torchvision import transforms as T
    return T.Compose([T.Normalize(mean=[cfg.pixel_mean], std=[cfg.pixel_std])])


def main() -> None:
    """CLI: python -m scripts.preprocess --size 160 --splits train test."""
    parser = argparse.ArgumentParser(description="Bangun cache gambar ter-resize.")
    parser.add_argument("--size", type=int, default=None, help="Sisi gambar (default: Config.image_size)")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--workers", type=int, default=None, help="Jumlah proses (default maks 4; kurangi jika RAM terbatas)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import pandas as pd  # hanya proses utama yang membutuhkan pandas

    cfg = Config()
    size = args.size or cfg.image_size
    for split in args.splits:
        out_path = cfg.cache_path(split, size)
        if out_path.exists() and not args.overwrite:
            print(f"Lewati {out_path.name} (sudah ada, gunakan --overwrite untuk membangun ulang)")
            continue
        df = pd.read_csv(cfg.train_csv if split == "train" else cfg.test_csv)
        failed = build_cache(df["filename"].tolist(), cfg.data_dir, out_path, size, args.workers)
        print(f"{split}: selesai, {len(failed)} file gagal dibaca. {failed[:5]}")


if __name__ == "__main__":
    main()