"""Konfigurasi terpusat proyek: path, konstanta kelas, dan hyperparameter default."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Pemetaan label -> nama kelas, sesuai deskripsi kompetisi.
CLASS_NAMES: Dict[int, str] = {
    0: "advertisement", 1: "budget", 2: "email", 3: "file_folder",
    4: "form", 5: "handwritten", 6: "invoice", 7: "letter",
    8: "memo", 9: "news_article", 10: "presentation", 11: "questionnaire",
    12: "resume", 13: "scientific_publication", 14: "scientific_report",
    15: "specification",
}
NUM_CLASSES: int = len(CLASS_NAMES)


def _default_data_dir() -> Path:
    """Ambil lokasi data dari env var DOC_DATA_DIR, fallback ke <repo>/data.

    Returns:
        Path folder data.
    """
    return Path(os.environ.get("DOC_DATA_DIR", PROJECT_ROOT / "data"))


@dataclass
class Config:
    """Container hyperparameter dan path. Ubah di satu tempat, berlaku di semua script.

    Attributes:
        data_dir: Folder berisi train/, test/, dan file CSV (override: env DOC_DATA_DIR).
        checkpoint_dir: Folder penyimpanan bobot model (di-ignore oleh git).
        submission_dir: Folder keluaran file submission.
        figures_dir: Folder gambar untuk laporan.
        image_size: Sisi gambar (piksel) pada cache dan input model. Dipilih kecil
            agar training di CPU layak.
        batch_size: Jumlah sampel per batch.
        num_workers: Jumlah worker DataLoader (0 aman untuk Windows; data sudah di-cache).
        seed: Random seed untuk reprodusibilitas.
        val_ratio: Proporsi data latih yang dijadikan validasi.
        pixel_mean: Mean piksel (grayscale) untuk normalisasi, rata-rata mean ImageNet.
        pixel_std: Std piksel (grayscale) untuk normalisasi, rata-rata std ImageNet.
    """

    data_dir: Path = field(default_factory=_default_data_dir)
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoint"
    submission_dir: Path = PROJECT_ROOT / "submissions"
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures"
    image_size: int = 160
    batch_size: int = 64
    num_workers: int = 0
    seed: int = 42
    val_ratio: float = 0.15
    pixel_mean: float = 0.449
    pixel_std: float = 0.226

    @property
    def train_csv(self) -> Path:
        """Path ke train.csv."""
        return self.data_dir / "train.csv"

    @property
    def test_csv(self) -> Path:
        """Path ke test.csv."""
        return self.data_dir / "test.csv"

    @property
    def sample_submission_csv(self) -> Path:
        """Path ke sample_submission.csv."""
        return self.data_dir / "sample_submission.csv"

    @property
    def cache_dir(self) -> Path:
        """Folder cache gambar hasil resize (di dalam data/, ikut di-ignore git)."""
        return self.data_dir / "cache"

    def cache_path(self, split: str, size: int = None) -> Path:
        """Path file cache .npy untuk satu split.

        Args:
            split: "train" atau "test".
            size: Ukuran sisi gambar. Default memakai self.image_size.

        Returns:
            Path ke file, contoh: data/cache/train_160.npy.
        """
        size = size or self.image_size
        return self.cache_dir / f"{split}_{size}.npy"
