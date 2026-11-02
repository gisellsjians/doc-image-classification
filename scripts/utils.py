"""Fungsi utilitas umum: seed, device, dan validasi struktur data."""

import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from scripts.config import CLASS_NAMES, Config


def set_seed(seed: int) -> None:
    """Set seed untuk random, numpy, dan torch agar eksperimen reprodusibel.

    Args:
        seed: Nilai seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Pilih device terbaik yang tersedia (CUDA, MPS, lalu CPU).

    Returns:
        Objek torch.device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def verify_data_layout(cfg: Config, sample_n: int = 200) -> Dict[str, object]:
    """Validasi bahwa data sudah terunduh dan konsisten dengan CSV.

    Pemeriksaan: ketiga CSV ada, jumlah baris sesuai spesifikasi, label cocok
    dengan CLASS_NAMES, id test identik dengan sample_submission, dan sampel
    file gambar benar-benar ada di disk.

    Args:
        cfg: Objek Config.
        sample_n: Jumlah file acak per split yang dicek keberadaannya.

    Returns:
        Dict ringkasan hasil validasi. Melempar AssertionError jika ada yang gagal.
    """
    train = pd.read_csv(cfg.train_csv)
    test = pd.read_csv(cfg.test_csv)
    sub = pd.read_csv(cfg.sample_submission_csv)

    assert len(train) == 32003, f"Jumlah train tidak sesuai: {len(train)}"
    assert len(test) == 7994, f"Jumlah test tidak sesuai: {len(test)}"
    assert test["id"].equals(sub["id"]), "Urutan id test != sample_submission"

    label_to_name = train.drop_duplicates("label").set_index("label")["class_name"].to_dict()
    assert label_to_name == CLASS_NAMES, "Mapping label-kelas berbeda dari CLASS_NAMES"

    missing: List[str] = []
    for df in (train, test):
        for rel in df["filename"].sample(sample_n, random_state=cfg.seed):
            if not (cfg.data_dir / rel).exists():
                missing.append(rel)
    assert not missing, f"{len(missing)} file gambar tidak ditemukan, contoh: {missing[:3]}"

    return {"n_train": len(train), "n_test": len(test), "n_classes": len(label_to_name),
            "images_checked": 2 * sample_n}
