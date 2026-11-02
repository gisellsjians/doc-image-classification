"""Arsitektur model, pengaturan layer beku, dan simpan/muat checkpoint.

Pendekatan: transfer learning dari backbone CNN ringan yang dilatih di ImageNet,
dengan head klasifikasi baru untuk 16 kelas dokumen. Backbone ringan dipilih
agar training layak di CPU.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torchvision import models

from scripts.config import NUM_CLASSES

SUPPORTED_MODELS = ("resnet18", "mobilenet_v3_small", "mobilenet_v3_large")


def build_model(name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                dropout: float = 0.2) -> nn.Module:
    """Bangun model klasifikasi dengan head baru.

    Args:
        name: Salah satu dari SUPPORTED_MODELS.
        num_classes: Jumlah kelas keluaran.
        pretrained: Jika True, muat bobot ImageNet (perlu internet saat pertama kali).
        dropout: Probabilitas dropout sebelum layer klasifikasi akhir.

    Returns:
        Objek nn.Module dengan keluaran logit berukuran (batch, num_classes).

    Raises:
        ValueError: Jika name tidak didukung.
    """
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif name in ("mobilenet_v3_small", "mobilenet_v3_large"):
        if name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
        else:
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.mobilenet_v3_large(weights=weights)
        # classifier = [Linear, Hardswish, Dropout, Linear]; ganti dropout dan layer akhir
        model.classifier[2] = nn.Dropout(p=dropout)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Model '{name}' tidak didukung. Pilihan: {SUPPORTED_MODELS}")
    return model


def head_module(model: nn.Module) -> nn.Module:
    """Ambil modul head klasifikasi (fc pada ResNet, classifier pada MobileNet).

    Args:
        model: Model hasil build_model.

    Returns:
        Modul head.
    """
    return model.fc if hasattr(model, "fc") else model.classifier


def split_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Pisahkan parameter menjadi (backbone, head).

    Args:
        model: Model hasil build_model.

    Returns:
        Tuple (backbone_params, head_params).
    """
    head_ids = {id(p) for p in head_module(model).parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    head = [p for p in model.parameters() if id(p) in head_ids]
    return backbone, head


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Bekukan atau buka backbone. Head selalu dapat dilatih.

    Args:
        model: Model hasil build_model.
        trainable: True untuk fine-tuning penuh, False untuk melatih head saja.
    """
    backbone, head = split_params(model)
    for p in backbone:
        p.requires_grad = trainable
    for p in head:
        p.requires_grad = True


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Hitung parameter total dan yang dapat dilatih.

    Args:
        model: Model apa pun.

    Returns:
        Tuple (total, trainable).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(path: Path, model: nn.Module, model_name: str, meta: Dict[str, Any]) -> None:
    """Simpan bobot dan metadata (ukuran gambar, split validasi, skor) ke satu file.

    Args:
        path: Lokasi file .pt.
        model: Model yang disimpan.
        model_name: Nama arsitektur (kunci untuk membangun ulang model).
        meta: Metadata berisi tipe python biasa (int, float, str, list).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": model_name, "state_dict": model.state_dict(), "meta": meta}, path)


def load_model_from_checkpoint(path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Muat model dari checkpoint dalam mode eval.

    Args:
        path: Lokasi file .pt hasil save_checkpoint.
        device: Device tujuan.

    Returns:
        Tuple (model, meta).
    """
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["model_name"], pretrained=False)  # bobot datang dari state_dict
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["meta"]
