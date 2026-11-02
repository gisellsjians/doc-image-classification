"""Training loop dua tahap untuk CPU: (1) latih head saja, (2) fine-tuning penuh.

Contoh:
    python -m scripts.train --benchmark                      # estimasi waktu per epoch
    python -m scripts.train --subset 2000 --epochs 2         # smoke test cepat
    python -m scripts.train --model resnet18 --epochs 8      # training penuh
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from scripts.config import Config
from scripts.dataset import make_loaders, make_splits
from scripts.model import (SUPPORTED_MODELS, build_model, count_parameters, save_checkpoint,
                           set_backbone_trainable, split_params)
from scripts.utils import get_device, set_seed


def _set_batchnorm_eval(model: nn.Module) -> None:
    """Kunci statistik BatchNorm (mode eval) agar backbone beku tidak bergeser.

    Args:
        model: Model yang sedang dalam mode train.
    """
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    backbone_frozen: bool = False, log_every: int = 50) -> Tuple[float, float]:
    """Jalankan satu epoch training.

    Args:
        model: Model yang dilatih.
        loader: DataLoader latih.
        criterion: Fungsi loss.
        optimizer: Optimizer.
        device: Device komputasi.
        backbone_frozen: Jika True, BatchNorm dikunci (tahap 1).
        log_every: Interval cetak progres (dalam batch).

    Returns:
        Tuple (rata-rata loss, akurasi) pada epoch ini.
    """
    model.train()
    if backbone_frozen:
        _set_batchnorm_eval(model)
    total_loss, correct, n = 0.0, 0, 0
    t0 = time.time()
    for step, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bs = y.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += bs
        if step % log_every == 0:
            print(f"  batch {step}/{len(loader)}  loss {total_loss / n:.4f}  "
                  f"acc {correct / n:.4f}  {time.time() - t0:.0f}s", flush=True)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluasi model tanpa gradien.

    Args:
        model: Model yang dievaluasi.
        loader: DataLoader yang menghasilkan (x, y).
        criterion: Fungsi loss.
        device: Device komputasi.

    Returns:
        Tuple (loss, akurasi, prediksi, target) dengan prediksi dan target berupa array numpy.
    """
    model.eval()
    total_loss, n = 0.0, 0
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        n += y.size(0)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(y.cpu().numpy())
    preds_arr, targets_arr = np.concatenate(preds), np.concatenate(targets)
    return total_loss / n, float((preds_arr == targets_arr).mean()), preds_arr, targets_arr


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, frozen: bool,
                    backbone_lr_mult: float = 0.1) -> torch.optim.Optimizer:
    """Bangun AdamW untuk tahap yang sedang berjalan.

    Tahap 1 (frozen): hanya head. Tahap 2: backbone memakai lr lebih kecil
    (discriminative learning rate) agar fitur pretrained tidak rusak.

    Args:
        model: Model.
        lr: Learning rate untuk head.
        weight_decay: Koefisien weight decay.
        frozen: True jika backbone sedang dibekukan.
        backbone_lr_mult: Pengali lr untuk backbone pada tahap 2.

    Returns:
        Optimizer AdamW.
    """
    backbone, head = split_params(model)
    if frozen:
        return AdamW(head, lr=lr, weight_decay=weight_decay)
    return AdamW([{"params": backbone, "lr": lr * backbone_lr_mult},
                  {"params": head, "lr": lr}], weight_decay=weight_decay)


def estimate_epoch_time(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                        device: torch.device, n_batches: int = 5) -> float:
    """Ukur kecepatan training pada beberapa batch lalu ekstrapolasi ke satu epoch.

    Args:
        model: Model (parameter akan berubah sedikit; gunakan model baru untuk training sungguhan).
        loader: DataLoader latih.
        criterion: Fungsi loss.
        device: Device komputasi.
        n_batches: Jumlah batch yang diukur (setelah 1 batch pemanasan).

    Returns:
        Estimasi detik per epoch.
    """
    set_backbone_trainable(model, True)
    optimizer = build_optimizer(model, 1e-4, 0.0, frozen=False)
    model.train()
    it = iter(loader)
    times = []
    for i in range(n_batches + 1):
        x, y = next(it)
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        criterion(model(x.to(device)), y.to(device)).backward()
        optimizer.step()
        if i > 0:  # batch pertama dibuang (pemanasan)
            times.append(time.time() - t0)
    return float(np.mean(times)) * len(loader)


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: Config,
        model_name: str, meta_base: Dict, epochs: int, freeze_epochs: int, lr: float,
        weight_decay: float, label_smoothing: float, patience: int,
        device: torch.device) -> Dict[str, List[float]]:
    """Latih model dua tahap dan simpan checkpoint terbaik (berdasarkan akurasi validasi).

    Args:
        model: Model hasil build_model.
        train_loader: DataLoader latih.
        val_loader: DataLoader validasi.
        cfg: Objek Config.
        model_name: Nama arsitektur.
        meta_base: Metadata yang disimpan bersama checkpoint.
        epochs: Total epoch.
        freeze_epochs: Jumlah epoch awal dengan backbone beku.
        lr: Learning rate head.
        weight_decay: Weight decay AdamW.
        label_smoothing: Label smoothing untuk cross-entropy.
        patience: Berhenti jika akurasi validasi tidak membaik selama sekian epoch.
        device: Device komputasi.

    Returns:
        Dict riwayat: train_loss, train_acc, val_loss, val_acc, epoch_seconds.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    history: Dict[str, List[float]] = {k: [] for k in
                                       ("train_loss", "train_acc", "val_loss", "val_acc", "epoch_seconds")}
    ckpt_path = cfg.checkpoint_dir / f"{model_name}_{cfg.image_size}_best.pt"
    hist_path = cfg.checkpoint_dir / f"{model_name}_{cfg.image_size}_history.json"
    best_acc, bad_epochs = -1.0, 0
    optimizer, scheduler = None, None

    for epoch in range(1, epochs + 1):
        frozen = epoch <= freeze_epochs
        if epoch == 1 or epoch == freeze_epochs + 1:  # awal tiap tahap: atur ulang optimizer
            set_backbone_trainable(model, not frozen)
            optimizer = build_optimizer(model, lr, weight_decay, frozen)
            scheduler = None if frozen else CosineAnnealingLR(optimizer, T_max=epochs - freeze_epochs)
            total, trainable = count_parameters(model)
            print(f"Tahap {'1 (head saja)' if frozen else '2 (fine-tuning penuh)'}: "
                  f"{trainable:,}/{total:,} parameter dilatih", flush=True)

        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, frozen)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        secs = time.time() - t0

        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc, secs)):
            history[k].append(float(v))
        print(f"Epoch {epoch}/{epochs}  train {tr_loss:.4f}/{tr_acc:.4f}  "
              f"val {va_loss:.4f}/{va_acc:.4f}  {secs:.0f}s", flush=True)

        if va_acc > best_acc:
            best_acc, bad_epochs = va_acc, 0
            meta = dict(meta_base, val_acc=float(va_acc), epoch=epoch)
            save_checkpoint(ckpt_path, model, model_name, meta)
            print(f"  checkpoint disimpan (val acc {va_acc:.4f})", flush=True)
        else:
            bad_epochs += 1
        hist_path.write_text(json.dumps(history))  # tulis tiap epoch agar aman jika proses terputus
        if bad_epochs >= patience:
            print(f"Early stopping: tidak ada perbaikan selama {patience} epoch.")
            break
    return history


def stratified_subset(idx: np.ndarray, labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Ambil subset stratified berukuran n dari indeks (untuk smoke test cepat).

    Args:
        idx: Indeks sumber.
        labels: Label seluruh data latih.
        n: Ukuran subset.
        seed: Random seed.

    Returns:
        Indeks terurut. Jika n >= len(idx), kembalikan idx apa adanya.
    """
    if n >= len(idx):
        return idx
    sub, _ = train_test_split(idx, train_size=n, stratify=labels[idx], random_state=seed)
    return np.sort(sub)


def main() -> None:
    """Entry point CLI training."""
    p = argparse.ArgumentParser(description="Training klasifikasi dokumen (PyTorch, CPU).")
    p.add_argument("--model", default="resnet18", choices=SUPPORTED_MODELS)
    p.add_argument("--size", type=int, default=None, help="Ukuran cache gambar (default Config.image_size)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--freeze-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--subset", type=int, default=None, help="Pakai subset train sebesar N (smoke test)")
    p.add_argument("--threads", type=int, default=None, help="Jumlah thread CPU (default semua core)")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--benchmark", action="store_true", help="Hanya estimasi waktu per epoch lalu keluar")
    args = p.parse_args()

    cfg = Config()
    if args.size:
        cfg.image_size = args.size
    if args.batch_size:
        cfg.batch_size = args.batch_size
    set_seed(cfg.seed)
    device = get_device()
    torch.set_num_threads(args.threads or os.cpu_count())
    print(f"Device: {device}, threads: {torch.get_num_threads()}, image_size: {cfg.image_size}")

    labels = pd.read_csv(cfg.train_csv)["label"].to_numpy()
    train_idx, val_idx = make_splits(labels, cfg.val_ratio, cfg.seed)
    if args.subset:
        train_idx = stratified_subset(train_idx, labels, args.subset, cfg.seed)
        val_idx = stratified_subset(val_idx, labels, max(args.subset // 4, 160), cfg.seed)
    print(f"Train {len(train_idx)}, val {len(val_idx)}")

    train_loader, val_loader = make_loaders(cfg, train_idx, val_idx, labels)
    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)

    if args.benchmark:
        secs = estimate_epoch_time(model, train_loader, nn.CrossEntropyLoss(), device)
        print(f"Estimasi fine-tuning penuh: {secs / 60:.1f} menit per epoch "
              f"(tahap head saja biasanya lebih cepat).")
        return

    meta_base = {"image_size": int(cfg.image_size), "seed": int(cfg.seed),
                 "val_ratio": float(cfg.val_ratio), "val_idx": [int(i) for i in val_idx]}
    history = fit(model, train_loader, val_loader, cfg, args.model, meta_base, args.epochs,
                  args.freeze_epochs, args.lr, args.weight_decay, args.label_smoothing,
                  args.patience, device)
    print(f"Selesai. Akurasi validasi terbaik: {max(history['val_acc']):.4f}")


if __name__ == "__main__":
    main()
