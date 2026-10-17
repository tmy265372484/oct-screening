"""Continue the saved PyTorch model; keep the original script and weights intact."""

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import random
import sys

from config import BASE_DIR, BATCH_SIZE, DATA_DIR, EPOCHS, IMG_SIZE, MODEL_PATH

RUN_DIR = BASE_DIR / "models" / "torch_resume"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
AUGMENTATION_PROFILE = "mild_oct_v1"
AUGMENTATION_CONFIG = dict(horizontal_flip_probability=0.5, rotation_degrees=10,
                           translate_fraction=[0.03, 0.03], scale_range=[0.97, 1.03],
                           brightness=0.10, contrast=0.10)


def build_transforms(image_size):
    from torchvision import transforms

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Grayscale(3), transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=AUGMENTATION_CONFIG["horizontal_flip_probability"]),
        transforms.RandomAffine(
            degrees=AUGMENTATION_CONFIG["rotation_degrees"],
            translate=AUGMENTATION_CONFIG["translate_fraction"],
            scale=AUGMENTATION_CONFIG["scale_range"],
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.ColorJitter(brightness=AUGMENTATION_CONFIG["brightness"],
                               contrast=AUGMENTATION_CONFIG["contrast"]),
        transforms.ToTensor(), normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Grayscale(3), transforms.Resize(image_size), transforms.ToTensor(), normalize,
    ])
    return train_tf, val_tf


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_save(torch, path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def infer_saved_epoch(state, batches_per_epoch):
    counts = {int(value.item()) for key, value in state.items() if key.endswith("num_batches_tracked")}
    if len(counts) != 1 or not batches_per_epoch:
        raise ValueError("无法从旧模型可靠推算轮数，未启动训练。")
    epoch, remainder = divmod(counts.pop(), batches_per_epoch)
    if remainder:
        raise ValueError("旧模型批次计数与当前数据集不匹配，未启动训练。")
    return epoch


def training_image_count():
    root = Path(DATA_DIR) / "train"
    return sum(
        Path(name).suffix.lower() in IMAGE_EXTENSIONS
        for category in root.iterdir() if category.is_dir()
        for _, _, names in os.walk(category, followlinks=True)
        for name in names
    )


def show_status(run_dir):
    import torch

    latest = run_dir / "latest.pt"
    progress_path = run_dir / "progress.json"
    if latest.exists():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
        best_acc, best_epoch = checkpoint['best_acc'], checkpoint['best_epoch']
        if (run_dir / "best.pt").exists():
            best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
            if best['val_acc'] > best_acc:
                best_acc, best_epoch = best['val_acc'], best['epoch']
        print(f"可恢复的完整检查点：第 {checkpoint['completed_epoch']} 轮")
        print(f"检查点中的目标轮数：{checkpoint['target_epochs']}")
        print(f"最佳验证准确率：{best_acc:.4f}（第 {best_epoch} 轮）")
        print(f"检查点：{latest}")
        print(f"续训增强方案：{checkpoint.get('augmentation_profile', '旧版：水平翻转及旋转')}")
        if checkpoint.get("augmentation_start_epoch") is not None:
            print(f"此方案从第 {checkpoint['augmentation_start_epoch']} 轮开始生效。")
    elif Path(MODEL_PATH).exists():
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        batches = math.ceil(training_image_count() / BATCH_SIZE)
        epoch = infer_saved_epoch(state, batches)
        print(f"旧模型可恢复到第 {epoch} 轮；下一轮编号为 {epoch + 1}。")
        print("该轮数按当前数据集、批大小及 BatchNorm 批次计数推算，要求与原训练一致。")
        print("旧训练最终停止轮数未知；这里显示的是已保存模型的轮数。")
        print("首次续训会重建优化器和学习率调度器，保留已有模型权重。")
        print(f"旧模型：{MODEL_PATH}")
    else:
        print("未找到已保存模型或续训检查点。")
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        print(f"最近进度记录：{progress['updated_at']}")
        print(f"记录状态：{progress['phase']}；进程 PID：{progress['pid']}")
        print(f"记录位置：第 {progress['current_epoch']} 轮，批次 {progress['batch']}/{progress['batches']}")
        if progress.get("error"):
            print(f"记录的错误：{progress['error']}")
        print("进度文件是上次写入的记录，不代表进程仍在运行；恢复以完整检查点为准。")
    else:
        print("新续训脚本尚无进度记录。")


@contextmanager
def training_lock(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "training.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("已有续训进程使用此目录，请勿重复启动。") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def train(run_dir, target_epochs):
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, models

    logger = logging.getLogger("oct_resume")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "training.log", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)

    progress = dict(pid=os.getpid(), phase="starting", current_epoch=0, completed_epoch=0,
                    target_epochs=target_epochs, batch=0, batches=0,
                    augmentation_profile=AUGMENTATION_PROFILE, augmentation_config=AUGMENTATION_CONFIG)

    def record(**updates):
        progress.update(updates, updated_at=datetime.now().astimezone().isoformat(timespec="seconds"))
        atomic_json(run_dir / "progress.json", progress)

    try:
        record()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("设备：%s；目标：%s 轮", device, target_epochs)
        train_tf, val_tf = build_transforms(IMG_SIZE)
        train_ds = datasets.ImageFolder(Path(DATA_DIR) / "train", transform=train_tf)
        val_ds = datasets.ImageFolder(Path(DATA_DIR) / "val", transform=val_tf)
        if train_ds.class_to_idx != val_ds.class_to_idx:
            raise ValueError("训练集和验证集的类别映射不一致。")
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                  pin_memory=device.type == "cuda")
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                pin_memory=device.type == "cuda")
        metadata = dict(train_samples=len(train_ds), batch_size=BATCH_SIZE,
                        batches_per_epoch=len(train_loader), class_to_idx=train_ds.class_to_idx,
                        img_size=list(IMG_SIZE))
        latest = run_dir / "latest.pt"
        checkpoint = torch.load(latest, map_location="cpu", weights_only=True) if latest.exists() else None
        if checkpoint:
            if checkpoint["metadata"] != metadata:
                raise ValueError("当前数据数量、批大小、图像尺寸或类别映射与检查点不一致，已停止。")
            state = checkpoint["model_state_dict"]
            completed_epoch = checkpoint["completed_epoch"]
        else:
            if not Path(MODEL_PATH).exists():
                raise FileNotFoundError("未找到原模型，已停止；此脚本不会自动从零训练。")
            state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            completed_epoch = infer_saved_epoch(state, len(train_loader))

        previous_profile = checkpoint.get("augmentation_profile") if checkpoint else None
        if previous_profile == AUGMENTATION_PROFILE:
            augmentation_start_epoch = checkpoint.get("augmentation_start_epoch", completed_epoch + 1)
        else:
            augmentation_start_epoch = completed_epoch + 1
            logger.info("切换训练增强：%s -> %s；从第 %s 轮生效，验证预处理不变。",
                        previous_profile or "旧版：水平翻转及旋转", AUGMENTATION_PROFILE, augmentation_start_epoch)
        logger.info("增强参数：%s", json.dumps(AUGMENTATION_CONFIG, ensure_ascii=False))
        record(augmentation_start_epoch=augmentation_start_epoch)

        model = models.resnet50(weights=None)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
        model.load_state_dict(state)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

        def validate():
            model.eval()
            loss_sum, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    output = model(images)
                    loss_sum += criterion(output, labels).item() * labels.size(0)
                    correct += (output.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            return loss_sum / total, correct / total

        if checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_acc, best_epoch = checkpoint["best_acc"], checkpoint["best_epoch"]
            random.setstate(checkpoint["python_rng_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if device.type == "cuda" and checkpoint["cuda_rng_state"]:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
            logger.info("恢复完整检查点：已完成 %s 轮，优化器和调度器已恢复。", completed_epoch)
        else:
            record(phase="validating_legacy", completed_epoch=completed_epoch, current_epoch=completed_epoch,
                   batches=len(train_loader))
            _, best_acc = validate()
            best_epoch = completed_epoch
            logger.info("恢复旧模型：已保存第 %s 轮；首次续训重建优化器和调度器。", completed_epoch)
            logger.info("旧轮数由 BatchNorm 计数推算，要求数据集、批大小与原训练一致。")

        def save_best():
            atomic_save(torch, run_dir / "best.pt", dict(
                epoch=best_epoch, val_acc=best_acc, metadata=metadata, model_state_dict=model.state_dict(),
                augmentation_profile=AUGMENTATION_PROFILE, augmentation_config=AUGMENTATION_CONFIG,
                augmentation_start_epoch=augmentation_start_epoch,
            ))

        best_path = run_dir / "best.pt"
        if best_path.exists():
            persisted_best = torch.load(best_path, map_location="cpu", weights_only=True)
            if persisted_best["metadata"] != metadata:
                raise ValueError("最佳模型的训练配置与当前数据不一致，已停止。")
            if persisted_best["val_acc"] < best_acc:
                raise ValueError("最佳模型与完整检查点不一致，已停止以保护已有权重。")
            if persisted_best["val_acc"] > best_acc:
                best_acc, best_epoch = persisted_best["val_acc"], persisted_best["epoch"]
                logger.info("保留独立保存的更佳模型：第 %s 轮，准确率 %.4f。", best_epoch, best_acc)
        elif checkpoint:
            raise FileNotFoundError("best.pt 缺失，已停止以免丢失检查点记录的最佳模型。")
        else:
            save_best()

        def save_checkpoint():
            atomic_save(torch, latest, dict(
                format_version=1, completed_epoch=completed_epoch, target_epochs=target_epochs,
                best_acc=best_acc, best_epoch=best_epoch, metadata=metadata,
                augmentation_profile=AUGMENTATION_PROFILE, augmentation_config=AUGMENTATION_CONFIG,
                augmentation_start_epoch=augmentation_start_epoch,
                model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(), python_rng_state=random.getstate(),
                torch_rng_state=torch.get_rng_state(),
                cuda_rng_state=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            ))

        save_checkpoint()
        record(phase="ready", completed_epoch=completed_epoch, current_epoch=completed_epoch, batch=0,
               batches=len(train_loader), best_acc=best_acc, best_epoch=best_epoch)
        for epoch in range(completed_epoch + 1, target_epochs + 1):
            record(phase="training", current_epoch=epoch, batch=0)
            logger.info("Epoch %s/%s 开始", epoch, target_epochs)
            model.train()
            correct, total, loss_sum = 0, 0, 0.0
            for batch, (images, labels) in enumerate(train_loader, 1):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * labels.size(0)
                correct += (output.argmax(1) == labels).sum().item()
                total += labels.size(0)
                if batch % 100 == 0 or batch == len(train_loader):
                    record(batch=batch)
                    logger.info("Epoch %s/%s 批次 %s/%s", epoch, target_epochs, batch, len(train_loader))
            record(phase="validating")
            val_loss, val_acc = validate()
            scheduler.step(val_loss)
            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, epoch
                save_best()
            completed_epoch = epoch
            save_checkpoint()
            record(phase="epoch_complete", completed_epoch=epoch, train_loss=loss_sum / total,
                   train_acc=correct / total, val_loss=val_loss, val_acc=val_acc,
                   best_acc=best_acc, best_epoch=best_epoch)
            logger.info("Epoch %s/%s 完成 | 训练准确率 %.4f | 验证准确率 %.4f | 完整检查点已保存",
                        epoch, target_epochs, correct / total, val_acc)
        record(phase="completed")
        logger.info("完成：已训练 %s 轮；最佳验证准确率 %.4f（第 %s 轮）", completed_epoch, best_acc, best_epoch)
    except KeyboardInterrupt:
        record(phase="interrupted")
        logger.warning("训练已中断；下次从最新完整检查点恢复，中断的这一轮会重跑。")
        raise
    except Exception as error:
        record(phase="failed", error=str(error))
        logger.exception("训练失败；原模型保留，下次从最新完整检查点恢复。")
        raise


def main():
    parser = argparse.ArgumentParser(description="查看训练进度，或从已有权重继续训练。")
    parser.add_argument("--status", action="store_true", help="只查看进度，不启动训练")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="目标总轮数，默认 50")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR, help="续训检查点和日志目录")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs 必须大于 0")
    if args.status:
        show_status(args.run_dir)
    else:
        with training_lock(args.run_dir):
            train(args.run_dir, args.epochs)


if __name__ == "__main__":
    main()
