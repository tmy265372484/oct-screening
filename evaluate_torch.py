"""Evaluate a fixed OCT checkpoint on the local test split without training.

This is a retrospective local dataset evaluation. Patient independence has not
been verified; its scores must not be described as clinical validation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import BASE_DIR, TEST_DIR
from predict_torch import CLASS_NAMES, DEFAULT_CHECKPOINT, OCTPredictor


INDEPENDENCE_NOTE = "尚未核验患者级独立性，也未完成训练/验证/测试图像去重审计。"
SCOPE_NOTE = "本地数据集离线评估，不是前瞻性临床验证，不用于证明临床诊断性能。"


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classification_metrics(matrix: list[list[int]]) -> dict:
    """Rows are true labels, columns predictions; undefined ratios are zero."""
    if len(matrix) != len(CLASS_NAMES) or any(len(row) != len(CLASS_NAMES) for row in matrix):
        raise ValueError("混淆矩阵必须是与四类别顺序一致的 4×4 矩阵。")
    total = sum(sum(row) for row in matrix)
    if not total:
        raise ValueError("测试集为空，无法计算指标。")
    per_class = {}
    for index, category in enumerate(CLASS_NAMES):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[category] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": support, "predicted_count": predicted_count,
        }
    return {
        "accuracy": sum(matrix[i][i] for i in range(len(CLASS_NAMES))) / total,
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(CLASS_NAMES),
        "per_class": per_class,
        "confusion_matrix_actual_rows_predicted_columns": matrix,
        "class_order": list(CLASS_NAMES),
        "zero_division_policy": "precision/recall/F1 无定义时记为 0",
        "test_images": total,
    }


def save_confusion_matrix(matrix: list[list[int]], output_dir: Path) -> None:
    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual / predicted", *CLASS_NAMES])
        for category, row in zip(CLASS_NAMES, matrix):
            writer.writerow([category, *row])

    # Pillow keeps this export usable without another plotting dependency.
    def font(size: int):
        for candidate in ("DejaVuSans.ttf", "arial.ttf"):
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
        return ImageFont.load_default()

    canvas = Image.new("RGB", (860, 820), "#f5f8fc")
    draw = ImageDraw.Draw(canvas)
    draw.text((52, 30), "OCT local test confusion matrix", font=font(29), fill="#152a3c")
    draw.text((52, 77), "Rows = actual | Columns = predicted | Patient independence unverified",
              font=font(16), fill="#506070")
    origin_x, origin_y, cell = 170, 180, 145
    peak = max(max(row) for row in matrix) or 1
    for index, category in enumerate(CLASS_NAMES):
        draw.text((origin_x + index * cell + cell // 2, origin_y - 33), category,
                  font=font(19), fill="#233d50", anchor="mm")
        draw.text((origin_x - 15, origin_y + index * cell + cell // 2), category,
                  font=font(19), fill="#233d50", anchor="rm")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            strength = value / peak
            color = tuple(round(light + strength * (dark - light))
                          for light, dark in zip((233, 243, 248), (18, 100, 145)))
            x, y = origin_x + col_index * cell, origin_y + row_index * cell
            draw.rectangle((x, y, x + cell - 3, y + cell - 3), fill=color)
            draw.text((x + cell // 2, y + cell // 2), str(value), font=font(31),
                      fill="white" if strength > 0.55 else "#173348", anchor="mm")
    draw.text((52, 786), "Research dataset evaluation; not clinical validation.",
              font=font(17), fill="#506070")
    canvas.save(output_dir / "confusion_matrix.png")


def evaluate(data_dir: Path, checkpoint: Path, device: str, output_dir: Path, batch_size: int = 16) -> dict:
    data_dir, checkpoint = data_dir.expanduser().resolve(), checkpoint.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"未找到测试目录：{data_dir}；请用 --data-dir 指向包含四个类别子目录的 test 目录。")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"未找到模型检查点：{checkpoint}；请用 --checkpoint 指定已保存的 best.pt。")
    if batch_size < 1:
        raise ValueError("batch_size 必须大于 0。")
    missing = [category for category in CLASS_NAMES if not (data_dir / category).is_dir()]
    if missing:
        raise ValueError(f"测试目录缺少类别子目录：{', '.join(missing)}。")

    started = time.perf_counter()
    before_hash = checkpoint_sha256(checkpoint)
    predictor = OCTPredictor(checkpoint, device)
    # ImageFolder converts to RGB, followed by exactly the predictor's existing
    # grayscale/resize/tensor/normalize validation transform. No augmentation.
    dataset = ImageFolder(data_dir, transform=predictor.transform)
    expected_mapping = {category: index for index, category in enumerate(CLASS_NAMES)}
    if dataset.class_to_idx != expected_mapping:
        raise ValueError("测试集类别必须恰好为 CNV、DME、DRUSEN、NORMAL。")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    matrix = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64)
    seen = 0
    print(f"固定检查点 {checkpoint.name}，轮次 {predictor.checkpoint_info['epoch']}；"
          f"测试图像 {len(dataset)} 张；设备 {predictor.device}。", flush=True)
    with predictor.lock, torch.inference_mode():
        predictor.model.eval()
        for batch, actual in loader:
            logits = predictor.model(batch.to(predictor.device))
            if not torch.isfinite(logits).all():
                raise RuntimeError("模型产生非有限数值，评估已停止。")
            predicted = logits.argmax(dim=1).cpu()
            combined = actual * len(CLASS_NAMES) + predicted
            matrix += torch.bincount(combined, minlength=len(CLASS_NAMES) ** 2).reshape_as(matrix)
            seen += actual.numel()
            if seen % 160 == 0 or seen == len(dataset):
                print(f"已评估 {seen}/{len(dataset)} 张，耗时 {time.perf_counter() - started:.1f} 秒。", flush=True)
    if checkpoint_sha256(checkpoint) != before_hash:
        raise RuntimeError("检查点文件在评估期间发生变化，结果未保存。")

    report = classification_metrics(matrix.tolist())
    report.update({
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluation_type": "fixed_checkpoint_local_test_split",
        "checkpoint": {
            key: predictor.checkpoint_info[key]
            for key in ("filename", "format", "epoch", "best_epoch", "val_acc", "best_val_acc")
        },
        "checkpoint_sha256": before_hash,
        "checkpoint_file_unchanged": True,
        "source": {
            "dataset": "Retinal OCT Images (Kermany2018), locally prepared copy",
            "split": "test",
            "directory": "data/OCT/test" if data_dir == TEST_DIR.resolve() else "CLI-provided local test directory",
            "preparation": "本项目本地 test 目录；压缩包原始来源和官方划分未在本次评估中重新核验。",
            "class_counts": {category: report["per_class"][category]["support"] for category in CLASS_NAMES},
            "patient_independence_verified": False,
            "patient_independence_note": INDEPENDENCE_NOTE,
            "clinical_validation": False,
        },
        "preprocessing": {
            "loader": "RGB (ImageFolder)", "grayscale_channels": 3,
            "resize_hw": list(predictor.image_size), "random_augmentation": False,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        },
        "device": str(predictor.device), "batch_size": batch_size,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "scope_note": SCOPE_NOTE,
        "selection_note": "评估前固定指定检查点；本次未基于测试指标调整参数或重新选模型。默认 best.pt 由训练阶段验证集选出。",
        "shared_summary_note": "仅包含汇总统计，不包含原图文件名、患者编号或本机绝对路径。",
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    (output_dir / "evaluation.json").write_text(encoded, encoding="utf-8")
    save_confusion_matrix(matrix.tolist(), output_dir)
    summary_path = BASE_DIR / "assets" / "test_evaluation.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(encoded, encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="用固定 OCT 检查点评估本地 test 集；不会启动训练或更新权重。")
    parser.add_argument("--data-dir", type=Path, default=TEST_DIR, help="包含四类别子目录的 test 目录")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="固定检查点，默认 best.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "outputs" / "evaluation")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.data_dir, args.checkpoint, args.device, args.output_dir, args.batch_size)
        print(f"准确率：{report['accuracy']:.2%}；宏平均 F1：{report['macro_f1']:.4f}")
        for category, scores in report["per_class"].items():
            print(f"{category}: precision={scores['precision']:.4f}, recall={scores['recall']:.4f}, "
                  f"F1={scores['f1']:.4f}, n={scores['support']}")
        print(INDEPENDENCE_NOTE)
        print(SCOPE_NOTE)
        print("结果已保存至指定输出目录，脱敏摘要已保存至 assets/test_evaluation.json。")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"评估失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
