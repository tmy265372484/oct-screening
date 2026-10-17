"""Export an anonymized board from actual saved local-example predictions.

This script does not infer, train, or evaluate a test set. Generate the source
JSON first with ``python predict_torch.py --examples --examples-dir assets/demo
--device cpu`` (or use the original local example directory).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_DIR = Path(__file__).resolve().parents[1]
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_CN = {"CNV": "脉络膜新生血管", "DME": "黄斑水肿", "DRUSEN": "玻璃膜疣", "NORMAL": "正常"}


def dataset_label_from_path(image_path: Path) -> str:
    """Accept exact demo stems, original prefixes, or class directories only.

    Never infer ground truth from the model's prediction, and reject conflicting
    directory/filename labels instead of silently choosing one.
    """
    stem, parent = image_path.stem.upper(), image_path.parent.name.upper()
    labels = {name for name in CLASSES
              if parent == name or stem == name or stem.startswith((name + "_", name + "-"))}
    if len(labels) != 1:
        raise ValueError("示例图片路径必须含唯一的明确类别标签（如 CNV.png、CNV_示例.jpeg 或 CNV/图片.png）。")
    return labels.pop()


def checkpoint_metadata(row: dict) -> dict:
    checkpoint = row.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("预测记录缺少检查点元数据，不能声明所用模型。")
    filename = checkpoint.get("filename")
    if (not isinstance(filename, str) or not filename
            or any(character in filename for character in ("/", "\\", ":"))):
        raise ValueError("检查点 filename 必须是明确的脱敏文件名。")
    for field in ("epoch", "best_epoch"):
        value = checkpoint.get(field)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise ValueError(f"检查点 {field} 不是有效轮次。")
    for field in ("val_acc", "best_val_acc"):
        value = checkpoint.get(field)
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)
                                  or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"检查点 {field} 不是有效准确率。")
    return checkpoint


def font(size, bold=False):
    for path in (Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
                 Path("C:/Windows/Fonts/simhei.ttf"),
                 Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def export_evidence(project_dir: Path):
    source = json.loads((project_dir / "outputs/predictions.json").read_text(encoding="utf-8"))
    by_class = {}
    shared_checkpoint = None
    for row in source.get("results", []):
        image_path = Path(row["image_path"]).expanduser()
        if not image_path.is_absolute():
            image_path = project_dir / image_path
        dataset_label = dataset_label_from_path(image_path)
        if dataset_label in by_class:
            raise ValueError("每类应恰有一张示例；请不要混合多次预测记录。")
        checkpoint = checkpoint_metadata(row)
        if shared_checkpoint is None:
            shared_checkpoint = dict(checkpoint)
        elif checkpoint != shared_checkpoint:
            raise ValueError("四条预测使用的检查点元数据不一致，不能合并为同一模型示例板。")
        if dataset_label in CLASSES and dataset_label not in by_class:
            score = row.get("model_score")
            probabilities = row.get("probabilities", {})
            if (not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1
                    or set(probabilities) != set(CLASSES)
                    or any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
                           for value in probabilities.values())
                    or abs(sum(probabilities.values()) - 1) > 1e-5
                    or row["predicted_class"] != max(probabilities, key=probabilities.get)
                    or abs(score - probabilities[row["predicted_class"]]) > 1e-6):
                raise ValueError("预测记录数值不一致，停止生成以免伪造展示。")
            if not image_path.is_file():
                raise FileNotFoundError("预测记录引用的原图不存在；请用 --examples-dir assets/demo 重新生成本机预测，避免把旧分数贴到其它图片。")
            by_class[dataset_label] = (row, image_path)
    if set(by_class) != set(CLASSES):
        raise ValueError("需要四类各一条真实示例输出。")
    board = Image.new("RGB", (1800, 1000), "#eef4f8")
    draw = ImageDraw.Draw(board)
    heading, subtitle, card_title = font(40, True), font(24), font(27, True)
    body, small, big = font(22), font(20), font(36, True)
    draw.text((52, 32), "OCT 四类示例 · 实际模型预测", font=heading, fill="#17364f")
    draw.text((54, 98), "本地示例的功能验证，非独立测试集准确率；图片与分数来自真实已保存输出。", font=subtitle, fill="#546d80")
    draw.rounded_rectangle((1345, 33, 1748, 85), radius=16, fill="#ddebef")
    epoch = shared_checkpoint.get("epoch")
    checkpoint_label = f"{shared_checkpoint['filename']}  /  " + (f"第{epoch}轮" if epoch is not None else "轮次未记录")
    badge_font = small
    for size in range(20, 11, -1):
        badge_font = font(size)
        if draw.textbbox((0, 0), checkpoint_label, font=badge_font)[2] <= 360:
            break
    if draw.textbbox((0, 0), checkpoint_label, font=badge_font)[2] > 360:
        raise ValueError("检查点文件名过长，无法清晰展示。")
    draw.text((1367, 45), checkpoint_label, font=badge_font, fill="#246477")
    sanitized = []
    for index, dataset_label in enumerate(CLASSES):
        row, image_path = by_class[dataset_label]
        x, y = 52 + index * 432, 174
        draw.rounded_rectangle((x, y, x + 400, y + 640), radius=16, fill="white")
        draw.text((x + 21, y + 18), dataset_label, font=card_title, fill="#17364f")
        draw.text((x + 21, y + 58), CLASS_CN[dataset_label], font=body, fill="#546d80")
        draw.rounded_rectangle((x + 18, y + 109, x + 382, y + 411), radius=8, fill="#0d1922")
        with Image.open(image_path) as image:
            original = image.convert("RGB").convert("L").convert("RGB")
            display = ImageOps.contain(original, (364, 300), Image.Resampling.LANCZOS)
            board.paste(display, (x + 18 + (364 - display.width) // 2,
                                  y + 110 + (300 - display.height) // 2))
        predicted = row["predicted_class"]
        draw.text((x + 21, y + 433), f"数据集真实标签：{dataset_label}", font=small, fill="#60798b")
        draw.text((x + 21, y + 471), f"模型预测：{predicted}", font=body, fill="#17364f")
        draw.text((x + 21, y + 515), f"{row['model_score']:.2%}", font=big, fill="#0c8f83")
        draw.text((x + 213, y + 531), "模型分数", font=small, fill="#60798b")
        draw.rounded_rectangle((x + 21, y + 578, x + 379, y + 588), radius=5, fill="#e5eff1")
        draw.rounded_rectangle((x + 21, y + 578, x + 21 + round(358 * row["model_score"]), y + 588), radius=5, fill="#0c9e90")
        draw.text((x + 21, y + 605), "四类别 softmax · 未经临床校准", font=small, fill="#60798b")
        sanitized.append({
            "example_id": f"example-{dataset_label.lower()}",
            "dataset_label": dataset_label,
            "predicted_class": predicted,
            "predicted_label_cn": CLASS_CN[predicted],
            "model_score": row["model_score"],
            "probabilities": row["probabilities"],
            "checkpoint": {key: row["checkpoint"].get(key)
                           for key in ("filename", "epoch", "val_acc", "best_epoch", "best_val_acc")},
        })
    draw.text((54, 855), "说明：仅展示4张既有示例的实际输出；不能将本板结果当作模型整体准确率。", font=subtitle, fill="#4f687b")
    draw.text((54, 903), "数据集标签依据现有示例文件分类；模型分数不等于患病概率，分类结果须由专业人员复核。", font=body, fill="#4f687b")
    draw.text((54, 951), "本图由程序生成，用于比赛功能展示；不是历史运行界面截图。", font=small, fill="#6b8191")
    screenshots = project_dir / "screenshots"
    screenshots.mkdir(exist_ok=True)
    target = screenshots / "08_四类示例预测.png"
    board.save(target)
    gradcam_source = project_dir / "outputs/gradcam/cnv_example/comparison.png"
    if not gradcam_source.is_file():
        raise FileNotFoundError("缺少已验证的 CNV Grad-CAM 对照板，请先生成它。")
    gradcam_target = screenshots / "09_GradCAM模型关注对照.png"
    shutil.copyfile(gradcam_source, gradcam_target)
    summary = {
        "schema_version": 1,
        "purpose": "四类本地示例功能验证，不是独立测试集评估或整体准确率",
        "generated_at": source.get("generated_at"),
        "model": {"checkpoint": shared_checkpoint["filename"],
                  **{key: shared_checkpoint.get(key)
                     for key in ("epoch", "val_acc", "best_epoch", "best_val_acc", "format")}},
        "score_note": source.get("score_note"),
        "results": sanitized,
        "source": "outputs/predictions.json",
    }
    assets = project_dir / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "example_predictions.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return target, gradcam_target


def main(argv=None):
    parser = argparse.ArgumentParser(description="从真实示例输出制作脱敏比赛证据。")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    args = parser.parse_args(argv)
    try:
        for path in export_evidence(args.project_dir.resolve()):
            print(f"已生成：{path.name}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"未能导出示例证据：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
