"""Generate a real-image OCT augmentation comparison, without training a model."""

from pathlib import Path
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image, ImageDraw, ImageFont

from config import DATA_DIR, IMG_SIZE
from train_resume import AUGMENTATION_CONFIG, AUGMENTATION_PROFILE, build_transforms


def main():
    torch.set_num_threads(1)
    train_tf, val_tf = build_transforms(IMG_SIZE)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def unnormalize(tensor):
        pixels = ((tensor.cpu() * std + mean).clamp(0, 1) * 255).round().byte()
        return Image.fromarray(pixels.permute(1, 2, 0).numpy(), mode="RGB")

    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msyh.ttc"
    bold_path = font_path.with_name("msyhbd.ttc")

    def font(size, bold=False):
        return ImageFont.truetype(str(bold_path if bold and bold_path.exists() else font_path), size)

    width, height = 1920, 1840
    canvas = Image.new("RGB", (width, height), "#F5F7FA")
    draw = ImageDraw.Draw(canvas)
    dark, muted, accent = "#172B3A", "#536876", "#007C83"
    draw.rectangle((0, 0, width, 12), fill=accent)
    draw.text((72, 46), "训练数据增强效果示例（程序生成）", font=font(50, True), fill=dark)
    draw.text((74, 122), "真实 OCT 训练图像 · 调用当前训练增强链 · 标准化后反归一化显示", font=font(28), fill=muted)

    tile_size, xs = 320, [336, 708, 1080, 1452]
    headers = ["原图（统一预处理）", "增强示例 1", "增强示例 2", "增强示例 3"]
    for x, text in zip(xs, headers):
        bounds = draw.textbbox((0, 0), text, font=font(28, True))
        text_width = bounds[2] - bounds[0]
        draw.text((x + (tile_size - text_width) // 2, 201), text, font=font(28, True), fill=dark)
    draw.line((72, 248, 1848, 248), fill="#D6E0E6", width=2)

    classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
    labels = ["脉络膜新生血管", "黄斑水肿", "玻璃膜疣", "正常"]
    metadata = []
    for row, (category, label) in enumerate(zip(classes, labels)):
        candidates = sorted(
            path for path in (Path(DATA_DIR) / "train" / category).iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        )
        if not candidates:
            raise FileNotFoundError(f"No training image available for {category}")
        with Image.open(candidates[0]) as source:
            image = source.convert("RGB")
        seeds = [20261017 + row * 100 + column for column in range(3)]
        panels = [unnormalize(val_tf(image))]
        for seed in seeds:
            torch.manual_seed(seed)
            panels.append(unnormalize(train_tf(image)))
        y = 270 + row * 354
        draw.rounded_rectangle((72, y + 104, 288, y + 206), radius=12, fill="#E5F1F2")
        draw.text((92, y + 114), category, font=font(30, True), fill=accent)
        draw.text((92, y + 160), label, font=font(23), fill=muted)
        for x, panel in zip(xs, panels):
            draw.rectangle((x - 3, y - 3, x + tile_size + 2, y + tile_size + 2), fill="#CBD6DE")
            canvas.paste(panel.resize((tile_size, tile_size), Image.Resampling.BILINEAR), (x, y))
        metadata.append({"class": category, "random_seeds": seeds})

    draw.line((72, 1690, 1848, 1690), fill="#D6E0E6", width=2)
    draw.text((74, 1710), "保留：50% 水平翻转、±10° 旋转；新增：±3% 平移、0.97～1.03 缩放。", font=font(27), fill=dark)
    draw.text((74, 1755), "新增：亮度 / 对比度各 0.9～1.1 倍。验证集只做灰度、缩放与标准化，不使用随机增强。", font=font(27), fill=dark)
    output = Path(__file__).with_name("06_数据增强效果对照.png")
    canvas.save(output)
    details = {
        "description": "程序生成的真实训练图像增强样例；不是操作截图或准确率证明。",
        "augmentation_profile": AUGMENTATION_PROFILE,
        "augmentation_config": AUGMENTATION_CONFIG,
        "input_size": list(IMG_SIZE),
        "image_selection": "每类训练目录内按文件名排序的第一张；不披露患者文件名。",
        "samples": metadata,
        "output_size": [width, height],
    }
    output.with_suffix(".json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {output} ({width} x {height})")


if __name__ == "__main__":
    main()
