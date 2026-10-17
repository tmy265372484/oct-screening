"""将 OCT 示例图拼成带类别标签的截图素材。"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from config import BASE_DIR, EXAMPLES_DIR


def make_collage(examples_dir: Path, output_path: Path) -> int:
    images = sorted(examples_dir.glob("*.jpeg"))
    if not images:
        raise FileNotFoundError(f"未找到 JPEG 示例图：{examples_dir}")

    columns = 4
    rows = max(5, (len(images) + columns - 1) // columns)
    tile_width, tile_height, label_height = 200, 150, 25
    canvas = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(images):
        row, column = divmod(index, columns)
        with Image.open(path) as source:
            image = source.convert("L").resize((tile_width, tile_height))
        x, y = column * tile_width, row * (tile_height + label_height)
        canvas.paste(image, (x, y))
        draw.text((x + 5, y + tile_height + 5), path.name.split("_")[0], fill="red")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print("OK:", output_path, len(images), "张")
    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples-dir", type=Path, default=EXAMPLES_DIR,
        help="示例图目录（默认：项目目录/data/OCT/examples）",
    )
    parser.add_argument(
        "--output", type=Path,
        default=BASE_DIR / "screenshots" / "02_数据集示例拼贴.png",
        help="拼贴输出路径（默认：项目目录/screenshots/02_数据集示例拼贴.png）",
    )
    args = parser.parse_args()
    try:
        make_collage(args.examples_dir, args.output)
    except FileNotFoundError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
