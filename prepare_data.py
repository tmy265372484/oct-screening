"""解压 OCT 数据集并划分 train/val/test，保留示例图。"""

import argparse
import random
import shutil
import zipfile
from pathlib import Path

from config import BASE_DIR, CLASS_NAMES, DATA_DIR


VAL_PER_CLASS = 500
EXAMPLES_PER_CLASS = 5


def prepare_dataset(zip_path: Path, base: Path) -> None:
    """沿用每类 500 张验证图、5 张示例图和随机种子 42 的原流程。"""
    if not zip_path.is_file():
        raise FileNotFoundError(f"未找到数据集压缩包：{zip_path}；请用 --zip 指定路径。")

    rng = random.Random(42)
    base = base.resolve()
    print("解压中...")
    base.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        prefix = "CellData/OCT/"
        for name in archive.namelist():
            if name.startswith(prefix) and not name.endswith("/"):
                target = (base / name[len(prefix):]).resolve()
                if not target.is_relative_to(base):
                    raise ValueError(f"压缩包包含越界路径：{name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    print("解压完成")

    examples_dir = base / "examples"
    examples_dir.mkdir(exist_ok=True)
    for category in CLASS_NAMES:
        train_dir = base / "train" / category
        val_dir = base / "val" / category
        val_dir.mkdir(parents=True, exist_ok=True)

        images = list(train_dir.glob("*.jpeg"))
        rng.shuffle(images)
        for image in images[:VAL_PER_CLASS]:
            shutil.move(str(image), str(val_dir / image.name))

        remaining = list(train_dir.glob("*.jpeg"))
        examples = rng.sample(remaining, min(EXAMPLES_PER_CLASS, len(remaining)))
        for image in examples:
            shutil.copy(str(image), str(examples_dir / f"{category}_{image.name}"))

        print(
            f"{category}: 剩余train={len(list(train_dir.glob('*.jpeg')))}, "
            f"val={len(list(val_dir.glob('*.jpeg')))}, examples={len(examples)}"
        )

    print("\n=== 最终数据集结构 ===")
    for split in ["train", "val", "test"]:
        total = 0
        for category in CLASS_NAMES:
            directory = base / split / category
            count = len(list(directory.glob("*.jpeg"))) if directory.exists() else 0
            total += count
            print(f"  {split}/{category}: {count}")
        print(f"  --- {split} 合计: {total}")
    print(f"\n示例图: {examples_dir} ({len(list(examples_dir.glob('*.jpeg')))} 张)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip", type=Path, default=BASE_DIR / "data" / "ZhangLabData.zip",
        help="ZhangLabData.zip 路径（默认：项目目录/data/ZhangLabData.zip）",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DATA_DIR,
        help="输出数据目录（默认：项目目录/data/OCT）；会解压并划分数据，请勿对已完成的划分重复运行。",
    )
    args = parser.parse_args()
    try:
        prepare_dataset(args.zip, args.output_dir)
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
