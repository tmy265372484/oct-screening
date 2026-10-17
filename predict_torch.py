"""Read-only OCT classification with the saved ResNet50 checkpoint.

Examples are a functional smoke test, not an independent test-set evaluation.
Only PyTorch's restricted ``weights_only=True`` loader is used. Never disable
this restriction for a downloaded checkpoint; this program rejects unsupported
checkpoint objects instead of executing their pickle contents.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
from typing import Any

from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torchvision import models, transforms

from config import BASE_DIR, CLASS_CN, EXAMPLES_DIR, IMG_SIZE


CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
DEFAULT_CHECKPOINT = BASE_DIR / "models" / "torch_resume" / "best.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SCORE_NOTE = "模型分数为四类别 softmax 输出，未经概率校准，不代表患病概率。"
USAGE_NOTE = "仅用于研究演示与辅助筛查；分类结果须由专业人员结合临床检查复核。"


class OCTPredictor:
    """Load once and reuse a model for classification and Grad-CAM.

    ``lock`` must also cover the Grad-CAM hook lifecycle when the same cached
    instance is shared by several Streamlit sessions.
    """

    def __init__(self, checkpoint_path=None, device="auto"):
        checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"找不到模型检查点：{checkpoint_path}。请先准备训练完成的 best.pt，"
                "或用 --checkpoint 指定已有的 latest.pt / 原始权重。"
            )
        if str(device) not in {"auto", "cpu", "cuda"}:
            raise ValueError("device 只支持 auto、cpu 或 cuda。")
        if str(device) == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前 Python 无法使用 CUDA，请选择 --device cpu。")
        self.device = torch.device(
            "cuda" if str(device) == "auto" and torch.cuda.is_available()
            else "cpu" if str(device) == "auto" else str(device)
        )
        # Small interactive requests are faster and avoid saturating the CPU.
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        self.lock = threading.RLock()
        try:
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError(
                f"无法安全读取检查点 {checkpoint_path.name}。需要兼容当前 PyTorch 的"
                "纯张量权重或本项目检查点；不会回退到不受限制的 pickle 加载。"
            ) from error
        if not isinstance(saved, dict):
            raise ValueError("检查点必须是本项目保存的字典或 state_dict。")
        if "model_state_dict" in saved:
            state = saved["model_state_dict"]
            metadata = saved.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("检查点 metadata 格式错误。")
            mapping = metadata.get("class_to_idx", {name: i for i, name in enumerate(CLASS_NAMES)})
            if mapping != {name: i for i, name in enumerate(CLASS_NAMES)}:
                raise ValueError("检查点类别顺序与 CNV / DME / DRUSEN / NORMAL 不一致。")
            size = metadata.get("img_size", IMG_SIZE)
            is_resume = "completed_epoch" in saved
            self.checkpoint_info = {
                "path": str(checkpoint_path),
                "filename": checkpoint_path.name,
                "format": "resume" if is_resume else "best",
                "epoch": saved.get("completed_epoch") if is_resume else saved.get("epoch"),
                # A full resume checkpoint records historical best_acc, not its
                # own val_acc. Keep the two explicitly separate in the UI.
                "val_acc": saved.get("val_acc"),
                "best_epoch": saved.get("best_epoch", saved.get("epoch")),
                "best_val_acc": saved.get("best_acc", saved.get("val_acc")),
                "augmentation_profile": saved.get("augmentation_profile"),
            }
        else:
            state, size = saved, IMG_SIZE
            self.checkpoint_info = {
                "path": str(checkpoint_path), "filename": checkpoint_path.name,
                "format": "legacy_state_dict", "epoch": None, "val_acc": None,
                "best_epoch": None, "best_val_acc": None, "augmentation_profile": None,
            }
        if (not isinstance(size, (tuple, list)) or len(size) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in size)):
            raise ValueError("检查点图像尺寸错误，预期为两个正整数 [高, 宽]。")
        self.image_size = tuple(size)
        if not isinstance(state, dict) or not state or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state.items()
        ):
            raise ValueError("检查点没有有效的模型 state_dict。")
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(CLASS_NAMES))
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError("权重与本项目四分类 ResNet50 架构不匹配。") from error
        self.model.requires_grad_(False)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Grayscale(3), transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def prepare_image(self, image: Image.Image) -> torch.Tensor:
        """Apply exactly the validation preprocessing (no augmentation)."""
        if not isinstance(image, Image.Image):
            raise TypeError("请传入 PIL.Image 图像。")
        # ImageFolder's loader converts source images to RGB before transforms.
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image) -> dict[str, Any]:
        with self.lock, torch.inference_mode():
            self.model.eval()
            scores = self.model(self.prepare_image(image)).softmax(dim=1)[0].cpu()
            if not bool(torch.isfinite(scores).all()):
                raise RuntimeError("模型输出了无效数值，请检查输入图像及模型权重。")
            predicted_index = int(scores.argmax().item())
            name = CLASS_NAMES[predicted_index]
            return {
                "predicted_class": name,
                "predicted_label_cn": CLASS_CN[name],
                "predicted_index": predicted_index,
                "probabilities": {category: float(scores[index]) for index, category in enumerate(CLASS_NAMES)},
                "model_score": float(scores[predicted_index]),
                "checkpoint": dict(self.checkpoint_info),
                "image_size": list(image.size),
                "score_note": SCORE_NOTE,
                "usage_note": USAGE_NOTE,
            }


def find_example_images(directory: str | Path | None = None) -> list[Path]:
    """Choose one local illustrative image per class; never read test images."""
    if directory is None:
        directory = EXAMPLES_DIR if EXAMPLES_DIR.is_dir() else BASE_DIR / "data" / "OCT" / "examples"
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"找不到示例图目录：{directory}。可使用 --examples-dir 指定。")
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    selected = []
    for category in CLASS_NAMES:
        matches = [path for path in files if path.parent.name.upper() == category or path.stem.upper() == category
                   or path.stem.upper().startswith((category + "_", category + "-"))]
        if not matches:
            raise FileNotFoundError(f"示例目录缺少 {category} 图像：{directory}")
        selected.append(matches[0])
    return selected


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="使用已有 OCT 模型做预测；不会启动训练。")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, help="要预测的本地图像")
    inputs.add_argument("--examples", action="store_true", help="预测四类本地示例，默认保存 outputs/predictions.json")
    parser.add_argument("--examples-dir", type=Path, help="本地示例目录（与 --examples 配合使用）")
    parser.add_argument("--checkpoint", type=Path, help="模型路径，默认 models/torch_resume/best.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, help="预测 JSON 保存路径")
    args = parser.parse_args(argv)
    try:
        if args.examples_dir is not None and not args.examples:
            parser.error("--examples-dir 需要同时指定 --examples。")
        paths = find_example_images(args.examples_dir) if args.examples else [args.image]
        predictor = OCTPredictor(args.checkpoint, args.device)
        results = []
        for path in paths:
            with Image.open(path) as image:
                image.load()
                result = predictor.predict(image)
            result["image_path"] = str(path.resolve())
            results.append(result)
            print(f"{path.name}: {result['predicted_class']}（{result['predicted_label_cn']}），模型分数 {result['model_score']:.2%}")
            for category, score in result["probabilities"].items():
                print(f"  {category}: {score:.2%}")
        output = args.output or (BASE_DIR / "outputs" / "predictions.json" if args.examples else None)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            report = {
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "purpose": "本地示例功能验证，不作为独立测试集准确率",
                "score_note": SCORE_NOTE, "usage_note": USAGE_NOTE,
                "device": str(predictor.device), "results": results,
            }
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"结果已保存：{output.resolve()}")
        print(SCORE_NOTE)
        return 0
    except (OSError, ValueError, RuntimeError, UnidentifiedImageError) as error:
        print(f"预测失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
