"""PyTorch Grad-CAM for the OCT classifier.

The color map indicates relative model attention, not a lesion boundary,
segmentation, or a calibrated probability. No training or weight update occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
import torch.nn.functional as F

from config import CLASS_CN, CLASS_NAMES

if TYPE_CHECKING:
    from predict_torch import OCTPredictor


class GradCAMError(RuntimeError):
    """A truthful attention map could not be produced for this input/target."""


def _target_index(target_class: str | int | None, scores: torch.Tensor) -> int:
    if target_class is None:
        return int(scores.argmax().item())
    if isinstance(target_class, str):
        name = target_class.strip().upper()
        if name in CLASS_NAMES:
            return CLASS_NAMES.index(name)
        if name.isdigit():
            target_class = int(name)
        else:
            raise ValueError(f"未知类别 {target_class!r}；可选：{', '.join(CLASS_NAMES)}。")
    if isinstance(target_class, (int, np.integer)) and not isinstance(target_class, bool):
        if 0 <= int(target_class) < len(CLASS_NAMES):
            return int(target_class)
    raise ValueError(f"target_class 必须是类别名或 0–{len(CLASS_NAMES) - 1} 的索引。")


def colorize_heatmap(heatmap: np.ndarray) -> Image.Image:
    """Map normalized attention to a blue/cyan/yellow/red RGB image."""
    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("热力图必须是有限数值组成的二维数组。")
    values = np.clip(values, 0, 1)
    channels = [np.clip(1.5 - np.abs(4 * values - offset), 0, 1)
                for offset in (3, 2, 1)]
    return Image.fromarray(np.rint(np.stack(channels, axis=-1) * 255).astype(np.uint8))


def generate_gradcam(
    predictor: OCTPredictor,
    image: Image.Image,
    target_class: str | int | None = None,
    alpha: float = 0.45,
) -> dict:
    """Return original-sized attention and overlay for a single OCT image.

    Gradients are taken from the selected logit to ``model.layer4[-1]``.
    Frozen model parameters are supported: only the input requires gradients.
    The predictor's shared lock prevents concurrent predictions/hooks from
    interfering. Hooks are removed even when a forward/gradient step fails.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image 必须是 PIL.Image 图片。")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha 必须在 0 到 1 之间。")
    # Match ImageFolder and the predictor exactly; callers may orient an image
    # before passing the same PIL object to both prediction and explanation.
    original = image.convert("RGB").convert("L").convert("RGB")
    activations: list[torch.Tensor] = []

    def capture_activation(_module, _inputs, output):
        activations.append(output)

    with predictor.lock, torch.inference_mode(False), torch.enable_grad():
        if predictor.model.training:
            raise GradCAMError("Grad-CAM 要求模型处于 eval 模式，以避免改变 BatchNorm 状态。")
        try:
            target_layer = predictor.model.layer4[-1]
        except (AttributeError, IndexError) as exc:
            raise GradCAMError("模型缺少 ResNet50 的 layer4[-1] 特征层。") from exc
        handle = target_layer.register_forward_hook(capture_activation)
        try:
            batch = predictor.prepare_image(image).detach().requires_grad_(True)
            logits = predictor.model(batch)
            if logits.shape != (1, len(CLASS_NAMES)) or not torch.isfinite(logits).all():
                raise GradCAMError("模型输出格式或数值异常，无法生成关注区域。")
            probabilities = logits.softmax(dim=1)[0]
            target_index = _target_index(target_class, probabilities)
            predicted_index = int(probabilities.argmax().item())
            if len(activations) != 1:
                raise GradCAMError("未能获取唯一的目标层特征。")
            features = activations[0]
            if features.ndim != 4 or not features.requires_grad:
                raise GradCAMError("目标层特征没有可用梯度，无法生成 Grad-CAM。")
            gradients = torch.autograd.grad(
                outputs=logits[0, target_index],
                inputs=features,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            if not torch.isfinite(gradients).all():
                raise GradCAMError("目标层梯度包含异常数值。")
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * features).sum(dim=1, keepdim=True).relu()
            if not torch.isfinite(cam).all() or float(cam.max().item()) <= 1e-12:
                raise GradCAMError("该图像和目标类别的 Grad-CAM 没有正响应，未生成热图。")
            cam = cam / cam.max()
            cam = F.interpolate(cam, size=(original.height, original.width),
                                mode="bilinear", align_corners=False)
            heatmap = cam[0, 0].detach().float().cpu().numpy().copy()
            scores = probabilities.detach().float().cpu().tolist()
        finally:
            handle.remove()
            activations.clear()

    heatmap = np.clip(heatmap, 0, 1).astype(np.float32, copy=False)
    colored = colorize_heatmap(heatmap)
    return {
        "original": original,
        "heatmap": heatmap,
        "heatmap_image": colored,
        "overlay": Image.blend(original, colored, float(alpha)),
        "target_class": CLASS_NAMES[target_index],
        "target_label_cn": CLASS_CN[CLASS_NAMES[target_index]],
        "target_index": target_index,
        "predicted_class": CLASS_NAMES[predicted_index],
        "predicted_index": predicted_index,
        "model_score": float(scores[target_index]),
        "probabilities": dict(zip(CLASS_NAMES, map(float, scores))),
        "target_layer": "layer4[-1]",
        "alpha": float(alpha),
        "checkpoint": dict(predictor.checkpoint_info),
        "notice": "Grad-CAM 显示模型关注区域，不代表病灶边界或分割结果；颜色按单图相对归一化。模型分数不等于临床患病概率。",
    }


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf"),
                 Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def make_comparison_board(result: dict) -> Image.Image:
    """Create an honest, exportable original/attention/overlay comparison."""
    board = Image.new("RGB", (1600, 900), "#eef3f8")
    draw = ImageDraw.Draw(board)
    title, body, small = _font(34), _font(23), _font(20)
    draw.text((38, 28), "OCT · Grad-CAM 模型关注区域", font=title, fill="#15334f")
    label = result["target_label_cn"]
    draw.text((40, 88),
              f"解释目标：{result['target_class']}（{label}）   模型分数：{result['model_score']:.2%}   目标层：layer4[-1]",
              font=body, fill="#415a70")
    panels = [("原始 OCT（灰度）", result["original"]),
              ("相对关注强度", result["heatmap_image"]),
              ("关注区域叠加", result["overlay"])]
    for index, (caption, picture) in enumerate(panels):
        x = 38 + index * 519
        draw.rounded_rectangle((x, 146, x + 500, 718), radius=14, fill="white")
        draw.text((x + 18, 165), caption, font=body, fill="#15334f")
        fit = ImageOps.contain(picture, (464, 484), Image.Resampling.LANCZOS)
        px, py = x + (500 - fit.width) // 2, 218 + (484 - fit.height) // 2
        board.paste(fit, (px, py))
    gradient = np.tile(np.linspace(0, 1, 400, dtype=np.float32), (18, 1))
    board.paste(colorize_heatmap(gradient), (600, 744))
    draw.text((480, 738), "低关注", font=small, fill="#415a70")
    draw.text((1020, 738), "高关注", font=small, fill="#415a70")
    draw.text((40, 796), "仅解释模型关注区域；不能据此确定病灶边界或计算病灶面积。", font=body, fill="#70532e")
    draw.text((40, 842), "模型分数未经临床校准。本图用于科研与辅助复核，不作为独立诊断结论。", font=small, fill="#536a7d")
    return board


def save_gradcam_outputs(result: dict, output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, filename in (("original", "original.png"), ("heatmap_image", "attention.png"),
                          ("overlay", "overlay.png")):
        path = output_dir / filename
        result[key].save(path)
        paths[key] = str(path.resolve())
    board_path = output_dir / "comparison.png"
    make_comparison_board(result).save(board_path)
    paths["comparison"] = str(board_path.resolve())
    npy_path = output_dir / "attention.npy"
    np.save(npy_path, result["heatmap"], allow_pickle=False)
    paths["heatmap_data"] = str(npy_path.resolve())
    metadata = {key: value for key, value in result.items()
                if key not in {"original", "heatmap", "heatmap_image", "overlay"}}
    metadata.update({"created_at": datetime.now(timezone.utc).isoformat(),
                     "original_size": list(result["original"].size),
                     "method": "Grad-CAM: spatial mean gradient weights, positive logit contribution",
                     "outputs": paths})
    metadata_path = output_dir / "gradcam.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["metadata"] = str(metadata_path.resolve())
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成真实 PyTorch Grad-CAM 模型关注区域（不会训练模型）。")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--target-class", default=None, help="默认解释预测类别；也可填 CNV/DME/DRUSEN/NORMAL 或索引。")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gradcam"))
    args = parser.parse_args(argv)
    try:
        from predict_torch import OCTPredictor
        predictor = OCTPredictor(checkpoint_path=args.checkpoint, device=args.device)
        with Image.open(args.image) as source:
            source.load()
            result = generate_gradcam(predictor, source, args.target_class, args.alpha)
        paths = save_gradcam_outputs(result, args.output_dir)
        print(json.dumps({"target_class": result["target_class"], "model_score": result["model_score"],
                          "notice": result["notice"], "outputs": paths}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"Grad-CAM 生成失败：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
