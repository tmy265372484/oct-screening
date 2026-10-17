"""Small, UI-independent helpers. Uploaded images are never persisted here."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import warnings

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 12_000_000


def decode_image(payload: bytes) -> Image.Image:
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("请选择不超过 10 MB 的 JPEG 或 PNG 图像。")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as check:
                if check.format not in {"JPEG", "PNG"}:
                    raise ValueError("图像实际格式必须是 JPEG 或 PNG。")
                if check.width * check.height > MAX_PIXELS:
                    raise ValueError("图像超过 1200 万像素，请先缩小。")
                check.verify()
            with Image.open(BytesIO(payload)) as source:
                source.load()
                # No EXIF rotation: this matches the training ImageFolder loader.
                return source.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("图像无法解码，请提供完整的 JPEG 或 PNG 文件。") from error


def image_sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def training_summary() -> dict:
    path = ROOT / "assets" / "training_summary.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def public_prediction(prediction: dict) -> dict:
    """Keep checkpoint identity, but don't expose this computer's home path."""
    result = dict(prediction)
    result["checkpoint"] = dict(prediction["checkpoint"])
    result["checkpoint"].pop("path", None)
    return result
