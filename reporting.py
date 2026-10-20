"""Offline, escaped review-draft exports; no prediction, network or file I/O.

``build_report_html(record)`` and ``build_report_markdown(record)`` accept a
record containing ``case_id``, ``created_at``, ``source_name``, ``source_sha256``,
``prediction`` (the OCTPredictor result), ``note``, and optional PIL ``original``
and ``overlay`` images. HTML embeds its images and can be printed offline.
The returned strings are not written to disk by this module.
"""

from __future__ import annotations

import base64
from html import escape
from io import BytesIO
import math
import re
from typing import Any

from PIL import Image


CLASS_NAMES = ("CNV", "DME", "DRUSEN", "NORMAL")
CLASS_CN = {"CNV": "脉络膜新生血管", "DME": "黄斑水肿", "DRUSEN": "玻璃膜疣", "NORMAL": "正常"}
SCORE_NOTE = "模型分数为四类别 softmax 输出，未经概率校准，不代表患病概率。"
CAM_NOTE = "Grad-CAM 展示模型对所选类别的相对关注区域，不是病灶边界、分割结果或临床定位结论。"
REVIEW_NOTE = "本报告为研究演示和辅助筛查草稿，未经医生复核，不作为独立诊断或治疗依据。"


def _text(value: Any, default="未记录") -> str:
    return default if value is None or value == "" else str(value)


def _percent(value: Any) -> str:
    try:
        if isinstance(value, bool):
            return "未记录"
        score = float(value)
    except (ValueError, TypeError, OverflowError):
        return "未记录"
    return f"{score:.2%}" if math.isfinite(score) and 0 <= score <= 1 else "未记录"


def _epoch(value: Any) -> str:
    # Do not infer training completion from a best checkpoint's epoch.
    return f"第 {value} 轮" if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else "未记录"


def _details(record: dict) -> dict:
    if not isinstance(record, dict) or not isinstance(record.get("prediction"), dict):
        raise ValueError("报告 record 必须包含 prediction 字典。")
    prediction = record["prediction"]
    checkpoint = prediction.get("checkpoint") or {}
    if not isinstance(checkpoint, dict):
        raise ValueError("prediction.checkpoint 必须为字典。")
    scores = prediction.get("probabilities") or {}
    if not isinstance(scores, dict):
        raise ValueError("prediction.probabilities 必须为字典。")
    name = _text(prediction.get("predicted_class"))
    label = _text(prediction.get("predicted_label_cn"), CLASS_CN.get(name, "未记录"))
    filename = checkpoint.get("filename")
    if not filename and checkpoint.get("path"):
        filename = str(checkpoint["path"]).replace("\\", "/").rsplit("/", 1)[-1]
    source_size = record.get("source_size")
    if (not isinstance(source_size, (list, tuple)) or len(source_size) != 2
            or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in source_size)):
        source_size = None
    slice_total = record.get("slice_total", 1)
    slice_index = record.get("slice_index", 1)
    series_id = record.get("series_id")
    slice_label = (f"第 {slice_index} / {slice_total} 张"
                   if isinstance(slice_total, int) and slice_total > 1
                   and isinstance(slice_index, int) and 1 <= slice_index <= slice_total
                   else "单张影像")
    annotations = record.get("annotations")
    annotation_count = len(annotations) if isinstance(annotations, list) else 0
    return {
        "case_id": _text(record.get("case_id")),
        "created_at": _text(record.get("created_at")),
        "source_name": _text(record.get("source_name")),
        "source_sha256": _text(record.get("source_sha256")),
        "series_id": _text(series_id, "未关联序列"),
        "slice_label": slice_label,
        "source_size": (f"{source_size[0]} × {source_size[1]} px" if source_size else "未记录"),
        "annotation_count": str(annotation_count),
        "note": _text(record.get("note"), "尚未填写；等待专业人员复核。"),
        "class_name": name,
        "class_label": label,
        "model_score": _percent(prediction.get("model_score")),
        "scores": [(category, CLASS_CN[category], _percent(scores.get(category))) for category in CLASS_NAMES],
        "checkpoint_name": _text(filename),
        "checkpoint_format": _text(checkpoint.get("format")),
        "checkpoint_epoch": _epoch(checkpoint.get("epoch")),
        "checkpoint_val_acc": _percent(checkpoint.get("val_acc")),
        "best_epoch": _epoch(checkpoint.get("best_epoch")),
        "best_val_acc": _percent(checkpoint.get("best_val_acc")),
    }


def _png_data(image: Image.Image) -> str:
    if not isinstance(image, Image.Image):
        raise TypeError("报告 original / overlay 必须是 PIL.Image 或 None。")
    # Work on a copy; discard metadata and limit export size, not model input.
    preview = image.convert("RGB")
    preview.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
    stream = BytesIO()
    preview.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def build_report_html(record: dict) -> str:
    """Return a standalone, printable HTML review draft with embedded images."""
    data = _details(record)

    def cell(key: str) -> str:
        return escape(data[key], quote=True)

    rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{escape(label)}</td><td>{escape(score)}</td></tr>"
        for name, label, score in data["scores"]
    )
    figures = []
    for key, caption in (("original", "输入影像"), ("annotated", "用户标注后的原始影像"),
                         ("overlay", "Grad-CAM 模型关注叠加图")):
        if record.get(key) is not None:
            figures.append(f'<figure><img alt="{caption}" src="{_png_data(record[key])}"><figcaption>{caption}</figcaption></figure>')
    image_section = '<section><h2>影像与模型关注图</h2><div class="images">' + "".join(figures) + "</div></section>" if figures else ""
    note = cell("note").replace("\n", "<br>")
    review_section = '<section><h2>人工复核记录（与模型提示分别保存）</h2>'
    for review in record.get('reviews', []):
        fields = [f"第 {review['revision']} 次复核", review['reviewed_at'], review['reviewer'],
                  review['decision'], review.get('category') or '未定', review['rationale']]
        review_section += '<p>' + '<br>'.join(escape(str(x)) for x in fields) + '</p>'
    review_section += '<p>本机填写记录，未经身份认证，不构成电子签名；不替代正式临床诊断。</p></section>'
    status = '已记录人工复核 · 未电子签署' if record.get('reviews') else '医生复核草稿 · 未签署'
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>OCT 筛查复核草稿 · {cell('case_id')}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#eef2f6;color:#172b42;font:14px/1.65 'Microsoft YaHei',Arial,sans-serif}}
main{{max-width:1040px;margin:28px auto;padding:40px;background:white}}header{{border-bottom:3px solid #137c94;padding-bottom:16px}}
h1{{font-size:27px;margin:0 0 4px}}h2{{font-size:17px;margin:0 0 12px}}.draft{{color:#956018;background:#fff4df;display:inline-block;padding:3px 10px;border-radius:4px}}
section{{margin-top:24px}}.meta{{display:grid;grid-template-columns:1fr 1fr;gap:8px 24px;margin:18px 0}}.wide{{grid-column:1/-1;overflow-wrap:anywhere}}
.result{{padding:18px 22px;background:#edf6f8;border-left:4px solid #137c94}}.result strong{{font-size:22px}}
table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:8px 12px;border:1px solid #d9e3eb}}th{{background:#f2f6fa}}
.images{{display:flex;gap:16px;align-items:flex-start}}figure{{margin:0;flex:1;min-width:0}}img{{width:100%;height:auto;display:block;background:#080b0f}}figcaption{{text-align:center;color:#617184;margin-top:6px}}
.notice{{color:#58697c;font-size:12px}}.review{{min-height:80px;border:1px solid #d9e3eb;padding:14px;overflow-wrap:anywhere}}
.sign{{display:flex;justify-content:space-between;border-top:1px solid #d9e3eb;margin-top:28px;padding-top:16px}}footer{{margin-top:20px;color:#63768b;font-size:12px}}
@media print{{body{{background:white}}main{{margin:0;padding:12mm;max-width:none}}section,figure,.result{{break-inside:avoid}}@page{{size:A4;margin:8mm}}}}
</style></head><body><main>
<header><h1>OCT 智能辅助筛查报告</h1><span class="draft">{status}</span></header>
<div class="meta"><div><b>病例编号：</b>{cell('case_id')}</div><div><b>生成时间：</b>{cell('created_at')}</div>
<div><b>序列编号：</b>{cell('series_id')}</div><div><b>序列位置：</b>{cell('slice_label')}</div>
<div><b>输入图像尺寸：</b>{cell('source_size')}</div><div><b>用户标注数：</b>{cell('annotation_count')}</div>
<div class="wide"><b>来源文件：</b>{cell('source_name')}</div><div class="wide"><b>图像 SHA-256：</b>{cell('source_sha256')}</div></div>
<section class="result"><h2>模型分类提示</h2><strong>{cell('class_name')} · {cell('class_label')}</strong><div>该类别模型分数：{cell('model_score')}</div></section>
<p class="notice">{escape(SCORE_NOTE)}</p>
<section><h2>四类别模型分数</h2><table><thead><tr><th>类别</th><th>中文名称</th><th>模型分数</th></tr></thead><tbody>{rows}</tbody></table></section>
{image_section}
<p class="notice">{escape(CAM_NOTE)}</p>
<section><h2>阅读提示</h2><p>请结合完整 OCT 扫描与临床检查，由专业人员复核。正常类提示也不能排除其他疾病。</p></section>
<section><h2>复核备注</h2><div class="review">{note}</div></section>
{review_section}
<div class="sign"><span>复核医生：________________</span><span>复核日期：________________</span></div>
<footer>{escape(REVIEW_NOTE if not record.get('reviews') else '本报告含本机人工复核记录，不作为独立诊断或治疗依据。')}<br>分析版本：{cell('checkpoint_name')} / {cell('checkpoint_epoch')} · 仅供结果追溯</footer>
</main></body></html>"""


def _markdown(value: str) -> str:
    # Escape raw HTML first, then Markdown structure including table separators.
    value = escape(value, quote=True)
    value = re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", value)
    return value.replace("\r", "").replace("\n", "<br>")


def build_report_markdown(record: dict) -> str:
    """Return a text-first draft; the optional images are embedded in HTML only."""
    data = _details(record)
    out = ["# OCT 智能辅助筛查报告", "", "**医生复核草稿 · 未签署**", ""]
    for title, key in (("病例编号", "case_id"), ("生成时间", "created_at"), ("来源文件", "source_name"), ("图像 SHA-256", "source_sha256")):
        out.append(f"- {title}：{_markdown(data[key])}")
    out += [f"- 序列编号：{_markdown(data['series_id'])}",
            f"- 序列位置：{_markdown(data['slice_label'])}",
            f"- 输入图像尺寸：{_markdown(data['source_size'])}",
            f"- 用户标注数：{data['annotation_count']}"]
    out += ["", "## 模型分类提示", "", f"{_markdown(data['class_name'])} · {_markdown(data['class_label'])}",
            f"该类别模型分数：{data['model_score']}", "", SCORE_NOTE, "", "## 四类别模型分数", "",
            "| 类别 | 中文名称 | 模型分数 |", "| --- | --- | --- |"]
    out += [f"| {name} | {label} | {score} |" for name, label, score in data["scores"]]
    out += ["", "## 图像说明", "", "影像与关注叠加图请查看同次导出的 HTML 报告。", CAM_NOTE,
            "", "## 阅读提示", "",
            "请结合完整 OCT 扫描与临床检查，由专业人员复核。正常类提示也不能排除其他疾病。",
            f"分析版本：{_markdown(data['checkpoint_name'])} / {data['checkpoint_epoch']}（仅供结果追溯）",
            "", "## 复核备注", "", _markdown(data["note"]), "",
            "复核医生：________________　复核日期：________________", "", REVIEW_NOTE, ""]
    if record.get('reviews'):
        out = [line.replace('医生复核草稿 · 未签署', '已记录人工复核 · 未电子签署').replace(REVIEW_NOTE, '本报告含本机人工复核记录，不作为独立诊断或治疗依据。') for line in out]
        out += ['## 人工复核记录', '', '本机填写记录，未经身份认证，不构成电子签名。']
        for r in record['reviews']:
            out += ['', _markdown(f"#{r['revision']} · {r['reviewed_at']} · {r['reviewer']} · {r['decision']} · {r.get('category') or '未定'}"), _markdown(r['rationale'])]
    return "\n".join(out)
