"""User-facing desktop OCT workspace. Research lives in research_app.py."""
from __future__ import annotations
import base64
from datetime import datetime, date
from html import escape
from pathlib import Path
import uuid
import os

import numpy as np
from PIL import Image, ImageEnhance
import streamlit as st
import spatial_ui
from access_control import require_access

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app_support import ROOT, decode_image, image_sha, png_bytes, public_prediction
from config import CLASS_CN, CLASS_NAMES
from grad_cam_torch import GradCAMError, colorize_heatmap, generate_gradcam
from predict_torch import OCTPredictor
from reporting import build_report_html, build_report_markdown
from case_library import save_case, list_cases, load_case, review_history
from library_ui import library_page
from clinical_review import review_panel
from health_guide import TOPICS, consultation_page
from viewer_features import (annotation_editor, annotated_image, comparison_canvas,
                             natural_sort_key, sequence_id, validate_annotations)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="眸析 · OCT 影像工作站", page_icon=":material/visibility:",
                   layout="wide", initial_sidebar_state="collapsed")
st.html((ROOT / "assets" / "workstation.css").read_text(encoding="utf-8"))
spatial_ui.install_style()
require_access()
st.session_state.setdefault("records", [])
st.session_state.setdefault("active_record", None)


@st.cache_resource(max_entries=2, show_spinner=False)
def get_predictor(path: str, modified_ns: int) -> OCTPredictor:
    return OCTPredictor(path, device="cpu")


def model_path() -> Path:
    return Path(os.getenv("OCT_MODEL_PATH", str(ROOT / "models" / "torch_resume" / "best.pt")))


def image_uri(image):
    return "data:image/png;base64," + base64.b64encode(png_bytes(image)).decode("ascii")


def banner(title, subtitle):
    st.html(f'<div class="workspace-heading"><div><div class="eyebrow">MOUXI / RETINAL IMAGING</div>'
            f'<h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>'
            '<span class="local-badge"><i></i> 本机影像处理</span></div>')


def downloads(record, prefix):
    export_record = dict(record)
    export_record["reviews"] = review_history(record["case_id"])
    if record.get("annotations"):
        export_record["annotated"] = annotated_image(
            record["original"], record["annotations"], record.get("source_size")
        )
    st.download_button("导出完整报告", build_report_html(export_record),
                       file_name=record["case_id"] + ".html", mime="text/html",
                       icon=":material/download:", width="stretch", type="primary", key=prefix+"_html")
    with st.popover("更多导出", icon=":material/more_horiz:", width="stretch"):
        st.download_button("下载文字报告", build_report_markdown(export_record),
                           file_name=record["case_id"]+".md", mime="text/markdown", key=prefix+"_md")
        if record.get("overlay") is not None:
            st.download_button("保存关注叠加图", png_bytes(record["overlay"]),
                               file_name=record["case_id"]+"-attention.png", mime="image/png", key=prefix+"_cam")
    st.caption("报告含原图、关注图与备注，可在浏览器中打印为 PDF。")


def analyze(payload, source_name, source, series=None, slice_index=0, slice_total=1):
    checkpoint = model_path()
    image = decode_image(payload)
    source_size = list(image.size)
    predictor = get_predictor(str(checkpoint), checkpoint.stat().st_mtime_ns)
    prediction = public_prediction(predictor.predict(image))
    cam, cam_error = None, None
    try:
        cam = generate_gradcam(predictor, image)
    except GradCAMError as error:
        cam_error = str(error)
    original = image.convert("L").convert("RGB")
    original.thumbnail((1280, 1280))
    heatmap, overlay = None, None
    if cam:
        heatmap = np.asarray(Image.fromarray(cam["heatmap"]).resize(
            original.size, Image.Resampling.BILINEAR), dtype=np.float32).copy()
        overlay = Image.blend(original, colorize_heatmap(heatmap), .45)
    record = dict(case_id="OCT-"+datetime.now().strftime("%Y%m%d-%H%M%S-")+uuid.uuid4().hex[:4],
                  created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                  source_name=source_name, source_sha256=image_sha(payload),
                  prediction=prediction, original=original, overlay=overlay, heatmap=heatmap,
                  cam_error=cam_error, note="", input_kind=source,
                  source_size=source_size, annotations=[],
                  series_id=series, slice_index=int(slice_index)+1, slice_total=int(slice_total))
    st.session_state.records = (st.session_state.records + [record])[-5:]
    st.session_state.active_record = record
    try:
        save_case(record)
        st.session_state["archive_error"] = None
    except Exception:
        st.session_state["archive_error"] = "分析已完成，但本地案例保存失败。请先导出报告，并检查磁盘空间或文件权限。"
    # Viewing parameters are per analysis; they never affect inference.
    for key in ("cam_alpha", "view_mode", "view_zoom", "view_brightness"):
        st.session_state.pop(key, None)
    st.rerun()


def canvas(image, title, zoom=1.0, brightness=1.0):
    adjusted = ImageEnhance.Brightness(image).enhance(brightness)
    st.html(f'<section class="image-viewport"><div class="viewport-label"><span>{escape(title)}</span>'
            f'<span>{image.width} × {image.height}</span></div>'
            f'<div class="image-scroll" tabindex="0" aria-label="{escape(title)}，放大后可滚动查看">'
            f'<div class="image-stage" style="width:{zoom*100:.0f}%;height:{zoom*100:.0f}%">'
            f'<img alt="{escape(title)}" src="{image_uri(adjusted)}"></div></div>'
            '<div class="viewport-foot">OCT · 阅片图像<span>眸析 MOUXI</span></div></section>')


def viewer(record=None, preview=None):
    with st.container(key="viewer"):
        a, b = st.columns([2, 1], vertical_alignment="center")
        with a:
            mode = st.segmented_control("阅片布局", ["双图对照", "原始影像", "关注叠加"],
                                        default="双图对照", key="view_mode", label_visibility="collapsed") or "双图对照"
        with b:
            with st.popover("显示调整", icon=":material/tune:", width="stretch"):
                st.slider("显示放大", 1.0, 3.0, 1.0, .25, key="view_zoom")
                st.slider("显示亮度", .5, 1.8, 1.0, .1, key="view_brightness")
                st.caption("仅调整阅片显示；放大后可滚动浏览。报告保留标准亮度、完整视野。")
        zoom = st.session_state.get("view_zoom", 1.0)
        bright = st.session_state.get("view_brightness", 1.0)
        if record:
            alpha = st.slider("关注图强度", 0.0, .8, .45, .05, key="cam_alpha",
                              disabled=record["heatmap"] is None)
            if record["heatmap"] is not None:
                record["overlay"] = Image.blend(record["original"], colorize_heatmap(record["heatmap"]), alpha)
            if mode == "双图对照":
                l, r = st.columns(2, gap="xsmall")
                with l:
                    canvas(record["original"], "01 / 原始影像", zoom, bright)
                with r:
                    if record["overlay"] is not None:
                        canvas(record["overlay"], "02 / AI 关注区域", zoom, bright)
                    else:
                        st.warning("本次未能生成关注图，分类结果仍可查看。")
            else:
                picture = record["original"] if mode == "原始影像" else record["overlay"]
                if picture is not None:
                    canvas(picture, mode, zoom, bright)
                else:
                    st.warning("本次未能生成关注图。")
            st.html('<div class="viewer-legend"><span class="heat-scale"></span>低关注 → 高关注'
                    '<span>颜色表示模型关注程度，不是病灶边界。</span></div>')
            st.caption(f"当前影像：{record['source_name']} · {record['created_at'][:19].replace('T', ' ')}")
            with st.expander("影像标注与测量", expanded=bool(record.get("annotations"))):
                st.caption("可标记点位、测量线段、框选 ROI 或添加文字。测量单位为输入影像像素 px；没有设备像素间距时不换算为 μm。")
                current_marks = validate_annotations(record.get("annotations", []), record["original"].size)
                changed = annotation_editor(record, key="marks_"+record["case_id"])
                if changed is not None:
                    record["annotations"] = validate_annotations(changed, record["original"].size)
                    try:
                        save_case(record)
                        st.session_state["annotation_save_error"] = None
                        st.rerun()
                    except Exception:
                        st.session_state["annotation_save_error"] = "标注已在当前页面更新，但暂未保存到本机案例库。"
                if st.session_state.get("annotation_save_error"):
                    st.warning(st.session_state["annotation_save_error"])
                elif current_marks:
                    st.caption(f"{len(current_marks)} 项标注已自动保存在本机案例库，并可随报告导出。")
                else:
                    st.caption("标注会自动保存到本机案例库；标注用于阅片记录，不代表模型病灶边界。")
        elif preview is not None:
            canvas(preview.convert("L").convert("RGB"), "已导入 / 等待分析", zoom, bright)
            st.caption("影像已就绪，点击左侧「开始分析」查看筛查提示与关注区域。")
        else:
            st.html('<div class="empty-view"><div class="scan-glyph">◎</div>'
                    '<h2>从 OCT 影像序列开始</h2><p>导入切片，浏览序列、标注测量并辅助筛查</p>'
                    '<div class="empty-flow"><span>01 导入序列</span><b>→</b><span>02 浏览与标注</span>'
                    '<b>→</b><span>03 复核与导出</span></div></div>')
            st.caption("支持 JPEG / PNG 单张或多张 OCT 切片。没有影像时，可在左侧选择「体验示例」。")


NEO_CHART_BG = "#f3f6fa"
NEO_CHART_GRID = "#dce3eb"
CLASS_CHART_COLORS = {
    "CNV": "#218b91",
    "DME": "#e39a55",
    "DRUSEN": "#8b78c9",
    "NORMAL": "#5f987e",
}


def _style_neumorphic_chart(ax, fig):
    """Keep plots legible while matching the workstation's soft neutral surfaces."""
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=NEO_CHART_GRID, linewidth=0.7, zorder=0)
    ax.tick_params(axis="x", labelsize=9, colors="#d5e0e4")
    ax.tick_params(axis="y", labelsize=8, colors="#c3d1d8")
    ax.xaxis.label.set_color("#d5e0e4")
    ax.yaxis.label.set_color("#d5e0e4")
    for text in ax.texts:
        text.set_color("#e0eaed")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#d2dae4")
    ax.spines["bottom"].set_color("#d2dae4")


def probability_bar_chart(probs, predicted_class):
    """绘制四分类分数柱状图，预测类别高亮。"""
    fig, ax = plt.subplots(figsize=(4.8, 3.0), dpi=110)
    labels = [CLASS_CN[n] for n in CLASS_NAMES]
    values = [float(probs.get(n, 0.0)) for n in CLASS_NAMES]
    colors = [CLASS_CHART_COLORS[n] if n == predicted_class else "#ccd5df" for n in CLASS_NAMES]
    ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=8, color="#374151")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("模型分数", fontsize=9, color="#53657a")
    _style_neumorphic_chart(ax, fig)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    fig.tight_layout()
    return fig


def category_distribution_chart(counts):
    """绘制案例库分类分布柱状图（案例数量）。"""
    labels = [CLASS_CN[n] for n in CLASS_NAMES]
    values = [int(counts.get(n, 0)) for n in CLASS_NAMES]
    colors = [CLASS_CHART_COLORS[n] for n in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(4.8, 3.0), dpi=110)
    ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    for i, v in enumerate(values):
        ax.text(i, v + 0.05, str(v), ha="center", fontsize=9, color="#374151")
    ax.set_ylim(0, max(1, max(values)) + 1)
    ax.set_ylabel("案例数", fontsize=9, color="#53657a")
    _style_neumorphic_chart(ax, fig)
    if max(values) <= 6:
        ax.set_yticks(range(0, max(1, max(values)) + 2))
    else:
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def _short_name(name, limit=12):
    name = name or ""
    return name if len(name) <= limit else name[: limit - 1] + "…"


def multi_image_probability_chart(records):
    """多张影像的四分类分数对比分组柱状图（X 轴为影像，四组柱子为类别）。"""
    palette = CLASS_CHART_COLORS
    n = len(records)
    names = [_short_name(r.get("source_name") or r.get("case_id", "")) for r in records]
    x = np.arange(n)
    width = 0.8 / len(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(max(5.5, n * 1.8), 3.6), dpi=110)
    for j, cls in enumerate(CLASS_NAMES):
        values = [float(r["prediction"]["probabilities"].get(cls, 0.0)) for r in records]
        offset = (j - (len(CLASS_NAMES) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=CLASS_CN[cls], color=palette[cls], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, color="#4b5c70", rotation=0 if n <= 4 else 20)
    ax.set_ylabel("模型分数", fontsize=9, color="#53657a")
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    _style_neumorphic_chart(ax, fig)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper right")
    fig.tight_layout()
    return fig


def result_panel(record):
    with st.container(border=True, key="result"):
        st.markdown("#### 筛查结果")
        if record is None:
            st.html('<div class="pending-result"><span>等待分析</span><h3>筛查提示将在此显示</h3>'
                    '<p>完成分析后，可查看分类分数、填写备注并导出影像报告。</p></div>')
            return
        pred = record["prediction"]
        review_panel(record, "workspace")
        st.html(f'<div class="finding"><span>影像分类提示</span><h2>{escape(pred["predicted_label_cn"])}</h2>'
                f'<div class="finding-code">{escape(pred["predicted_class"])}'
                f'<strong>{pred["model_score"]:.2%}</strong></div><p>模型匹配分数 · 不代表患病概率</p></div>')
        st.pyplot(probability_bar_chart(pred["probabilities"], pred["predicted_class"]))
        st.markdown("#### 临床复核参考")
        st.info(TOPICS.get(pred["predicted_class"], "请结合完整检查由医生判断。")
                + "\n\n以上为一般性科普，不是个体治疗方案；具体请由眼科医生结合完整检查制定。")
        with st.expander("查看精确分数", expanded=False):
            score_rows = []
            for name in sorted(CLASS_NAMES, key=lambda c: pred["probabilities"][c], reverse=True):
                value = pred["probabilities"][name]
                score_rows.append(f'<div class="score-label"><span>{escape(CLASS_CN[name])}</span><b>{value:.2%}</b></div>'
                                  f'<div class="score-track"><i style="width:{value*100:.3f}%"></i></div>')
            st.html("".join(score_rows))
        if pred["predicted_class"] == "NORMAL":
            st.caption("本图更接近正常类别，仍不能排除其他异常。请结合完整检查判断。")
        else:
            st.caption("请结合完整 OCT 扫描与临床检查复核，单张影像分类不等于确诊。")
        record["note"] = st.text_area("阅片备注", value=record["note"], height=90, max_chars=1000,
                                     placeholder="记录需要进一步复核的影像表现…", key="note_"+record["case_id"])
        if st.button("保存备注与当前关注图", icon=":material/save:", width="stretch"):
            try:
                save_case(record)
                st.success("已更新到案例库")
            except Exception:
                st.error("保存失败，请检查本地磁盘或权限。仍可导出报告。")
        downloads(record, "screen")


def _set_slice(index):
    st.session_state["oct_slice_index"] = int(index)


def screening():
    banner("影像工作台", "导入 · 序列阅片 · 标注测量 · 辅助筛查")
    controls, images, results = st.columns([1.05, 3.4, 1.4], gap="medium")
    payload, preview, source_name = None, None, ""
    sequence, source = [], "上传影像"
    with controls, st.container(border=True, key="import"):
        st.markdown("#### 检查影像")
        source = st.segmented_control("影像来源", ["上传影像", "example 图库", "体验示例"], default="上传影像",
                                      key="source_choice", label_visibility="collapsed") or "上传影像"
        if source == "上传影像":
            uploads = st.file_uploader("选择 OCT 影像切片", type=["jpg", "jpeg", "png"],
                                       max_upload_size=10, accept_multiple_files=True,
                                       key="oct_upload_series")
            if uploads:
                candidates = []
                for uploaded in uploads:
                    item_payload = uploaded.getvalue()
                    item_name = Path(uploaded.name).name
                    try:
                        decode_image(item_payload)
                        candidates.append((item_name, item_payload))
                    except ValueError as error:
                        st.error(f"{item_name}：{error}")
                sequence = sorted(candidates, key=lambda item: natural_sort_key(item[0]))
            st.caption("可一次选择多张切片；文件名中的数字会按自然顺序排列。每张 JPEG / PNG 最大 10 MB。")
        elif source == "example 图库":
            folder = ROOT / "example"
            candidates = sorted(p for p in folder.rglob("*") if p.is_file()
                                and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                                and p.resolve().is_relative_to(folder.resolve())) if folder.is_dir() else []
            if candidates:
                selected = st.selectbox("选择本地图片", candidates,
                                        format_func=lambda p: str(p.relative_to(folder)), key="folder_image")
                if selected.stat().st_size <= 10*1024*1024:
                    sequence = [(selected.name, selected.read_bytes())]
                else:
                    st.error("这张图片超过 10 MB，请选择其他图片。")
            else:
                st.info("example 文件夹暂无 JPEG / PNG 图片。将图片放入后刷新，或使用上传入口。")
        else:
            category = st.selectbox("示例影像", CLASS_NAMES, format_func=lambda c: CLASS_CN[c], key="example_class")
            sample = ROOT / "assets" / "demo" / f"{category}.png"
            if sample.is_file():
                sequence = [(sample.name, sample.read_bytes())]
            st.caption("演示影像 · 非当前就诊者资料")

        series = sequence_id(sequence) if sequence else None
        if st.session_state.get("oct_sequence_id") != series:
            st.session_state["oct_sequence_id"] = series
            st.session_state["oct_slice_index"] = 0
        slice_index = 0
        if sequence:
            if len(sequence) > 1:
                options = list(range(len(sequence)))
                st.select_slider("切片位置", options=options,
                                 format_func=lambda i: f"{i+1:02d} / {len(sequence):02d}",
                                 key="oct_slice_index")
                slice_index = int(st.session_state["oct_slice_index"])
            source_name, payload = sequence[slice_index]
            try:
                preview = decode_image(payload)
            except ValueError as error:
                st.error(str(error))
                payload, source_name, preview = None, "", None
            if preview is not None:
                st.image(preview, width="stretch")
                st.caption(f"{source_name} · {preview.width} × {preview.height} px")
        if not model_path().is_file():
            st.warning("分析服务未就绪，请联系系统维护人员。")
        action_label = "开始分析当前切片" if len(sequence) > 1 else "开始分析"
        if st.button(action_label, key="run_analysis", type="primary", icon=":material/auto_awesome:",
                     width="stretch", disabled=payload is None or not model_path().is_file()):
            try:
                with st.spinner("正在分析当前影像，请稍候…"):
                    analyze(payload, source_name, source, series, slice_index, len(sequence))
            except (ValueError, RuntimeError, OSError) as error:
                st.error(f"分析未完成：{error}")
        st.divider()
        st.markdown("**当前工作区**")
        st.caption(f"序列 {len(sequence)} 张 · 当前 {slice_index+1 if sequence else 0} / {len(sequence)}")
        st.caption(f"本次会话已分析 {len(st.session_state.records)} 张")
        st.caption("分析结果自动保存到本机案例库，可跨会话回看。图片不发送第三方。")

    selected_hash = image_sha(payload) if payload else None
    matching = next((r for r in reversed(st.session_state.records)
                     if selected_hash and r.get("source_sha256") == selected_hash), None)
    if matching is None:
        active = st.session_state.get("active_record")
        matching = (active if active and (selected_hash is None
                                           or active.get("source_sha256") == selected_hash) else None)
    record = matching
    if record is not None:
        st.session_state.active_record = record
    with images:
        if st.session_state.get("archive_error"):
            st.warning(st.session_state["archive_error"])
        if len(sequence) > 1:
            st.caption(f"当前序列 · {sequence[slice_index][0]} · 第 {slice_index+1} / {len(sequence)} 张")
            thumb_count = min(9, len(sequence))
            first = max(0, min(slice_index - thumb_count // 2, len(sequence) - thumb_count))
            visible_indexes = range(first, first + thumb_count)
            with st.container(horizontal=True, gap="small"):
                for index in visible_indexes:
                    with st.container(border=True, key=f"slice_thumb_{series}_{index}"):
                        thumb = decode_image(sequence[index][1])
                        thumb.thumbnail((112, 70), Image.Resampling.LANCZOS)
                        st.image(thumb, width=112)
                        label = f"{index+1:02d}" + (" · 当前" if index == slice_index else "")
                        st.button(label, key=f"slice_select_{series}_{index}",
                                  type="primary" if index == slice_index else "secondary",
                                  width="stretch", on_click=_set_slice, args=(index,))
        if payload is not None and record is None:
            st.info("当前切片尚未分析。分析后可查看模型提示、添加阅片标注并保存到案例库。")
        viewer(record, preview)
    with results:
        result_panel(record)


def guide_page():
    banner("使用帮助", "导入序列 · 阅片标注 · 检查对比 · 报告保存")
    a, b, c = st.columns(3)
    for col, heading, content in [
        (a, "01  导入序列", "在影像工作台上传一张或多张 JPEG / PNG OCT 切片，按切片条浏览并逐张分析。"),
        (b, "02  标注与对比", "在原始影像上添加点位、距离、ROI 和文字；可在「影像对比」中同步查看两次检查。"),
        (c, "03  保存报告", "标注会保存在本机案例库，并可随完整报告导出。用浏览器打开 HTML 报告，可打印为 PDF。")]:
        with col, st.container(border=True):
            st.subheader(heading)
            st.write(content)
    with st.expander("如何理解分类与关注图？", expanded=True):
        st.write("支持脉络膜新生血管（CNV）、黄斑水肿（DME）、玻璃膜疣（DRUSEN）与正常（NORMAL）四类影像提示。")
        st.write("模型分数表示四类之间的相对匹配程度，未经概率校准，不是患病概率。正常类也不能排除其他疾病。")
        st.write("热图展示模型关注的位置，不能用来测量病灶范围或判断严重程度。结果需结合完整检查，由专业人员复核。")
    with st.expander("影像与记录保存在哪里？"):
        st.write("分析后自动保存到本机案例库，包含预览影像、关注叠加图、切片序列信息、标注和结果。可在案例详情保存备注、导出报告或删除记录。当前无账号隔离，仅供本机使用。咨询记录只保留在当前会话。")
    st.caption("眸析 · 辅助筛查演示系统，未经临床验证。")



def archive_page():
    library_page(banner,canvas,downloads,image_uri)

def health_page():
    consultation_page(banner)

def _comparison_records(limit=8):
    """优先使用本会话已分析记录，不足以案例库最近记录补充，用于多影像分数对比。"""
    out, seen = [], set()
    for r in st.session_state.get("records", []):
        out.append(r)
        seen.add(r.get("case_id"))
    for row in list_cases():
        if len(out) >= limit:
            break
        if row["id"] in seen:
            continue
        rec = load_case(row["id"])
        if rec is not None:
            out.append(rec)
            seen.add(row["id"])
    return out


def _image_comparison_records(limit=40):
    out, seen = [], set()
    candidates = []
    active = st.session_state.get("active_record")
    if active:
        candidates.append(active)
    candidates.extend(reversed(st.session_state.get("records", [])))
    for row in list_cases():
        if len(candidates) >= limit * 2:
            break
        if row["id"] not in seen:
            record = load_case(row["id"])
            if record is not None:
                candidates.append(record)
                seen.add(row["id"])
    for record in candidates:
        case_id = record.get("case_id")
        if not case_id or case_id in {x.get("case_id") for x in out}:
            continue
        out.append(record)
        if len(out) >= limit:
            break
    return out


def _comparison_label(record):
    when = str(record.get("created_at", "未记录"))[:10]
    slice_info = ""
    if record.get("slice_total", 1) > 1:
        slice_info = f" · 第 {record.get('slice_index', '?')} / {record.get('slice_total')} 张"
    return f"{record.get('source_name', '未命名影像')} · {when} · {record.get('case_id', '')}{slice_info}"


def image_comparison_page():
    banner("影像对比", "选择两次检查 · 同步阅片 · 人工观察影像差异")
    records = _image_comparison_records()
    if len(records) < 2:
        st.info("至少需要两张已保存的影像才能进行对比。请先在影像工作台分析另一张切片或从案例库保存更多案例。")
        return
    lookup = {record["case_id"]: record for record in records}
    ids = list(lookup)
    left, right = st.columns(2, gap="large")
    with left:
        case_a = st.selectbox("A · 基线影像", ids, index=0,
                              format_func=lambda key: _comparison_label(lookup[key]), key="compare_case_a")
    with right:
        default_b = 1 if len(ids) > 1 else 0
        case_b = st.selectbox("B · 对照影像", ids, index=default_b,
                              format_func=lambda key: _comparison_label(lookup[key]), key="compare_case_b")
    if case_a == case_b:
        st.warning("请选择两张不同的影像进行对比。")
        return
    record_a, record_b = lookup[case_a], lookup[case_b]
    controls, _ = st.columns([1, 2.2])
    with controls:
        view_mode = st.segmented_control("对比内容", ["原始影像", "AI 关注图", "用户标注"],
                                         default="原始影像", key="compare_view_mode") or "原始影像"
    if view_mode == "AI 关注图":
        image_a = record_a.get("overlay") or record_a["original"]
        image_b = record_b.get("overlay") or record_b["original"]
        if record_a.get("overlay") is None or record_b.get("overlay") is None:
            st.warning("至少一张影像没有可用的 AI 关注图，该侧将显示原始影像。")
    elif view_mode == "用户标注":
        image_a = annotated_image(record_a["original"], record_a.get("annotations", []), record_a.get("source_size"))
        image_b = annotated_image(record_b["original"], record_b.get("annotations", []), record_b.get("source_size"))
    else:
        image_a, image_b = record_a["original"], record_b["original"]
    same_series = bool(record_a.get("series_id") and record_a.get("series_id") == record_b.get("series_id"))
    if same_series:
        st.caption(f"同一导入序列 · 第 {record_a.get('slice_index', '?')} 张与第 {record_b.get('slice_index', '?')} 张。")
    else:
        st.caption("不同检查或序列。两张影像不会被假设为切片一一对应；当前视图用于人工对照阅片。")
    st.caption("鼠标滚轮缩放，拖动平移。未启用经过验证的自动病灶变化检测；请勿将视觉差异解释为病情进展结论。")
    label_a = "A · " + _comparison_label(record_a)
    label_b = "B · " + _comparison_label(record_b)
    comparison_canvas(image_a, image_b, label_a, label_b,
                      key=f"compare_{case_a}_{case_b}_{view_mode}", linked=True)
    with st.expander("对比影像信息"):
        detail_a, detail_b = st.columns(2)
        for col, label, record in ((detail_a, "A · 基线", record_a), (detail_b, "B · 对照", record_b)):
            with col:
                st.markdown(f"**{label}**")
                st.write(f"文件：{record.get('source_name', '未记录')}")
                st.write(f"案例编号：{record.get('case_id', '未记录')}")
                size = record.get("source_size") or record["original"].size
                st.write(f"输入图像尺寸：{size[0]} × {size[1]} px")
                st.write(f"模型分类提示：{record['prediction'].get('predicted_label_cn', '未记录')}（待专业人员复核）")


def statistics_page():
    banner("统计分析", "分类分数分布 · 多影像对比 · 案例库概况")
    active = st.session_state.get("active_record")
    rows = list_cases()
    if active is None and not rows:
        st.info("暂无统计数据。完成一次影像分析后，这里会展示分类分数分布、多影像对比与案例库统计。")
        return
    left, right = st.columns([1, 1], gap="large")
    with left, st.container(border=True):
        st.markdown("#### 最近一次分析 · 分类分数")
        if active is None:
            st.info("本会话还没有分析记录。可前往「影像工作台」导入影像并开始分析。")
        else:
            pred = active["prediction"]
            st.pyplot(probability_bar_chart(pred["probabilities"], pred["predicted_class"]))
            st.caption(f"案例 {escape(active['source_name'])} · 预测 {escape(pred['predicted_label_cn'])}"
                       f" · 匹配分数 {pred['model_score']:.2%}")
            with st.expander("精确分数", expanded=False):
                for name in sorted(CLASS_NAMES, key=lambda c: pred["probabilities"][c], reverse=True):
                    st.write(f"**{CLASS_CN[name]}**　{pred['probabilities'][name]:.2%}")
    with right, st.container(border=True):
        st.markdown("#### 案例库 · 分类分布")
        if not rows:
            st.info("案例库暂无已保存的案例。")
        else:
            all_dates = sorted({r["created"][:10] for r in rows if r.get("created")})
            filtered = rows
            if all_dates:
                lo_d = date.fromisoformat(all_dates[0])
                hi_d = date.fromisoformat(all_dates[-1])
                sel = st.date_input("时间范围", value=(lo_d, hi_d),
                                    min_value=lo_d, max_value=hi_d,
                                    key="stat_date_range")
                if isinstance(sel, (tuple, list)):
                    lo, hi = sel[0].isoformat(), sel[1].isoformat()
                else:
                    lo = hi = sel.isoformat()
                filtered = [r for r in rows if lo <= r["created"][:10] <= hi]
            counts = {c: 0 for c in CLASS_NAMES}
            for row in filtered:
                counts[row["category"]] = counts.get(row["category"], 0) + 1
            st.pyplot(category_distribution_chart(counts))
            suffix = "（已按时间范围筛选）" if len(filtered) != len(rows) else ""
            st.caption(f"共 {len(filtered)} 个案例{suffix}。分类来自模型预测标签，需专业人员复核。")
    with st.container(border=True):
        st.markdown("#### 多张影像 · 分数对比")
        compare = _comparison_records(limit=8)
        if len(compare) < 2:
            st.info("完成至少两次影像分析后，这里会展示多张影像的分类分数对比。")
        else:
            st.pyplot(multi_image_probability_chart(compare))
            st.caption("每条代表一张影像，四色柱子为该影像四个类别的相对匹配分数。分数未经概率校准。")
    st.caption("分数为四类之间的相对匹配程度，未经概率校准，不是患病概率。")


def spatial_home():
    spatial_ui.home(feature_pages)


home_page = st.Page(spatial_home, title="首页", icon=":material/home:", default=True)
feature_pages = {
    "workspace": st.Page(screening, title="影像工作台", icon=":material/view_quilt:", url_path="workspace"),
    "cases": st.Page(archive_page, title="案例库", icon=":material/folder_open:", url_path="cases"),
    "compare": st.Page(image_comparison_page, title="影像对比", icon=":material/compare:", url_path="compare"),
    "statistics": st.Page(statistics_page, title="统计分析", icon=":material/bar_chart:", url_path="statistics"),
    "consultation": st.Page(health_page, title="辅助诊断", icon=":material/forum:", url_path="consultation"),
}
help_page = st.Page(guide_page, title="使用帮助", icon=":material/help:", url_path="guide")
page = st.navigation([home_page, *feature_pages.values(), help_page], position="hidden")
spatial_ui.topbar(feature_pages, home_page, help_page, page)
if page.url_path:
    with st.container(key="feature_surface"):
        page.run()
else:
    page.run()
