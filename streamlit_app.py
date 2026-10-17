"""User-facing desktop OCT workspace. Research lives in research_app.py."""
from __future__ import annotations
import base64
from datetime import datetime
from html import escape
from pathlib import Path
import uuid

import numpy as np
from PIL import Image, ImageEnhance
import streamlit as st

from app_support import ROOT, decode_image, image_sha, png_bytes, public_prediction
from config import CLASS_CN, CLASS_NAMES
from grad_cam_torch import GradCAMError, colorize_heatmap, generate_gradcam
from predict_torch import OCTPredictor
from reporting import build_report_html, build_report_markdown

st.set_page_config(page_title="眸析 · OCT 影像工作站", page_icon=":material/visibility:",
                   layout="wide", initial_sidebar_state="expanded")
st.session_state.setdefault("records", [])
st.session_state.setdefault("active_record", None)
st.html((ROOT / "assets" / "workstation.css").read_text(encoding="utf-8"))


@st.cache_resource(max_entries=2, show_spinner=False)
def get_predictor(path: str, modified_ns: int) -> OCTPredictor:
    return OCTPredictor(path, device="cpu")


def model_path() -> Path:
    return ROOT / "models" / "torch_resume" / "best.pt"


def image_uri(image):
    return "data:image/png;base64," + base64.b64encode(png_bytes(image)).decode("ascii")


def banner(title, subtitle):
    st.html(f'<div class="workspace-heading"><div><div class="eyebrow">MOUXI / RETINAL IMAGING</div>'
            f'<h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>'
            '<span class="local-badge"><i></i> 本机影像处理</span></div>')


def downloads(record, prefix):
    st.download_button("导出完整报告", build_report_html(record),
                       file_name=record["case_id"] + ".html", mime="text/html",
                       icon=":material/download:", width="stretch", type="primary", key=prefix+"_html")
    with st.popover("更多导出", icon=":material/more_horiz:", width="stretch"):
        st.download_button("下载文字报告", build_report_markdown(record),
                           file_name=record["case_id"]+".md", mime="text/markdown", key=prefix+"_md")
        if record.get("overlay") is not None:
            st.download_button("保存关注叠加图", png_bytes(record["overlay"]),
                               file_name=record["case_id"]+"-attention.png", mime="image/png", key=prefix+"_cam")
    st.caption("报告含原图、关注图与备注，可在浏览器中打印为 PDF。")


def analyze(payload, source_name, source):
    checkpoint = model_path()
    image = decode_image(payload)
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
                  cam_error=cam_error, note="", input_kind=source)
    st.session_state.records = (st.session_state.records + [record])[-5:]
    st.session_state.active_record = record
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
            '<div class="viewport-foot">OCT · 单张切片<span>眸析 MOUXI</span></div></section>')


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
        elif preview is not None:
            canvas(preview.convert("L").convert("RGB"), "已导入 / 等待分析", zoom, bright)
            st.caption("影像已就绪，点击左侧「开始分析」查看筛查提示与关注区域。")
        else:
            st.html('<div class="empty-view"><div class="scan-glyph">◎</div>'
                    '<h2>从一张 OCT 影像开始</h2><p>导入影像，查看筛查提示与模型关注区域</p>'
                    '<div class="empty-flow"><span>01 导入影像</span><b>→</b><span>02 对照阅片</span>'
                    '<b>→</b><span>03 导出报告</span></div></div>')
            st.caption("支持 JPEG / PNG 单张 OCT 切片。没有影像时，可在左侧选择「体验示例」。")


def result_panel(record):
    with st.container(border=True, key="result"):
        st.markdown("#### 筛查结果")
        if record is None:
            st.html('<div class="pending-result"><span>等待分析</span><h3>筛查提示将在此显示</h3>'
                    '<p>完成分析后，可查看分类分数、填写备注并导出影像报告。</p></div>')
            return
        pred = record["prediction"]
        st.badge("待专业人员复核", color="orange", icon=":material/pending:")
        st.html(f'<div class="finding"><span>影像分类提示</span><h2>{escape(pred["predicted_label_cn"])}</h2>'
                f'<div class="finding-code">{escape(pred["predicted_class"])}'
                f'<strong>{pred["model_score"]:.2%}</strong></div><p>模型匹配分数 · 不代表患病概率</p></div>')
        with st.expander("查看全部类别分数", expanded=True):
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
        downloads(record, "screen")


def screening():
    banner("影像工作台", "导入 · 对照阅片 · 辅助筛查 · 报告")
    controls, images, results = st.columns([1.05, 3.4, 1.4], gap="medium")
    payload, preview, source_name = None, None, ""
    with controls, st.container(border=True, key="import"):
        st.markdown("#### 检查影像")
        source = st.segmented_control("影像来源", ["上传影像", "体验示例"], default="上传影像",
                                      key="source_choice", label_visibility="collapsed") or "上传影像"
        if source == "上传影像":
            upload = st.file_uploader("选择 OCT 影像", type=["jpg", "jpeg", "png"],
                                      max_upload_size=10, key="oct_upload")
            if upload:
                payload, source_name = upload.getvalue(), Path(upload.name).name
        else:
            category = st.selectbox("示例影像", CLASS_NAMES, format_func=lambda c: CLASS_CN[c], key="example_class")
            sample = ROOT / "assets" / "demo" / f"{category}.png"
            if sample.is_file():
                payload, source_name = sample.read_bytes(), sample.name
            st.caption("演示影像 · 非当前就诊者资料")
        if payload:
            try:
                preview = decode_image(payload)
                st.image(preview, width="stretch")
                st.caption(f"{source_name} · {preview.width} × {preview.height} px")
            except ValueError as error:
                st.error(str(error))
                payload = None
        if not model_path().is_file():
            st.warning("分析服务未就绪，请联系系统维护人员。")
        if st.button("开始分析", key="run_analysis", type="primary", icon=":material/auto_awesome:",
                     width="stretch", disabled=payload is None or not model_path().is_file()):
            try:
                with st.spinner("正在分析影像，请稍候…"):
                    analyze(payload, source_name, source)
            except (ValueError, RuntimeError, OSError) as error:
                st.error(f"分析未完成：{error}")
        st.caption("单张 JPEG / PNG · 最大 10 MB")
        st.divider()
        st.markdown("**当前工作区**")
        st.caption(f"本次会话已分析 {len(st.session_state.records)} 张")
        st.caption("影像在本机处理，不发送第三方。最近 5 次结果暂存于本会话，请及时导出报告。")
    record = st.session_state.active_record
    with images:
        if record and payload is not None and image_sha(payload) != record["source_sha256"]:
            st.warning("当前显示上一次分析结果。新影像已选好，请点击「开始分析」。")
        viewer(record, preview)
    with results:
        result_panel(record)


def records_page():
    banner("检查记录", "回看本次会话影像，整理备注并导出报告")
    records = st.session_state.records
    st.caption("最近 5 次分析暂存在当前会话；请在关闭页面前导出所需报告。")
    if not records:
        st.info("暂无检查记录。请在影像工作台导入影像并完成分析。")
        return
    st.dataframe([{"检查编号":r["case_id"], "影像":r["source_name"],
                   "分析时间":r["created_at"][:19].replace("T"," "), "分类提示":r["prediction"]["predicted_label_cn"],
                   "状态":"待复核"} for r in reversed(records)], hide_index=True, width="stretch")
    index = st.selectbox("选择检查", list(range(len(records))), index=len(records)-1,
                         format_func=lambda i: records[i]["source_name"]+" · "+records[i]["case_id"], key="report_choice")
    record = records[index]
    l, r = st.columns([3, 1.25], gap="large")
    with l:
        a,b = st.columns(2)
        with a:
            canvas(record["original"], "原始影像")
        with b:
            if record["overlay"] is not None:
                canvas(record["overlay"], "AI 关注区域")
    with r, st.container(border=True):
        st.badge("报告草稿 · 未签署", color="orange")
        st.subheader(record["prediction"]["predicted_label_cn"])
        st.write(f"模型匹配分数：{record['prediction']['model_score']:.1%}")
        record["note"] = st.text_area("阅片备注", record["note"], max_chars=1000,
                                     key="history_note_"+record["case_id"])
        downloads(record, "history")
    with st.expander("管理会话记录"):
        st.caption("清空后无法从本会话恢复，已经下载的报告不受影响。")
        if st.button("清空本次会话", icon=":material/delete:"):
            st.session_state.records = []
            st.session_state.active_record = None
            st.rerun()


def guide_page():
    banner("使用帮助", "了解影像导入、结果解读与报告保存")
    a, b, c = st.columns(3)
    for col, heading, content in [
        (a, "01  导入影像", "在影像工作台上传 JPEG / PNG 格式的 OCT 单张切片，点击开始分析。也可以先体验内置示例。"),
        (b, "02  对照阅片", "查看原图与 AI 关注图；支持单图与双图切换、放大、显示亮度及关注图强度调整。"),
        (c, "03  保存报告", "填写阅片备注，导出完整报告。用浏览器打开下载文件，可打印为 PDF。")]:
        with col, st.container(border=True):
            st.subheader(heading)
            st.write(content)
    with st.expander("如何理解分类与关注图？", expanded=True):
        st.write("支持脉络膜新生血管（CNV）、黄斑水肿（DME）、玻璃膜疣（DRUSEN）与正常（NORMAL）四类影像提示。")
        st.write("模型分数表示四类之间的相对匹配程度，未经概率校准，不是患病概率。正常类也不能排除其他疾病。")
        st.write("热图展示模型关注的位置，不能用来测量病灶范围或判断严重程度。结果需结合完整检查，由专业人员复核。")
    with st.expander("影像与记录保存在哪里？"):
        st.write("当前版本在本机处理影像，最近 5 次分析保存在当前浏览器会话中；不是永久病例库。关闭会话前请导出报告。")
    st.caption("眸析 · 辅助筛查演示系统，未经临床验证。")


st.logo(str(ROOT / "assets" / "mouxi-logo.svg"), size="large")
page = st.navigation({"工作空间":[
    st.Page(screening, title="影像工作台", icon=":material/view_quilt:", default=True),
    st.Page(records_page, title="检查记录", icon=":material/folder_open:", url_path="reports")],
    "支持":[st.Page(guide_page, title="使用帮助", icon=":material/help:", url_path="guide")]})
with st.sidebar:
    st.html('<div class="sidebar-note"><b>让影像更易读</b><p>辅助筛查 · 专业复核</p>'
            '<small>本机运行 / 演示版本</small></div>')
page.run()
