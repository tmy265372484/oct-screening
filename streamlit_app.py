"""Local desktop OCT research workstation. Run with streamlit, never trains."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import uuid

import pandas as pd
from PIL import Image
import streamlit as st

from app_support import ROOT, decode_image, image_sha, png_bytes, public_prediction, training_summary
from config import CLASS_CN, CLASS_NAMES
from grad_cam_torch import GradCAMError, colorize_heatmap, generate_gradcam
from predict_torch import OCTPredictor, SCORE_NOTE
from reporting import build_report_html, build_report_markdown

st.set_page_config(page_title="眸析 · OCT 筛查工作台", page_icon=":material/visibility:",
                   layout="wide", initial_sidebar_state="expanded")
st.session_state.setdefault("records", [])
st.session_state.setdefault("active_record", None)


@st.cache_resource(max_entries=2, show_spinner=False)
def get_predictor(path: str, modified_ns: int) -> OCTPredictor:
    return OCTPredictor(path, device="cpu")


def model_path() -> Path:
    return ROOT / "models" / "torch_resume" / st.session_state.get("checkpoint_choice", "best.pt")


def checkpoint_status():
    summary = training_summary()
    with st.sidebar:
        st.divider()
        st.caption("推理模型")
        st.selectbox("检查点", ["best.pt", "latest.pt"], key="checkpoint_choice",
                     help="默认最佳验证检查点；切换不会训练或覆盖文件。")
        st.caption("CPU 本地推理 · ResNet50 · 160 × 160")
        if summary:
            st.caption(f"训练完成 {summary.get('completed_epoch', '—')} 轮 · 最佳第 {summary.get('best_epoch', '—')} 轮")
        st.caption("研究演示版本 / 结果待专业人员复核")


def metrics_strip():
    info = training_summary()
    cols = st.columns(4)
    cols[0].metric("分析范围", "4 类 OCT")
    cols[1].metric("训练完成", f"{info.get('completed_epoch', '—')} / {info.get('target_epochs', '—')} 轮")
    value = info.get("best_val_acc")
    cols[2].metric("最佳验证准确率", f"{value:.2%}" if value is not None else "未提供")
    cols[3].metric("本会话分析", f"{len(st.session_state.records)} 例")


def report_downloads(record, prefix):
    with st.container(horizontal=True):
        st.download_button("下载 HTML 报告", build_report_html(record),
                           file_name=f"{record['case_id']}.html", mime="text/html",
                           icon=":material/download:", key=f"{prefix}_html")
        st.download_button("下载文字报告", build_report_markdown(record),
                           file_name=f"{record['case_id']}.md", mime="text/markdown",
                           key=f"{prefix}_md")


def screening():
    st.caption("影像工作台 / OCT 单图辅助筛查")
    metrics_strip()
    st.divider()
    controls, images = st.columns([1, 3], gap="large")
    payload = None
    source_name = ""
    with controls:
        st.subheader("检查影像", icon=":material/upload_file:")
        source = st.radio("输入方式", ["内置示例", "上传影像"], horizontal=True, key="source_choice")
        if source == "内置示例":
            category = st.selectbox("示例类别", CLASS_NAMES, format_func=lambda x: f"{x} · {CLASS_CN[x]}", key="example_class")
            sample = ROOT / "assets" / "demo" / f"{category}.png"
            if sample.is_file():
                payload, source_name = sample.read_bytes(), sample.name
            else:
                st.info("示例文件缺失，请上传 JPEG / PNG。")
            st.caption("示例仅展示功能，标签不参与推理，不能代替测试集评估。")
        else:
            upload = st.file_uploader("上传 OCT 切片", type=["jpg", "jpeg", "png"],
                                      max_upload_size=10, key="oct_upload")
            if upload:
                payload, source_name = upload.getvalue(), Path(upload.name).name
        if payload:
            try:
                preview = decode_image(payload)
                st.image(preview, width="stretch")
                st.caption(f"{preview.width} × {preview.height} px · {len(payload) / 1024:.0f} KB")
            except ValueError as error:
                st.error(str(error))
                payload = None
        checkpoint = model_path()
        if not checkpoint.is_file():
            st.warning("缺少检查点。请将可信的 best.pt 放到 models/torch_resume/，详见使用说明。")
        run = st.button("开始分析", type="primary", icon=":material/analytics:", width="stretch",
                        disabled=payload is None or not checkpoint.is_file(), key="run_analysis")
        st.caption("图像 → 四类模型分数 → Grad-CAM → 复核报告")
        st.caption("上传图仅保留在本浏览器会话内存中；不会发往第三方。")
    if run:
        try:
            with st.spinner("正在执行本地推理与 Grad-CAM…"):
                image = decode_image(payload)
                predictor = get_predictor(str(checkpoint), checkpoint.stat().st_mtime_ns)
                prediction = public_prediction(predictor.predict(image))
                cam, cam_error = None, None
                try:
                    cam = generate_gradcam(predictor, image)
                except GradCAMError as error:
                    cam_error = str(error)
                display_original = image.convert("L").convert("RGB")
                display_original.thumbnail((1280, 1280))
                display_heatmap = None
                display_overlay = None
                if cam:
                    import numpy as np
                    display_heatmap = np.asarray(Image.fromarray(cam["heatmap"]).resize(
                        display_original.size, Image.Resampling.BILINEAR), dtype=np.float32).copy()
                    display_overlay = Image.blend(display_original, colorize_heatmap(display_heatmap), .45)
                record = {
                    "case_id": "OCT-" + datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4],
                    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "source_name": source_name, "source_sha256": image_sha(payload),
                    "prediction": prediction, "original": display_original,
                    "overlay": display_overlay,
                    "heatmap": display_heatmap, "cam_error": cam_error,
                    "note": "", "input_kind": source,
                }
                # Bound per-session memory even for large uploads.
                st.session_state.records = (st.session_state.records + [record])[-5:]
                st.session_state.active_record = record
                st.rerun()
        except (ValueError, RuntimeError, OSError) as error:
            st.error(f"分析未完成：{error}")
    with images:
        record = st.session_state.active_record
        if record is None:
            st.subheader("阅片与结果", icon=":material/monitor:")
            st.info("选择示例或上传影像，然后点击‘开始分析’。这里将并排展示原图、模型关注区域和四类结果。")
            st.caption("支持 JPEG / PNG 单张切片；此版本不进行视网膜层分割或病灶面积测量。")
            return
        prediction = record["prediction"]
        st.subheader("阅片与结果", icon=":material/monitor:")
        st.caption(f"{record['case_id']} · {record['created_at']} · {record['source_name']}")
        if payload is not None and image_sha(payload) != record["source_sha256"]:
            st.warning("当前显示上一次分析结果。新选择的图像需点击‘开始分析’。")
        if prediction["checkpoint"]["filename"] != checkpoint.name:
            st.warning("模型选择已改变。下方仍是原检查点的结果，需重新分析才能更新。")
        left, middle, right = st.columns([1, 1, 1.05], gap="medium")
        with left, st.container(border=True):
            st.markdown("**01 · OCT 原图**")
            st.image(record["original"], width="stretch")
            st.caption("预览最长边不超过 1280 px；推理时灰度化并缩放至 160 × 160。")
        with middle, st.container(border=True):
            st.markdown("**02 · 模型关注区域**")
            if record["heatmap"] is not None:
                alpha = st.slider("热图透明度", 0.0, 0.8, 0.45, 0.05, key="cam_alpha")
                overlay = Image.blend(record["original"], colorize_heatmap(record["heatmap"]), alpha)
                record["overlay"] = overlay
                st.image(overlay, width="stretch")
                st.caption(f"Grad-CAM / layer4 · 目标 {prediction['predicted_class']}")
                st.caption("蓝 → 红表示相对响应增强；不代表病灶边界。")
                st.download_button("保存叠加图", png_bytes(overlay), file_name=record["case_id"] + "-attention.png",
                                   mime="image/png", key="cam_download")
            else:
                st.warning(record["cam_error"] or "未生成热图。")
        with right, st.container(border=True):
            st.markdown("**03 · 辅助筛查结果**")
            st.subheader(prediction["predicted_class"])
            st.write(prediction["predicted_label_cn"])
            st.metric("最高类别模型分数", f"{prediction['model_score']:.2%}")
            scores = pd.DataFrame({"类别": CLASS_NAMES,
                                   "模型分数": [prediction["probabilities"][c] for c in CLASS_NAMES]})
            st.dataframe(scores, hide_index=True, width="stretch",
                         column_config={"模型分数": st.column_config.ProgressColumn("模型分数", min_value=0, max_value=1, format="percent")})
            st.caption(SCORE_NOTE)
            st.caption(f"检查点 {prediction['checkpoint']['filename']} · 第 {prediction['checkpoint']['epoch']} 轮")
        st.caption("分类结果需结合完整 OCT 扫描与临床检查复核；热图仅解释本次分类的相对关注区域。")
        with st.expander("复核备注与报告下载", expanded=True):
            note = st.text_area("复核备注（写入报告草稿）", value=record["note"], max_chars=1000,
                                key="note_" + record["case_id"], height=80)
            record["note"] = note
            report_downloads(record, "screen")


def records_page():
    st.caption("会话工作区 / 报告草稿")
    st.info("仅保存当前浏览器会话最近 5 例，刷新或关闭会话后可能清空。正式病例库与账号权限尚未实现。")
    records = st.session_state.records
    if not records:
        st.write("暂无分析记录。请先在影像工作台完成一次分析。")
        return
    rows = [{"检查编号": r["case_id"], "来源": r["source_name"], "时间": r["created_at"],
             "模型分类": r["prediction"]["predicted_class"], "模型分数": f"{r['prediction']['model_score']:.2%}",
             "状态": "待专业人员复核"} for r in reversed(records)]
    st.dataframe(rows, hide_index=True, width="stretch")
    choice = st.selectbox("选择报告", list(range(len(records))), index=len(records)-1,
                          format_func=lambda i: records[i]["case_id"], key="report_choice")
    record = records[choice]
    a, b = st.columns([1, 1])
    with a:
        st.image(record["overlay"] or record["original"], width="stretch")
    with b:
        st.markdown(build_report_markdown(record))
        report_downloads(record, "history")
    with st.expander("清理本会话"):
        if st.button("清空分析记录", icon=":material/delete:", key="clear_records"):
            st.session_state.records = []
            st.session_state.active_record = None
            st.rerun()


def evidence_page():
    st.caption("实验档案 / 本地训练记录快照")
    info = training_summary()
    if not info:
        st.info("未提供训练摘要，可运行 tools/export_training_summary.py 导出。")
        return
    a, b, c, d = st.columns(4)
    a.metric("已完成训练", f"{info['completed_epoch']} / {info['target_epochs']} 轮")
    b.metric("最佳验证准确率", f"{info['best_val_acc']:.2%}")
    c.metric("最佳检查点轮次", str(info["best_epoch"]))
    d.metric("最后一轮验证准确率", f"{info['final_val_acc']:.2%}")
    st.caption(f"摘要时间：{info.get('updated_at', '未记录')} · phase={info['phase']}。队友电脑上此页展示随仓库提供的实验快照。")
    history = pd.DataFrame(info.get("history", []))
    if not history.empty:
        st.subheader("训练与验证准确率")
        chart = history[["epoch", "train_acc", "val_acc"]].rename(columns={"epoch": "轮次", "train_acc": "训练准确率", "val_acc": "验证准确率"})
        st.line_chart(chart, x="轮次", y=["训练准确率", "验证准确率"], height=300)
        st.caption("仅绘制留有完成日志的轮次；未记录的早期轮次不补造。纵轴 0–1 表示准确率。")
    st.info(f"第 {info['augmentation_start_epoch']} 轮起使用轻度增强（{info['augmentation_profile']}）。训练中途改变策略，不能据此宣称它优于无增强基线，需独立对照实验。")
    with st.expander("最佳检查点验证集混淆矩阵与实验边界", expanded=True):
        matrix = info.get("best_confusion_matrix")
        if matrix:
            st.dataframe(pd.DataFrame(matrix, index=["真实 " + c for c in CLASS_NAMES],
                                      columns=["预测 " + c for c in CLASS_NAMES]), width="stretch")
            st.caption("第 18 轮检查点，验证集共 2000 张；行=真实标签，列=模型预测。")
        st.write("当前验证结果用于模型选择，不等于独立测试集结果。后续应核验患者层面的划分、开展测试集评估，并报告每类召回率、F1、混淆矩阵与错误案例。")
    test_path = ROOT / "assets" / "test_evaluation.json"
    if test_path.is_file():
        evaluation = json.loads(test_path.read_text(encoding="utf-8"))
        st.subheader("固定检查点 · 本地测试集评估")
        p, q, r = st.columns(3)
        p.metric("测试图像", str(evaluation["test_images"]))
        q.metric("测试准确率", f"{evaluation['accuracy']:.2%}")
        r.metric("宏平均 F1", f"{evaluation['macro_f1']:.4f}")
        st.dataframe(pd.DataFrame(evaluation["per_class"]).T.rename(
            columns={"precision": "精确率", "recall": "召回率", "f1": "F1", "support": "样本数", "predicted_count": "预测数量"}),
            width="stretch", column_config={c: st.column_config.NumberColumn(c, format="%.3f") for c in ["精确率", "召回率", "F1"]})
        recall = evaluation["per_class"]["DRUSEN"]["recall"]
        st.warning(f"DRUSEN 召回率 {recall:.1%}。请结合各类漏检与错误案例判断系统可靠性。患者独立性与图像去重尚未核验，本页不是临床验证。")
        st.dataframe(pd.DataFrame(evaluation["confusion_matrix_actual_rows_predicted_columns"],
                                  index=["真实 " + c for c in CLASS_NAMES], columns=["预测 " + c for c in CLASS_NAMES]), width="stretch")
        st.caption(f"固定 {evaluation['checkpoint']['filename']} 第 {evaluation['checkpoint']['epoch']} 轮；测试期间未训练或调参。")
    st.download_button("下载实验摘要 JSON", json.dumps(info, ensure_ascii=False, indent=2),
                       file_name="training_summary.json", mime="application/json")


def guide_page():
    st.caption("使用说明 / 研究演示边界")
    st.markdown("""### 演示流程
1. 在影像工作台选择示例或上传单张 OCT 切片。
2. 点击开始分析，查看原图、模型关注区域及四类模型分数。
3. 填写复核备注，下载 HTML 报告（可用浏览器打印为 PDF）或文字报告。
4. 在会话报告中回看最近 5 例，在实验档案中核对真实训练指标。

### 模型配置
默认读取项目下 `models/torch_resume/best.pt`，也可切换到 `latest.pt`。
权重不上传 GitHub，队友需要从队长处获取可信检查点并放入该目录。
推理全程在本机 CPU 执行，不访问外部 API，不启动训练。

### 如何理解结果
CNV：脉络膜新生血管；DME：黄斑水肿；DRUSEN：玻璃膜疣；NORMAL：正常类。
分数是未经校准的四类 softmax 输出；NORMAL 也不能排除模型类别以外的疾病。
Grad-CAM 是对所选分类的关注解释，不能用于病灶分割、面积测量或严重程度判定。
报告是离线模板草稿，须由专业人员复核；不会自动生成治疗建议。

### 接下来值得做的实验
已完成固定检查点的本地测试集评估，见实验档案。下一步：核验患者划分与重复图像 → 分类错误分析 →
独立的无增强 / 轻度增强对照 → 校准与低分样本转人工策略。
不要反复用测试集调参；没有结果时不填写提升幅度。
""")


with st.sidebar:
    st.header("眸析 MOUXI", icon=":material/visibility:")
    st.caption("OCT 视网膜影像辅助筛查")

page = st.navigation({
    "影像中心": [st.Page(screening, title="影像工作台", icon=":material/monitor:", default=True),
                 st.Page(records_page, title="会话报告", icon=":material/description:", url_path="reports")],
    "科研与说明": [st.Page(evidence_page, title="实验档案", icon=":material/analytics:", url_path="experiments"),
                   st.Page(guide_page, title="使用说明", icon=":material/help:", url_path="guide")],
}, position="sidebar")
checkpoint_status()
st.title(page.title, icon=page.icon)
page.run()
