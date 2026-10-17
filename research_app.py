import json
import pandas as pd
import streamlit as st
from app_support import ROOT, training_summary
from config import CLASS_NAMES
st.set_page_config(page_title='眸析研发档案', layout='wide')
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


def audit_page():
    path = ROOT / "assets" / "data_audit_summary.json"
    if not path.is_file():
        st.info("尚未生成审计摘要，请运行 python tools/audit_oct.py。")
        return
    audit = json.loads(path.read_text(encoding="utf-8"))
    st.caption(f"完整数据快照 · {audit['generated_at']} · 原始数据和模型只读")
    pair = audit["pairs"]["train_val"]
    a, b, c = st.columns(3)
    a.metric("检查图像", f"{sum(sum(v.values()) for v in audit['counts'].values()):,}")
    b.metric("验证图与训练完全重复", str(pair['file_sha256']['val_images']))
    c.metric("相同内容跨标签组数", str(audit['duplicate_groups']['file_sha256']['conflicting_label_groups']))
    st.warning("当前验证划分存在数据交叉，应先处理重复内容、核验标签，并按患者分组设计新划分，再进行训练对照。")
    st.subheader("划分交叉证据")
    st.dataframe([{"数据对": key, "候选编号交集": value['candidate_id']['shared_values'],
                   "字节完全相同组": value['file_sha256']['shared_values'],
                   "灰度像素相同组": value['pixel_sha256']['shared_values']}
                  for key, value in audit['pairs'].items()], hide_index=True, width="stretch")
    st.caption("候选患者编号取自文件名中间数字，未通过患者映射表核验。没有完全重复不等于排除了近似图或相邻切片。")
    st.subheader("DRUSEN：全部测试图像的错误去向")
    st.dataframe([{"预测类别": key, "数量": value['count'],
                   "模型分数中位数": value['median_model_score'],
                   "分数≥90%的数量": value['score_at_least_90_percent']}
                  for key, value in audit['drusen']['score_statistics'].items()], hide_index=True, width="stretch")
    st.write("完整报告包含原图、预测类别的 Grad-CAM，以及真实标签 DRUSEN 的目标关注图。热图用于提出待复核问题，不能独立证明误判原因。")
    st.caption(audit['drusen']['selection_rule'])
    local_report = ROOT / "outputs" / "data_audit" / "report.html"
    if local_report.is_file():
        st.download_button("下载完整审计与案例报告", local_report.read_bytes(),
                           file_name="OCT数据与错误分析.html", mime="text/html", icon=":material/download:")
    else:
        st.info("仓库只提供汇总；原图案例报告保存在生成审计的本机。")
    st.markdown("**建议顺序：** 核验患者标识与冲突标签 → 去重并按患者分组隔离划分 → 在新划分重建基线 → 仅在验证集比较类别加权、解冻层数和增强强度。已经查看的测试案例用于探索分析，最终方案应使用未参与开发的留出数据评估。")



st.navigation([st.Page(evidence_page, title='实验档案'), st.Page(audit_page, title='数据与误判审查')]).run()
