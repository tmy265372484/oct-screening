# 眸析 · OCT 视网膜病变智能筛查系统

> 全国大学生数字媒体科技作品及创意竞赛参赛作品

## 项目简介
基于眼部 OCT（光学相干断层扫描）影像，自动识别视网膜常见病变
（CNV 脉络膜新生血管 / DME 黄斑水肿 / DRUSEN 玻璃膜疣 / NORMAL 正常），
并提供病灶热力图与可读报告的智能辅助筛查系统。

## 核心功能
1. 上传 OCT 影像，自动分类并给出置信度
2. Grad-CAM 可解释热力图，高亮病灶位置
3. 一键生成「医生专业版 / 患者通俗版」双轨报告
4. 多模态融合策略（早期 / 晚期融合）效果对比可视化

## 技术栈
- Python · TensorFlow/Keras · ResNet 微调
- Streamlit（网页界面）
- Grad-CAM（可解释性）
- Plotly（可视化）
- LLM API（报告生成）

## 数据集
Kaggle: Retinal OCT Images (Kermany2018) —— 本地存放，不入库

## 团队协作
- git pull：开工前先拉最新代码
- git add . && git commit -m "说明你改了什么"
- git push：干完推上去，队友再 pull
