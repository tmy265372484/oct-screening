"""Export anonymized, reproducible training evidence without loading weights.

Only completed-epoch lines are used for history. Missing epochs and losses stay
missing; the old diagnostic ``latest`` result is never used as the final model.
On a teammate's checkout without local logs, the committed summary is preserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from io import BytesIO
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parents[1]
LOG_TIME = r"(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
START = re.compile(LOG_TIME + r" Epoch (?P<epoch>\d+)/(?P<target>\d+) 开始")
COMPLETE = re.compile(
    LOG_TIME + r" Epoch (?P<epoch>\d+)/(?P<target>\d+) 完成 \| "
    r"训练准确率 (?P<train>[0-9.]+) \| 验证准确率 (?P<val>[0-9.]+) \| 完整检查点已保存"
)
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def _number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def parse_completed_history(log_text: str) -> list[dict]:
    starts, records = {}, {}
    for line in log_text.splitlines():
        match = START.search(line)
        if match:
            # A deliberately restarted partial epoch is not a completion.
            starts[int(match["epoch"])] = datetime.strptime(match["time"], "%Y-%m-%d %H:%M:%S,%f")
        match = COMPLETE.search(line)
        if not match:
            continue
        epoch = int(match["epoch"])
        finished = datetime.strptime(match["time"], "%Y-%m-%d %H:%M:%S,%f")
        train_acc, val_acc = float(match["train"]), float(match["val"])
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (train_acc, val_acc)):
            raise ValueError(f"第 {epoch} 轮日志准确率不是有效的 0–1 数值。")
        duration = (finished - starts[epoch]).total_seconds() if epoch in starts else None
        records[epoch] = {
            "epoch": epoch, "train_loss": None, "train_acc": train_acc,
            "val_loss": None, "val_acc": val_acc,
            "seconds": round(duration, 3) if duration is not None and duration >= 0 else None,
            "completed_at": finished.isoformat(timespec="seconds"),
        }
    return [records[epoch] for epoch in sorted(records)]


def build_summary(project_dir: Path) -> dict:
    run_dir = project_dir / "models" / "torch_resume"
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    history = parse_completed_history((run_dir / "training.log").read_text(encoding="utf-8"))
    completed = progress.get("completed_epoch")
    if not isinstance(completed, int) or isinstance(completed, bool) or completed < 0:
        raise ValueError("progress.json 没有有效的 completed_epoch。")
    history = [record for record in history if record["epoch"] <= completed]
    # progress stores the latest completed epoch's real losses. Do not fill
    # historical gaps with diagnostic evaluations or synthetic interpolation.
    final_record = next((record for record in history if record["epoch"] == completed), None)
    if final_record is not None and progress.get("phase") in {"completed", "epoch_complete"}:
        for field in ("train_loss", "train_acc", "val_loss", "val_acc"):
            value = _number(progress.get(field))
            if value is not None:
                final_record[field] = value
    available = [record["epoch"] for record in history]
    missing = [epoch for epoch in range(1, completed + 1) if epoch not in available]
    summary = {
        "schema_version": 1,
        "completed_epoch": completed,
        "target_epochs": progress.get("target_epochs"),
        "phase": progress.get("phase", "unknown"),
        "completion_verified": (progress.get("phase") == "completed"
                                and completed >= progress.get("target_epochs", 50)
                                and (run_dir / "latest.pt").is_file()),
        "best_epoch": progress.get("best_epoch"),
        "best_val_acc": _number(progress.get("best_acc")),
        "final_val_acc": final_record["val_acc"] if final_record else None,
        "augmentation_start_epoch": progress.get("augmentation_start_epoch"),
        "augmentation_profile": progress.get("augmentation_profile"),
        "updated_at": progress.get("updated_at"),
        "classes": CLASS_NAMES,
        "history": history,
        "history_available_epochs": available,
        "missing_history_epochs": missing,
        "sources": ["models/torch_resume/progress.json", "models/torch_resume/training.log"],
        "notes": [
            "本文件是本次本地训练的脱敏真实记录，不包含患者文件名、PID 或绝对路径。",
            "history 仅包含 training.log 的完整轮次完成记录；缺失轮次不补造。",
            "完成行准确率保留日志精度；第50轮的损失与准确率使用 progress.json 完整精度。",
            "历史损失未逐轮记录，缺失值为 null；不能据此绘制完整损失曲线。",
            "第24轮开始启用轻量数据增强，此次训练不能单独作为公平的增强消融实验。",
            "验证集结果用于模型选择，不是独立测试集结果。",
        ],
    }
    diagnostic_path = project_dir / "diagnostics" / "validation_20261017_134225.json"
    if diagnostic_path.is_file():
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        best = next((entry for entry in diagnostic.get("results", [])
                     if entry.get("name") == "best" and entry.get("epoch") == summary["best_epoch"]), None)
        matrix = best.get("confusion_matrix_actual_rows_predicted_columns") if best else None
        if (diagnostic.get("classes") == CLASS_NAMES and isinstance(matrix, list)
                and len(matrix) == len(CLASS_NAMES)
                and all(isinstance(row, list) and len(row) == len(CLASS_NAMES) for row in matrix)
                and all(isinstance(value, int) and value >= 0 for row in matrix for value in row)):
            summary.update({
                "best_confusion_matrix": matrix,
                "best_confusion_matrix_epoch": best["epoch"],
                "best_confusion_matrix_source": "validation",
                "best_confusion_matrix_axes": "actual_rows_predicted_columns",
                "validation_images": sum(sum(row) for row in matrix),
            })
            summary["sources"].append("diagnostics/validation_20261017_134225.json")
            summary["notes"].append("混淆矩阵来自第18轮最佳模型在2000张验证图上的CPU复核；旧诊断的 latest 第21轮结果未当作最终第50轮结果。")
    return summary


def _font(size: int, bold: bool = False):
    paths = [Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
             Path("C:/Windows/Fonts/simhei.ttf"),
             Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
             Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for path in paths:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_training_chart(summary: dict, output_path: Path):
    """Standard Matplotlib plot with a Pillow evidence-card layout."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import PercentFormatter

    canvas = Image.new("RGB", (1800, 1120), "#eff4f8")
    draw = ImageDraw.Draw(canvas)
    heading, subhead, label, small = _font(40, True), _font(29, True), _font(24), _font(21)
    draw.text((52, 35), "OCT 模型训练完成 · 真实训练曲线", font=heading, fill="#18344c")
    draw.text((54, 99), "程序生成的训练记录图；历史曲线只使用日志中有完成记录的轮次。", font=label, fill="#52697c")
    def percent(value):
        return f"{value:.2%}" if value is not None else "未记录"
    cards = [("完成轮次", f"{summary['completed_epoch']} / {summary.get('target_epochs', 50)}"),
             ("最佳验证准确率", percent(summary["best_val_acc"])),
             ("最佳模型轮次", f"第 {summary['best_epoch']} 轮"),
             ("最终轮验证准确率", percent(summary["final_val_acc"]))]
    for index, (name, value) in enumerate(cards):
        x = 52 + 435 * index
        draw.rounded_rectangle((x, 158, x + 411, 281), radius=14, fill="white")
        draw.text((x + 24, 176), name, font=label, fill="#586f81")
        draw.text((x + 24, 218), value, font=subhead, fill="#11647a")
    draw.rounded_rectangle((52, 312, 1748, 892), radius=16, fill="white")
    available = summary["history_available_epochs"]
    span = f"第 {min(available)}–{max(available)} 轮" if available else "无可用记录"
    draw.text((83, 334), f"训练 / 验证准确率  |  有数据轮次：{span}（{len(available)}轮）", font=subhead, fill="#1c3b52")
    low, high = 0.5, 0.9
    observed = [record[field] for record in summary["history"] for field in ("train_acc", "val_acc")]
    if observed:
        low = max(0, math.floor(min(observed) * 10) / 10)
        high = min(1, math.ceil(max(observed) * 10) / 10)
        if low == high:
            low, high = max(0, low - 0.1), min(1, high + 0.1)
    first_epoch = min(available) if available else 1
    last_epoch = max(available) if available else max(2, summary["completed_epoch"])
    figure = Figure(figsize=(16.1, 5.03), dpi=100, facecolor="white")
    plot_canvas = FigureCanvasAgg(figure)
    axis = figure.add_axes([0.049, 0.12, 0.922, 0.79])
    font_path = getattr(_font(20), "path", None)
    plot_font = FontProperties(fname=font_path, size=14) if font_path else FontProperties(size=14)
    axis.set_ylim(low, high)
    axis.set_xlim(first_epoch, last_epoch if last_epoch > first_epoch else first_epoch + 1)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_yticks([low + (high - low) * i / 5 for i in range(6)])
    ticks = sorted(set([first_epoch, last_epoch] + list(range(math.ceil(first_epoch / 5) * 5, last_epoch, 5))))
    axis.set_xticks(ticks)
    axis.set_xlabel("Epoch", loc="right", fontproperties=plot_font, color="#6a8090")
    axis.grid(axis="y", color="#dde6ed", linewidth=0.8)
    axis.grid(axis="x", color="#edf1f5", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(length=0, pad=8, labelsize=14, colors="#6a8090")
    for tick in axis.get_xticklabels() + axis.get_yticklabels():
        tick.set_fontproperties(plot_font)
    for spine in axis.spines.values():
        spine.set_visible(False)
    augmentation = summary.get("augmentation_start_epoch")
    if augmentation is not None and first_epoch <= augmentation <= last_epoch:
        axis.axvline(augmentation, color="#d69c3c", linestyle="--", linewidth=1.4)
        axis.text(augmentation + 0.3, 1.025, f"第{augmentation}轮：启用轻量增强",
                  transform=axis.get_xaxis_transform(), color="#9b6b1e", fontproperties=plot_font)
    history_by_epoch = {record["epoch"]: record for record in summary["history"]}
    epochs = list(range(first_epoch, last_epoch + 1))
    for field, color in (("train_acc", "#438ac2"), ("val_acc", "#0a9989")):
        scores = [history_by_epoch[epoch][field] if epoch in history_by_epoch else float("nan")
                  for epoch in epochs]
        axis.plot(epochs, scores, color=color, linewidth=2.4, marker="o", markersize=4)
    best = next((record for record in summary["history"] if record["epoch"] == summary["best_epoch"]), None)
    if best:
        axis.scatter([best["epoch"]], [best["val_acc"]], color="#f1b740", edgecolors="#865e1d", s=130, zorder=5)
        axis.annotate(f"最佳 {percent(summary['best_val_acc'])}", (best["epoch"], best["val_acc"]),
                      xytext=(10, -25), textcoords="offset points", fontproperties=plot_font, color="#7e581a")
    buffer = BytesIO()
    plot_canvas.print_png(buffer)
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        canvas.paste(rendered.convert("RGB"), (85, 383))
    figure.clear()
    for x, color, text in ((1120, "#438ac2", "训练准确率"), (1370, "#0a9989", "验证准确率")):
        draw.line((x, 351, x + 42, 351), fill=color, width=4)
        draw.text((x + 53, 336), text, font=small, fill="#52697c")
    missing = summary["missing_history_epochs"]
    missing_note = f"第{min(missing)}–{max(missing)}轮缺少日志" if missing else "无缺失轮次"
    draw.text((60, 925), f"数据说明：{missing_note}，未补造；有{len(available)}条完整轮次记录，历史曲线仅绘制这些轮次。", font=label, fill="#4e667a")
    draw.text((60, 969), f"损失只保留最终轮真实值，未伪造历史损失。增强开始于第{augmentation}轮，不能单独证明增强有效。", font=label, fill="#4e667a")
    draw.text((60, 1013), f"最终轮损失：训练 {summary['history'][-1]['train_loss']:.4f} / 验证 {summary['history'][-1]['val_loss']:.4f}" if summary["history"] and summary["history"][-1]["train_loss"] is not None and summary["history"][-1]["val_loss"] is not None else "最终轮损失：未记录", font=label, fill="#4e667a")
    draw.text((60, 1062), f"完成状态：{summary['phase']}    更新时间：{summary.get('updated_at', '未记录')}    仅代表验证结果", font=small, fill="#657c8d")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="从真实本地日志导出脱敏训练摘要和曲线。")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--output", type=Path, default=Path("assets/training_summary.json"))
    parser.add_argument("--chart", type=Path, default=Path("screenshots/07_训练完成与曲线.png"))
    args = parser.parse_args(argv)
    root = args.project_dir.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    chart = args.chart if args.chart.is_absolute() else root / args.chart
    required = [root / "models/torch_resume/progress.json", root / "models/torch_resume/training.log"]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        print(f"本地训练记录缺失（{', '.join(missing)}）；保留已有 training_summary.json，不覆盖。")
        return 0
    try:
        summary = build_summary(root)
        if output.is_file():
            old = json.loads(output.read_text(encoding="utf-8"))
            if old.get("completed_epoch", 0) > summary["completed_epoch"]:
                print("本地记录比已有摘要更早；保留已有摘要，不覆盖。")
                return 0
        # Render before replacing a valid existing summary.
        draw_training_chart(summary, chart)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"摘要已导出：{output.name}；有完成记录 {len(summary['history'])} 轮；缺失轮次 {summary['missing_history_epochs']}。")
        print(f"曲线已生成：{chart.name}（程序生成图，非历史界面截图）。")
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"导出失败：{exc}；未覆盖已有摘要。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
