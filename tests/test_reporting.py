"""Export safety and truthful checkpoint labels without loading any model."""

import unittest

from PIL import Image

from reporting import build_report_html, build_report_markdown


def record():
    return {
        "case_id": "DEMO-001", "created_at": "2026-10-17T22:40:00+08:00",
        "source_name": "local-example.jpeg", "source_sha256": "a" * 64,
        "note": "待复核。", "prediction": {
            "predicted_class": "DME", "predicted_label_cn": "黄斑水肿", "model_score": 0.7,
            "probabilities": {"CNV": 0.1, "DME": 0.7, "DRUSEN": 0.05, "NORMAL": 0.15},
            "checkpoint": {"filename": "best.pt", "format": "best", "epoch": 18,
                           "val_acc": 0.7525, "best_epoch": 18, "best_val_acc": 0.7525},
        },
    }


class ReportTests(unittest.TestCase):
    def test_html_escapes_all_user_fields(self):
        sample = record()
        attack = '<script>alert("unsafe")</script><img src=x onerror=alert(1)>'
        for key in ("case_id", "created_at", "source_name", "source_sha256", "note"):
            sample[key] = attack
        sample["prediction"]["predicted_class"] = attack
        sample["prediction"]["predicted_label_cn"] = attack
        sample["prediction"]["checkpoint"]["filename"] = attack
        text = build_report_html(sample)
        self.assertNotIn("<script>", text)
        self.assertNotIn("<img src=x", text)
        self.assertIn("&lt;script&gt;alert(&quot;unsafe&quot;)&lt;/script&gt;", text)
        self.assertIn("default-src 'none'", text)

    def test_markdown_does_not_allow_injected_table_or_link(self):
        sample = record()
        sample["note"] = '[link](javascript:alert(1))\n| hidden | row |\n<script>bad()</script>'
        text = build_report_markdown(sample)
        self.assertIn(r"\[link\]\(javascript:alert\(1\)\)", text)
        self.assertIn(r"\| hidden \| row \|", text)
        self.assertNotIn("<script>", text)

    def test_latest_does_not_present_historical_best_as_own_accuracy(self):
        sample = record()
        sample["prediction"]["checkpoint"].update(filename="latest.pt", format="resume", epoch=50, val_acc=None)
        text = build_report_markdown(sample)
        self.assertIn("所用检查点轮次：第 50 轮", text)
        self.assertIn("该检查点验证准确率：未记录", text)
        self.assertIn("历史最佳验证记录：75.25%（第 18 轮）", text)
        self.assertNotIn("已完成 50", text)
        self.assertNotIn("训练已完成", text)

    def test_offline_images_and_original_are_preserved(self):
        sample = record()
        original = Image.new("RGB", (1800, 1300), (80, 120, 160))
        sample.update(original=original, overlay=Image.new("RGB", (32, 32), "red"))
        text = build_report_html(sample)
        self.assertEqual(text.count("data:image/png;base64,"), 2)
        self.assertNotIn('src="http', text)
        self.assertEqual(original.size, (1800, 1300))
        self.assertEqual(original.getpixel((0, 0)), (80, 120, 160))

    def test_scores_and_interpretation_are_not_medical_claims(self):
        sample = record()
        for builder in (build_report_html, build_report_markdown):
            text = builder(sample)
            self.assertIn("70.00%", text)
            self.assertIn("不代表患病概率", text)
            self.assertIn("不是病灶边界", text)
            self.assertIn("未经医生复核", text)
            self.assertNotIn("建议用药", text)
            self.assertNotIn("重度", text)

    def test_missing_or_nonfinite_scores_are_not_fabricated(self):
        sample = record()
        sample["prediction"]["model_score"] = float("nan")
        sample["prediction"]["checkpoint"] = {}
        text = build_report_markdown(sample)
        self.assertIn("该类别模型分数：未记录", text)
        self.assertIn("所用检查点轮次：未记录", text)
        self.assertNotIn("75.25%", text)


if __name__ == "__main__":
    unittest.main()
