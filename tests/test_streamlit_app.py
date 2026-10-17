"""One optional real-model UI workflow plus upload boundary checks.

The real-model workflow is skipped when private weights or demo images are not
present, so a teammate/CI can run the other tests without training or downloads.
No Streamlit server is started.
"""

import base64
from io import BytesIO
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app_support import ROOT, decode_image, image_sha, public_prediction
from reporting import build_report_html


def png_payload(size=(8, 6)):
    stream = BytesIO()
    Image.new("RGB", size, (65, 125, 185)).save(stream, format="PNG")
    return stream.getvalue()


class UploadBoundaryTests(unittest.TestCase):
    def test_valid_image_is_detached_rgb_and_can_be_read_repeatedly(self):
        payload = png_payload()
        image = decode_image(payload)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (8, 6))
        self.assertEqual(image.getpixel((0, 0)), (65, 125, 185))
        self.assertEqual(image_sha(payload), image_sha(payload))

    def test_empty_invalid_and_renamed_gif_are_rejected(self):
        gif = BytesIO()
        Image.new("RGB", (2, 2)).save(gif, format="GIF")
        for payload in (b"", b"not a picture", gif.getvalue(), png_payload()[:20]):
            with self.subTest(payload_size=len(payload)):
                with self.assertRaises(ValueError):
                    decode_image(payload)

    def test_encoded_size_limit_precedes_decode(self):
        with patch("app_support.MAX_UPLOAD_BYTES", 5):
            with self.assertRaisesRegex(ValueError, "10 MB"):
                decode_image(b"123456")

    def test_pixel_limit_is_checked_before_full_decode(self):
        with patch("app_support.MAX_PIXELS", 10):
            with self.assertRaisesRegex(ValueError, "像素"):
                decode_image(png_payload((4, 4)))

    def test_exif_orientation_matches_training_loader(self):
        stream = BytesIO()
        source = Image.new("RGB", (3, 5), "gray")
        exif = Image.Exif()
        exif[274] = 6
        source.save(stream, format="JPEG", exif=exif)
        self.assertEqual(decode_image(stream.getvalue()).size, (3, 5))

    def test_public_export_does_not_mutate_original_metadata(self):
        private = {"predicted_class": "CNV", "checkpoint": {"path": "private/path.pt", "filename": "best.pt"}}
        exported = public_prediction(private)
        self.assertNotIn("path", exported["checkpoint"])
        self.assertEqual(private["checkpoint"]["path"], "private/path.pt")


@unittest.skipUnless(
    (ROOT / "models" / "torch_resume" / "best.pt").is_file()
    and (ROOT / "assets" / "demo" / "CNV.png").is_file()
    and (ROOT / "assets" / "demo" / "DME.png").is_file(),
    "需要本地可信 best.pt 与内置示例；不在测试中下载或训练",
)
class RealModelUIWorkflowTests(unittest.TestCase):
    def test_predict_explain_export_and_stale_result_warning(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.button(key="run_analysis").disabled)
        app.session_state["source_choice"] = "体验示例"
        app.run()
        app.button(key="run_analysis").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        record = app.session_state["active_record"]
        case_id = record["case_id"]
        self.assertIsNotNone(record["heatmap"])
        self.assertEqual(record["source_sha256"], image_sha((ROOT / "assets" / "demo" / "CNV.png").read_bytes()))
        self.assertIn(record["prediction"]["predicted_class"], {"CNV", "DME", "DRUSEN", "NORMAL"})
        self.assertAlmostEqual(sum(record["prediction"]["probabilities"].values()), 1.0, places=5)
        self.assertEqual(record["prediction"]["checkpoint"]["filename"], "best.pt")

        app.slider(key="cam_alpha").set_value(0.2).run()
        self.assertEqual(len(app.exception), 0)
        record = app.session_state["active_record"]
        self.assertEqual(record["case_id"], case_id)
        images = re.findall(r'src="data:image/png;base64,([^"]+)"', build_report_html(record))
        self.assertEqual(len(images), 2)
        report_overlay = Image.open(BytesIO(base64.b64decode(images[1]))).convert("RGB")
        self.assertTrue(np.array_equal(np.asarray(report_overlay), np.asarray(record["overlay"])))
        # Ensure the cache is bounded independently of the original upload.
        self.assertLessEqual(max(record["original"].size), 1280)
        original_before = np.asarray(record["original"]).copy()
        scores_before = dict(record["prediction"]["probabilities"])
        app.slider(key="view_zoom").set_value(2.0).run()
        app.slider(key="view_brightness").set_value(1.5).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["active_record"]["case_id"], case_id)
        self.assertEqual(app.session_state["active_record"]["prediction"]["probabilities"], scores_before)
        self.assertTrue(np.array_equal(original_before, np.asarray(app.session_state["active_record"]["original"])))

        app.selectbox(key="example_class").select("DME").run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("上一次分析结果" in element.value for element in app.warning))
        self.assertEqual(app.session_state["active_record"]["case_id"], case_id)
        self.assertEqual(app.session_state["active_record"]["source_sha256"], record["source_sha256"])


if __name__ == "__main__":
    unittest.main()
