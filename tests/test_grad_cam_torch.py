"""Grad-CAM checks use a tiny deterministic model; no checkpoint/training needed."""
import threading
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

from grad_cam_torch import GradCAMError, generate_gradcam


class TinyModel(nn.Module):
    def __init__(self, zero=False):
        super().__init__()
        self.layer4 = nn.Sequential(nn.Conv2d(3, 2, 1, bias=False), nn.ReLU())
        self.fc = nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.layer4[0].weight.fill_(0.2)
            self.fc.weight.copy_(torch.tensor([[2., 1.], [1., 1.], [0.2, 0.1], [0., 0.]]))
            if zero:
                self.fc.weight.zero_()
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        return self.fc(self.layer4(x).mean(dim=(2, 3)))


class TinyPredictor:
    def __init__(self, zero=False):
        self.model = TinyModel(zero)
        self.device = torch.device("cpu")
        self.lock = threading.RLock()
        self.checkpoint_info = {"epoch": 1, "test_model": True}

    def prepare_image(self, image):
        array = np.asarray(image.convert("L").convert("RGB").resize((16, 16)), dtype=np.float32) / 255
        return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)


class GradCAMTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.fromarray(np.tile(np.arange(32, dtype=np.uint8) * 7, (24, 1)))

    def test_frozen_model_produces_map_without_mutation_or_hook_leak(self):
        predictor = TinyPredictor()
        before = {key: value.clone() for key, value in predictor.model.state_dict().items()}
        for _ in range(2):
            result = generate_gradcam(predictor, self.image)
            self.assertEqual(result["target_class"], "CNV")
            self.assertEqual(result["overlay"].size, self.image.size)
            self.assertEqual(result["heatmap"].shape, (24, 32))
            self.assertEqual(result["heatmap"].dtype, np.float32)
            self.assertTrue(np.isfinite(result["heatmap"]).all())
            self.assertGreater(float(result["heatmap"].max()), 0)
            self.assertLessEqual(float(result["heatmap"].max()), 1)
            self.assertEqual(len(predictor.model.layer4[-1]._forward_hooks), 0)
        for key, value in predictor.model.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), key)
        self.assertTrue(all(parameter.grad is None for parameter in predictor.model.parameters()))

    def test_explicit_target_and_invalid_inputs(self):
        predictor = TinyPredictor()
        self.assertEqual(generate_gradcam(predictor, self.image, "DME")["target_index"], 1)
        for value in ["unknown", 4, -1, True]:
            with self.assertRaises(ValueError):
                generate_gradcam(predictor, self.image, value)
        for alpha in [float("nan"), -0.1, 1.1]:
            with self.assertRaises(ValueError):
                generate_gradcam(predictor, self.image, alpha=alpha)
        self.assertEqual(len(predictor.model.layer4[-1]._forward_hooks), 0)

    def test_no_positive_response_returns_error_not_fake_map(self):
        predictor = TinyPredictor(zero=True)
        with self.assertRaisesRegex(GradCAMError, "没有正响应"):
            generate_gradcam(predictor, self.image)
        self.assertEqual(len(predictor.model.layer4[-1]._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
