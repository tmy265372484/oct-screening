"""
推理脚本：加载训练好的模型，对单张OCT图片做预测
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import sys
from config import MODEL_PATH, IMG_SIZE, CLASS_NAMES, CLASS_CN


def load_model():
    if not MODEL_PATH.exists():
        print(f"模型不存在: {MODEL_PATH}")
        print("请先运行 train.py 训练模型")
        sys.exit(1)
    return keras.models.load_model(MODEL_PATH)


def predict_image(model, img_path):
    """对单张图片做预测，返回类别和置信度"""
    img = Image.open(img_path).convert("L").resize(IMG_SIZE)
    img = np.stack([img, img, img], axis=-1)  # 灰度转3通道
    img = np.expand_dims(img, axis=0) / 255.0

    preds = model.predict(img, verbose=0)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], preds[idx], dict(zip(CLASS_NAMES, preds))


if __name__ == "__main__":
    model = load_model()
    # 测试一张示例图
    from config import EXAMPLES_DIR
    test_img = list(EXAMPLES_DIR.glob("*.jpeg"))[0]
    label, conf, all_probs = predict_image(model, test_img)
    print(f"图片: {test_img.name}")
    print(f"预测: {label} ({CLASS_CN[label]})，置信度: {conf:.2%}")
    print(f"各类别概率: {all_probs}")
