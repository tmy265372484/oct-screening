"""
Grad-CAM 可解释热力图
在OCT原图上高亮模型关注的病灶区域
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from config import MODEL_PATH, IMG_SIZE, CLASS_NAMES, EXAMPLES_DIR


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):
    """生成Grad-CAM热力图"""
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        loss = preds[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_gradcam(img_path, heatmap, save_path, alpha=0.4):
    """把热力图叠加到原图上保存"""
    img = Image.open(img_path).convert("L")
    img = img.resize(IMG_SIZE)

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")
    plt.imshow(heatmap, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Grad-CAM 病灶热力图")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"热力图已保存: {save_path}")


if __name__ == "__main__":
    model = keras.models.load_model(MODEL_PATH)
    test_img = list(EXAMPLES_DIR.glob("*.jpeg"))[0]

    img = Image.open(test_img).convert("L").resize(IMG_SIZE)
    img_array = np.stack([img, img, img], axis=-1)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    heatmap = make_gradcam_heatmap(img_array, model)
    save_gradcam(test_img, heatmap, "screenshots/03_gradcam示例.png")
