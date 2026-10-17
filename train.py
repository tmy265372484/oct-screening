"""
眸析 - OCT视网膜病变分类训练脚本
使用预训练ResNet50做迁移学习
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# ===== 配置 =====
DATA_DIR = r"C:\Users\LEGION\Doubao\chats\2026-10-15\new-chat-3\OCT-screening\data\OCT"
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 4

# ===== 加载数据 =====
print("加载训练数据...")
train_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print("类别:", class_names)

# ===== 数据增强 =====
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

# ===== 构建模型（迁移学习）=====
print("构建模型...")
base_model = keras.applications.ResNet50(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # 冻结预训练层

inputs = keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = keras.applications.resnet50.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ===== 训练 =====
print("\n开始训练...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# ===== 保存模型 =====
os.makedirs("models", exist_ok=True)
model.save("models/oct_classifier.keras")
print("\n模型已保存到 models/oct_classifier.keras")

# ===== 测试集评估 =====
print("\n在测试集上评估...")
test_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)
test_loss, test_acc = model.evaluate(test_ds)
print(f"测试集准确率: {test_acc:.4f}")
