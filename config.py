"""
全局配置
"""
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 数据集路径
DATA_DIR = BASE_DIR / "data" / "OCT"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
EXAMPLES_DIR = DATA_DIR / "examples"

# 模型保存路径
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "oct_classifier.keras"

# 图片参数
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
NUM_CLASSES = 4
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_CN = {
    "CNV": "脉络膜新生血管",
    "DME": "黄斑水肿",
    "DRUSEN": "玻璃膜疣",
    "NORMAL": "正常",
}

# 训练参数
EPOCHS = 50
LEARNING_RATE = 0.001
