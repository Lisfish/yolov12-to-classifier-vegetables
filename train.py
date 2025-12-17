import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 直接加载预训练权重（推荐方式，无需手动加载yaml+pt）
    # 注：确保yolov12n.pt已下载到代码根目录（E:\yolo12\yolov12-main\yolov12-main\）
    model = YOLO("yolov12n.pt")  # 自动匹配内置配置，无需指定yaml路径

    # 2. 训练参数优化（适配你的4类蔬菜数据集+RTX4060 Laptop GPU）
    model.train(
        data=r'E:\yolo12\yolov12-main\yolov12-main\dataset\data.yaml',  # 绝对路径避免找不到文件
        imgsz=640,                # 输入尺寸，与预训练一致
        epochs=50,                # 微调建议50轮（原1轮仅测试，无训练效果）
        batch=8,                  # RTX4060 8G显存适配batch=8（原4太小，训练效率低）
        workers=0,                # Windows系统建议设0，避免多线程报错
        device=0,                 # 指定GPU 0（原''会用CPU，训练极慢）
        warmup_epochs=3,           # 你的数据集不大，3轮足够
        warmup_momentum=0.8,       # 从较低的动量开始
        warmup_bias_lr=0.1,        # bias参数的学习率热身

    )