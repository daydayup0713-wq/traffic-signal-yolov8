import os
import cv2
import warnings
from ultralytics import YOLO

# ----------------- 配置 -----------------
VIDEO_PATH = "ceshi.mp4"      # 输入视频路径
FRAME_DIR = "frames"                      # 存放抽帧的临时目录
INFER_PROJECT = "inference_frames"    # 推理结果输出目录
INFER_NAME = "video_6fps"                 # 推理子目录名称
FPS_TARGET = 6                              # 目标检测频率（每秒检测次数）
CONF_THRESH = 0.86                          # 置信度阈值
NMS_IOU = 0.5                               # NMS IoU 阈值
DEVICE = '0'                                # GPU 设备
WEIGHTS = "runs/traffic_v8n/train_na2/weights/best.pt"  # 训练好的模型权重

# ----------------- 抽帧 -----------------
os.makedirs(FRAME_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {VIDEO_PATH}")

orig_fps = cap.get(cv2.CAP_PROP_FPS)
# 计算采样间隔
interval = max(1, int(orig_fps / FPS_TARGET))
print(f"原始帧率: {orig_fps:.2f} fps, 抽帧间隔: 每 {interval} 帧保留 1 帧")

idx = 0
saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if idx % interval == 0:
        out_path = os.path.join(FRAME_DIR, f"frame_{saved:06d}.jpg")
        cv2.imwrite(out_path, frame)
        saved += 1
    idx += 1
cap.release()
print(f"共抽取 {saved} 张帧，用于推理。")

# ----------------- 加载模型并推理 -----------------
warnings.filterwarnings("ignore")
model = YOLO(WEIGHTS)
results = model.predict(
    source=FRAME_DIR,
    conf=CONF_THRESH,
    iou=NMS_IOU,
    save=True,
    project=INFER_PROJECT,
    name=INFER_NAME,
    exist_ok=True,
    device=DEVICE
)

# ----------------- 结束 -----------------
print(f"推理完成，结果保存在: {INFER_PROJECT}/{INFER_NAME}")
