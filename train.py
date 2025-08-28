from ultralytics import YOLO
import torch
import warnings



if __name__ == '__main__':
    model = YOLO('./models/yolov8n.pt')
    warnings.filterwarnings("ignore")
    model.train(
        data=r"D:\study_python_project\yolov8_traffic\naruto.yaml",
        cache=False,
        imgsz=640,
        epochs=300,
        batch=16,
        workers=8,
        device='0',
        project='runs/traffic_v8n',
        name='train_s',
        close_mosaic=10,  # 最后10轮关闭 mosaic，防止过度增强
        fliplr=0.0,  # ❗禁用水平翻转（防止左右混淆）
        flipud=0.0,  # ✅禁用垂直翻转（一般没事，保险起见也关）
        hsv_h=0.0,  # 可选：关掉颜色扰动，防止信号灯颜色混乱
        hsv_s=0.0,
        hsv_v=0.0
    )
