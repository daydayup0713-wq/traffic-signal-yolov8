# traffic — 交通信号灯检测与状态识别 (YOLOv8) / Traffic Light Detection & Status Classification (YOLOv8)

> **CN**：使用 Ultralytics YOLOv8 实现交通信号灯（红/黄/绿等）目标检测与状态分类，支持视频与图像。  
> **EN**: Traffic light (red/yellow/green) detection and status classification using Ultralytics YOLOv8, supporting images and videos.

---

## ✨ 功能 / Features

- **CN**：红绿灯目标检测与状态分类；支持离线批量推理与实时显示；可训练自定义数据集  
- **EN**: Detect traffic lights and classify their states; batch inference & real-time display; train on custom datasets

---

## 🧩 环境与版本 / Environment & Versions

- **Python**: 3.9（3.8–3.11 兼容） / 3.9 (3.8–3.11 compatible)  
- **PyTorch**: 2.0+（建议安装匹配 CUDA 的官方版本）/ 2.0+ (install CUDA-matched build from the official site)  
- **Ultralytics (YOLOv8)**: 8.2+  
- **OpenCV**: 4.7+

**安装 / Install**
```bash
# 创建环境（可选）/ create venv (optional)
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

# 先根据官方指引安装 PyTorch（含 CUDA），再安装以下包
pip install ultralytics opencv-python
```
> 如需 GPU 推理/训练，请先从 https://pytorch.org 获取与你 CUDA 版本匹配的 PyTorch。

---

## 📦 数据准备 / Dataset Preparation

- 仓库内提供示例数据压缩包 **`datas.zip`**。解压到项目根目录的 `data/`：  
  The repo includes **`datas.zip`**. Unzip it into `data/` at project root.

```
data/
├─ images/
│  ├─ train/  # 训练图片 / training images
│  └─ val/    # 验证图片 / validation images
└─ labels/
   ├─ train/  # YOLO txt 标注 / YOLO txt labels
   └─ val/
```

- 数据集配置 YAML（示例 `data/traffic.yaml`，请按你的实际类目修改）：  
  Dataset config YAML (example `data/traffic.yaml`, edit to your classes):

```yaml
# data/traffic.yaml
path: data
train: images/train
val: images/val
nc: 3
names: [red, yellow, green]
```

> 请确保 `images/*` 与 `labels/*` 的文件名一一对应，标签为 YOLO txt 标注（class cx cy w h，相对坐标）。

---

## 🗂️ 项目结构 / Project Structure

```
traffic/
├─ README.md
├─ datas.zip                 # 示例数据 / example dataset (zip)
├─ data/                     # 解压后数据 / unzipped dataset
├─ models/
│  └─ yolov8n.pt             # 预训练权重 / pre-trained weights (optional)
├─ train.py                  # 训练脚本 / training script
├─ val.py                    # 抽帧批量推理 / batch inference with frame sampling
├─ text.py                   # 实时逐帧显示推理 / real-time frame-by-frame display
└─ runs/                     # 训练与推理输出 / training & inference outputs
   ├─ traffic_v8n/train_s/weights/best.pt
   └─ inference_frames/video_6fps/
```
> 提示 / Note: `runs/` 内容在首次训练/推理后生成。`models/yolov8n.pt` 可由 Ultralytics 自动下载或手动放置。

---

## 🏋️‍♂️ 训练 / Training

### 方式 A：使用仓库脚本 / Use repository script
1) **编辑 `train.py`**，将 `data=...` 指向你的数据配置（如 `data/traffic.yaml`）；确保 `model='./models/yolov8n.pt'` 存在或可下载。  
2) **运行 / Run**：
```bash
python train.py
```

`train.py` 的关键参数（可在脚本中调整）/ Key params in `train.py`:
- `imgsz=640`, `epochs=300`, `batch=16`, `device='0'`
- 输出目录：`project='runs/traffic_v8n', name='train_s'`
- 数据增强（避免颜色/翻转干扰信号灯）：`close_mosaic=10`, `fliplr=0.0`, `flipud=0.0`, `hsv_h/s/v=0.0`

训练完成后，最优权重位于 / After training, best weights:
```
runs/traffic_v8n/train_s/weights/best.pt
```

### 方式 B：YOLOv8 CLI（等价）/ YOLOv8 CLI (equivalent)
```bash
yolo detect train \
  data=data/traffic.yaml model=models/yolov8n.pt \
  imgsz=640 epochs=300 batch=16 device=0 \
  project=runs/traffic_v8n name=train_s \
  close_mosaic=10 fliplr=0.0 flipud=0.0 hsv_h=0.0 hsv_s=0.0 hsv_v=0.0
```

---

## 🎬 推理 / Inference

### 方法一：抽帧批量推理（`val.py`）/ Batch via frame sampling (`val.py`)
在 `val.py` 中设置 / Set in `val.py`:
- `VIDEO_PATH = "ceshi.mp4"`（替换为你的输入视频 / your input video）
- `WEIGHTS = "runs/traffic_v8n/train_s/weights/best.pt"`（或自定义权重 / or your weights）
- 可选：`FPS_TARGET=6`, `CONF_THRESH=0.86`, `NMS_IOU=0.5`, `DEVICE='0'`

**运行 / Run**
```bash
python val.py
```
输出结果保存到 / Annotated frames saved to:
```
inference_frames/video_6fps/
```

### 方法二：实时逐帧显示（`text.py`）/ Real-time display (`text.py`)
在 `text.py` 中设置 / Set in `text.py`:
- `model = YOLO("runs/traffic_v8n/train_s/weights/best.pt")`
- `video_path = "ceshi.mp4"`（或摄像头索引 0 / or webcam index 0）

**运行 / Run**
```bash
python text.py
```
将打开窗口实时显示检测结果；关闭窗口结束推理。  
A window shows real-time detections; close to stop.

### 方法三：YOLOv8 CLI 直接推理 / Direct CLI predict
```bash
# 图片/文件夹/视频/通配符均可 / image, folder, video, glob supported
yolo detect predict \
  source=ceshi.mp4 \
  model=runs/traffic_v8n/train_s/weights/best.pt \
  conf=0.5 device=0 save=True
```

---

## 📊 可视化 / Visualization

- 训练图表与日志（如 `results.png`、混淆矩阵等）位于 `runs/traffic_v8n/train_s/`。  
- 推理后的图片帧位于 `inference_frames/video_6fps/`。

Training charts/logs in `runs/traffic_v8n/train_s/`; annotated frames in `inference_frames/video_6fps/`.

---

## 🛠️ 常见问题 / FAQ

- **权重路径不一致 / Weights path mismatch**  
  将 `val.py` 与 `text.py` 中的权重路径统一为训练产生的 `runs/traffic_v8n/train_s/weights/best.pt`。

- **GPU/CPU 切换 / Switch GPU/CPU**  
  `DEVICE='0'` 表示使用第 0 块 GPU；改为 `DEVICE='cpu'` 可在 CPU 上运行。

- **阈值调优 / Threshold tuning**  
  提高 `CONF_THRESH` 可减少误检但可能增加漏检；`NMS_IOU` 控制重叠框抑制强度。

---

## 🙏 致谢 / Acknowledgments

- Ultralytics YOLOv8 与开源社区 / Ultralytics YOLOv8 and the open-source community  
- 数据集版权归原作者所有 / Dataset copyrights belong to their authors

---

## 📝 许可 / License

- 代码许可以仓库 License 为准；数据使用受其各自许可与条款约束。  
  Code is governed by the repository’s license; datasets are subject to their own licenses/terms.

---

## 🚀 快速开始 / Quick Start

```bash
# 1) 安装依赖 / install deps
pip install ultralytics opencv-python

# 2) 准备数据 / prepare data
unzip datas.zip -d data

# 3) 训练 / train
python train.py

# 4) 推理（抽帧批量）/ predict (batch)
python val.py

# 5) 推理（实时显示）/ predict (realtime)
python text.py
```
> 欢迎提交 Issue/PR 以改进数据与模型配置。Contributions are welcome!
