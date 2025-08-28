traffic 项目 – 红绿灯检测与状态分类 (YOLOv8)
项目简介 / Project Introduction
项目 traffic 利用 Ultralytics 提供的 YOLOv8 模型对交通信号灯进行检测，并对其状态（红灯/黄灯/绿灯等）进行分类识别。该模型可以从视频或图像中实时识别交通灯的位置和颜色状态，帮助实现自动驾驶等应用场景下对红绿灯的自动检测。本项目使用 Python 编写，基于 Ultralytics 的 YOLOv8 库和 PyTorch 框架进行开发和训练。 The traffic project uses the Ultralytics YOLOv8 model to detect traffic lights and classify their status (red, yellow, green, etc.). The model can identify the location and color state of traffic lights in images or videos in real time, facilitating tasks such as automatic detection of traffic signals for autonomous driving applications. The project is implemented in Python and built on the Ultralytics YOLOv8 library and the PyTorch deep learning framework.
安装依赖 / Installation Requirements
请确保开发环境中安装了以下依赖和工具（以及相应版本）： Make sure the following dependencies and tools (with compatible versions) are installed in your environment:
Python: 3.8以上版本 / Python 3.8 or higher (tested on Python 3.9)
PyTorch: 深度学习框架 (需支持CUDA以利用GPU) / PyTorch deep learning framework (CUDA support recommended for GPU)
Ultralytics YOLOv8: YOLOv8库 (例如 v8.0.x) / Ultralytics YOLOv8 library (e.g. version 8.0.x)
OpenCV (cv2): 图像与视频处理库 / OpenCV library for image and video processing
预训练YOLOv8模型权重: 如 yolov8n.pt (用于模型初始化) / Pre-trained YOLOv8 model weights such as yolov8n.pt (for model initialization)
安装上述依赖的示例命令如下 (首先确保已安装合适版本的PyTorch): Use pip to install the required libraries (make sure to install an appropriate PyTorch version first):
pip install ultralytics opencv-python
(如果未提供 yolov8n.pt 文件，Ultralytics 库会在首次运行时自动下载预训练权重，或者您可以从官方仓库获取该文件)
(If the yolov8n.pt file is not included, the Ultralytics library will attempt to download the pre-trained weights on first use, or you can manually download it from the official YOLOv8 repository.)
数据准备 / Dataset Preparation
本项目使用了公开的交通灯数据集（已打包为 datas.zip）来训练模型。请按照以下步骤准备数据： This project uses a public traffic light dataset (provided as datas.zip) for training the model. Prepare the dataset as follows:
下载数据集：获取 datas.zip 文件，并将其下载到项目根目录。/ Download the dataset: Obtain the datas.zip file and place it in the project root directory.
解压数据集：将压缩包解压。例如，可以将内容解压到 data/ 目录。解压后，应包含图像和标签文件的子目录，如 data/images/ 和 data/labels/。/ Extract the dataset: Unzip the file (e.g., into a data/ folder). After extraction, you should have subfolders for images and labels, e.g. data/images/ and data/labels/, containing the training/validation images and their annotation files.
配置数据路径：确保模型训练使用正确的数据路径。通常数据集中会包含一个配置文件（如 traffic.yaml），其中指定了训练、验证集路径和类别信息。请根据解压后的实际路径修改该配置文件，或者在训练脚本中更新数据集路径。/ Configure dataset paths: Ensure the training will point to the correct dataset location. Typically, the dataset includes a configuration file (e.g. traffic.yaml) specifying the train/val directories and class names. Update this config file with the correct paths after unzipping, or adjust the dataset path in the training script accordingly.
数据集中标注了交通灯的不同状态为分类类别，例如红灯、黄灯、绿灯三种主要颜色（如果数据包含转向箭头灯或行人信号灯，它们也会作为独立类别进行标注）。每个图像对应一个标签文件（YOLO格式的文本，包含边界框坐标和类别ID）。 The dataset labels traffic lights in different states as separate classes – for example, the three primary colors: Red, Yellow, Green (if the dataset includes turn arrow signals or pedestrian crossing lights, those are labeled as additional classes). Each image has a corresponding annotation file (in YOLO text format, containing bounding box coordinates and class IDs).
模型训练 / Model Training
开始模型训练之前，请先打开并编辑 train.py，确保其中的数据配置路径指向您解压后的数据集。默认情况下，脚本使用路径 naruto.yaml 作为数据集配置文件（请将其修改为实际数据配置文件的路径，如 data/traffic.yaml）。同时确认预训练模型权重路径正确（默认为 ./models/yolov8n.pt）。准备就绪后，运行以下命令开始训练模型： Before starting training, open the train.py file and ensure the dataset path is correctly set to your extracted data. By default, the script is configured to use naruto.yaml as the dataset config file (you should change this to your actual dataset config file path, e.g. data/traffic.yaml). Also verify the pre-trained weights path is correct (default ./models/yolov8n.pt). Once configured, run the following command to start training the model:
python train.py
训练过程开始后，Ultralytics YOLOv8 将输出每个 epoch 的损失、精度等日志，并在训练完成时自动保存最优模型权重文件。默认配置下，训练将进行 300 个 epoch（您可以在 train.py 中调整 epochs 参数）。训练完成后，最好的模型权重保存在 runs/traffic_v8n/train_s/weights/best.pt。您可以使用该文件用于后续推理。 During training, Ultralytics YOLOv8 will print out logs for each epoch (loss, accuracy, mAP, etc.) and automatically save the best model weights. By default, the training runs for 300 epochs (you can adjust the epochs parameter in train.py if needed). After training completes, the best model weights are saved to runs/traffic_v8n/train_s/weights/best.pt. This weights file will be used for inference in the next steps.
视频推理 / Video Inference
训练完模型后，您可以使用提供的脚本对视频进行红绿灯检测。本项目提供了两种视频推理方式：一种是将视频抽帧批量推理并保存结果，另一种是实时读取视频逐帧显示检测结果。 After training the model, you can perform traffic light detection on videos using the provided scripts. The project offers two methods for video inference: one extracts frames from a video and processes them in batch (saving the results), and the other reads the video in real-time and displays the detection results frame by frame. 方法一 / Method 1: 使用 val.py 脚本对视频文件进行批处理推理。请先在 val.py 中将 VIDEO_PATH 设置为待检测的视频文件路径，然后运行 python val.py。脚本会以每秒约 6 帧的频率从视频中抽取帧，并对这些帧执行交通灯检测。检测完成后，带有标注框和类别标签的结果帧将保存至输出目录 inference_frames/video_6fps/（脚本中可配置）。您可以前往该目录查看每一帧的检测效果。 Method 1: Use the val.py script for batch inference on a video file. First, open val.py and set VIDEO_PATH to the path of the video you want to analyze, then run python val.py. The script will sample frames from the video at approximately 6 frames per second and perform traffic light detection on each extracted frame. After inference, the frames with bounding boxes and class labels drawn on the traffic lights will be saved to the output folder inference_frames/video_6fps/ (configurable in the script). You can check this directory to review the detection results frame by frame. 方法二 / Method 2: 使用 text.py 脚本进行实时视频检测。将 text.py 脚本中的 video_path 修改为您的视频文件路径，然后运行 python text.py 开始推理。该脚本将逐帧读取视频，并弹出一个窗口实时显示检测结果。每个检测到的交通灯都会以红色方框标出，并在其上方标注预测的灯色类别（如 Red、Green 等）。推理过程中窗口会连续播放视频，您可以手动关闭窗口（或等待视频播放结束）以停止推理。 Method 2: Use the text.py script for real-time video detection. Update the video_path in text.py to point to your video file, then run python text.py to start inference. This script will read the video frame by frame and open a window to display the detection in real time. Each detected traffic light is highlighted with a red bounding box, and a label (e.g., Red, Green, etc.) is drawn above the box indicating the predicted light color. The video will play continuously in the window; you can close the window (or wait for the video to finish) to stop the inference.
可视化演示 / Visualization
训练完成后，YOLOv8 会在 runs/traffic_v8n/train_s/ 目录下保存训练过程的可视化结果，例如 results.png（内含损失下降曲线、精度指标等）及混淆矩阵图表。这些图表可以帮助您直观评估模型的收敛情况和性能表现。 After training, YOLOv8 saves visualization artifacts in the runs/traffic_v8n/train_s/ folder, such as results.png (showing training loss curves, accuracy, mAP over epochs) and confusion matrix images. These charts allow you to visually assess the model’s convergence and performance. 在推理阶段，您可以通过前述的输出结果来直观展示模型的检测效果。例如，在输出的图像帧或视频中，每个交通信号灯被检测后都会绘制一个边界框，并标注其类别名称（红灯、黄灯、绿灯等）。通过查看这些带标注的结果图像，您可以验证模型是否正确识别了红绿灯及其状态。 For inference results, you can visualize the model’s performance by examining the output frames or video. For instance, in the saved result frames (or the real-time display), each detected traffic light is drawn with a bounding box and labeled with its class name (Red, Yellow, Green, etc.). By reviewing these annotated images, you can verify whether the model has correctly identified the traffic lights and their states. (提示：您可以根据需要调整置信度阈值 CONF_THRESH 等参数，以控制检测结果的置信度过滤。在可视化结果中，只有高于该阈值的检测才会被标出。)
(Tip: You can adjust parameters like the confidence threshold CONF_THRESH in the scripts to control the confidence level required for detections. In the visualized results, only detections above this threshold are shown.)
项目结构 / Project Structure
项目主要文件和目录结构如下所示： The project’s main files and directory structure are organized as follows:
traffic/
├── README.md               # 项目说明文档 (bilingual project README)
├── datas.zip               # 示例数据集压缩包 (example dataset archive)
├── data/                   # 解压后的数据集目录 (dataset directory after extracting datas.zip)
│   ├── images/             # 图像文件夹 (traffic light images for train/val)
│   └── labels/             # 标签文件夹 (YOLO格式标注 for corresponding images)
├── models/
│   └── yolov8n.pt          # 预训练YOLOv8n模型权重 (pre-trained YOLOv8n weights)
├── train.py                # 模型训练脚本 (training script for model fine-tuning)
├── val.py                  # 视频批量推理脚本 (video inference script with frame extraction)
├── text.py                 # 视频实时推理脚本 (video inference script for real-time display)
└── runs/                   # 输出结果目录 (outputs for training/inference runs)
    ├── traffic_v8n/train_s/        # 训练输出 (training output folder)
    │   ├── weights/best.pt         # 最优模型权重 (best model weights)
    │   ├── results.png             # 训练结果图表 (training curves and metrics)
    │   └── ...                     # 其他训练日志文件 (other training logs)
    └── inference_frames/video_6fps/  # 推理输出帧 (output frames from video inference)
        └── *.jpg                   # 带检测标注的图像文件 (images with detection boxes)
说明 / Note: runs/ 目录及其子文件在运行训练或推理脚本后自动生成。初始克隆仓库时可能不存在这些文件夹。训练前请确保提供数据集 (data/) 和预训练模型权重 (models/)。
Note: The runs/ directory and its contents are generated after running training or inference. They may not exist when you first clone the repository. Ensure that the dataset (data/) and pre-trained model weights (models/) are in place before training.
致谢 / Acknowledgments
感谢 Ultralytics 团队开源并维护了强大的 YOLOv8 模型和工具库，使我们能够方便地构建本项目。在此也感谢开源社区提供的交通灯数据集及相关资源，为模型训练和测试提供了支持。 We thank the Ultralytics team for open-sourcing and maintaining the powerful YOLOv8 model and library, which made this project possible. We also acknowledge the open-source community for providing traffic light datasets and related resources used for model training and testing.
