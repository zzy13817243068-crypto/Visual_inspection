=======================================================
          YOLOv26 目标检测项目 - Linux 使用指南
=======================================================

一、项目介绍
---------------------------------------
本项目基于 Ultralytics YOLOv26 框架，用于目标检测模型的训练、
验证和推理。包含完整的数据集划分、模型训练、检测推理流程。

二、目录结构
---------------------------------------
yolov26/
├── dataset/          # 数据集目录
│   ├── images_raw/   # 原始图片（放这里）
│   ├── labels_raw/   # 原始标签（放这里）
│   ├── images/       # 划分后的图片
│   └── labels/       # 划分后的标签
├── output/           # 训练结果输出
├── runs/             # 检测结果输出
├── args.yaml         # 训练参数配置
├── data.yaml         # 数据集配置
├── train.py          # 训练脚本
├── detect.py         # 检测脚本
├── split_dataset.py  # 数据集划分脚本
├── memory_manager.py # 内存管理脚本
├── init_directories.py # 目录初始化脚本
└── requirements.txt  # 依赖清单

三、环境准备
---------------------------------------

1. 更新系统
   sudo apt update && sudo apt upgrade -y

2. 安装 Python 和基础工具
   sudo apt install -y python3.10 python3.10-dev python3.10-venv python3-pip build-essential libgl1-mesa-glx libglib2.0-0

3. 创建虚拟环境
   cd "/home/fst/zzy work/yolov26"
   python3.10 -m venv venv
   source venv/bin/activate

4. 安装依赖
   pip install -r requirements.txt

5. 安装 PyTorch（GPU版本）
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   如果没有GPU，安装CPU版本：
   pip install torch torchvision torchaudio

6. 验证环境
   python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

四、数据准备
---------------------------------------

1. 将图片放入 dataset/images_raw/ 目录
2. 将标签放入 dataset/labels_raw/ 目录
3. 图片和标签文件名必须一致（如 image001.jpg 和 image001.txt）
4. 标签格式为 YOLO 格式：<类别> <x_center> <y_center> <width> <height>

五、数据集划分
---------------------------------------
执行以下命令划分训练集和验证集：

cd "/home/fst/zzy work/yolov26"
source venv/bin/activate
python split_dataset.py

输出示例：
总的图片数目：14846
总的标签数目：4256
有效的（有标签的）图片数目：4256
训练集：3617，验证集：639，测试集：0

六、模型训练
---------------------------------------

1. 修改配置（可选）
   编辑 args.yaml 文件，可修改：
   - epochs: 训练轮数（默认60）
   - batch: 批次大小（默认8）
   - model: 预训练模型（默认yolo26s.pt）

2. 开始训练
   cd "/home/fst/zzy work/yolov26"
   source venv/bin/activate
   python train.py

3. 使用内存管理（自动处理CUDA内存不足）
   python memory_manager.py

4. 使用后台训练（推荐）
   nohup python train.py > train.log 2>&1 &
   tail -f train.log

七、模型验证
---------------------------------------
训练完成后自动验证，也可手动验证：

python detect.py --mode validate --model "/home/fst/zzy work/yolov26/output/exp-det/weights/best.pt"

八、目标检测
---------------------------------------

1. 检测图片目录
   python detect.py --source "/home/fst/zzy work/yolov26/dataset/images/val"

2. 检测视频目录
   python detect.py --source "/home/fst/zzy work/detect"

3. 检测单张图片
   python detect.py --source test.jpg

4. 检测视频文件
   python detect.py --source test.mp4

5. 指定模型路径
   python detect.py --source "/home/fst/zzy work/detect" --model "/home/fst/zzy work/yolov26/detect/best.pt"

九、常见问题
---------------------------------------

1. CUDA out of memory
   解决：降低batch大小，在args.yaml中修改 batch: 4

2. 找不到模型文件
   解决：确保yolo26s.pt已下载到项目根目录

3. 标签不匹配
   解决：检查图片和标签文件名是否一致

4. 安装速度慢
   解决：使用国内镜像
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

5. libGL.so.1 not found
   解决：sudo apt install libgl1-mesa-glx

6. 路径含空格问题
   解决：使用引号包裹路径
   cd "/home/fst/zzy work/yolov26"

十、训练结果
---------------------------------------
训练完成后，结果保存在 output/exp-det/ 目录：
- weights/best.pt    # 最佳模型
- weights/last.pt    # 最后一轮模型
- results.png        # 训练曲线
- confusion_matrix.png # 混淆矩阵

十一、完整命令汇总
---------------------------------------

# 一次性安装所有依赖
cd "/home/fst/zzy work/yolov26" && \
python3.10 -m venv venv && \
source venv/bin/activate && \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
pip install -r requirements.txt

# 数据准备
python split_dataset.py

# 开始训练
python train.py

# 目标检测
python detect.py --source "/home/fst/zzy work/detect" --model "/home/fst/zzy work/yolov26/output/exp-det/weights/best.pt"

=======================================================
                     结束
=======================================================