# ============================================
# YOLOv26 检测与验证脚本
# 功能: 支持图像检测、视频检测、摄像头实时检测和模型验证
# 新增: 检测完成后自动创建txt文件夹，保存YOLO格式的识别结果
# ============================================

import warnings
import os
import sys
import argparse
warnings.filterwarnings('ignore')

from ultralytics import YOLO


def save_detection_results(results, txt_dir):
    """
    将检测结果保存为YOLO格式的txt文件
    
    参数:
        results: YOLO检测结果对象
        txt_dir: txt文件保存目录
    """
    os.makedirs(txt_dir, exist_ok=True)
    
    for result in results:
        img_path = result.path
        img_name = os.path.basename(img_path)
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(txt_dir, txt_name)
        
        img_height, img_width = result.orig_shape
        
        with open(txt_path, 'w') as f:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                x_center = round(x_center, 6)
                y_center = round(y_center, 6)
                width = round(width, 6)
                height = round(height, 6)
                
                f.write(f"{cls} {x_center} {y_center} {width} {height}\n")
    
    print(f"识别结果已保存到txt文件夹: {txt_dir}")


def detect_image(model, image_path, output_dir='runs/detect', name='exp', txt_dir='txt'):
    """
    对单张图像或图像目录进行检测
    
    参数:
        model: YOLO 模型对象
        image_path: 图像路径或图像目录路径
        output_dir: 输出目录
        name: 实验名称
        txt_dir: txt结果保存目录
    """
    print(f"\n检测图像: {image_path}")
    
    results = model.predict(
        source=image_path,
        imgsz=640,
        project=output_dir,
        name=name,
        save=True,
        conf=0.25,
        iou=0.72,
        show=False
    )
    
    save_detection_results(results, txt_dir)
    
    print(f"检测结果已保存到: {output_dir}/{name}")
    return results


def detect_video(model, video_path, output_dir='runs/detect', name='exp', txt_dir='txt'):
    """
    对视频文件进行检测
    
    参数:
        model: YOLO 模型对象
        video_path: 视频文件路径
        output_dir: 输出目录
        name: 实验名称
        txt_dir: txt结果保存目录
    """
    print(f"\n检测视频: {video_path}")
    
    results = model.predict(
        source=video_path,
        imgsz=640,
        project=output_dir,
        name=name,
        save=True,
        conf=0.25,
        iou=0.72,
        show=False
    )
    
    print(f"检测结果已保存到: {output_dir}/{name}")
    return results


def detect_camera(model, camera_id=0, output_dir='runs/detect', name='exp', txt_dir='txt'):
    """
    实时摄像头检测
    
    参数:
        model: YOLO 模型对象
        camera_id: 摄像头ID，0表示默认摄像头
        output_dir: 输出目录
        name: 实验名称
        txt_dir: txt结果保存目录
    """
    print(f"\n启动摄像头检测 (ID: {camera_id})")
    print("按 'q' 键退出")
    
    results = model.predict(
        source=camera_id,
        imgsz=640,
        project=output_dir,
        name=name,
        save=True,
        conf=0.25,
        iou=0.72,
        show=True
    )
    
    print(f"检测结果已保存到: {output_dir}/{name}")
    return results


def validate_model(model, data_path):
    """
    验证模型性能（在验证集上评估）
    
    参数:
        model: YOLO 模型对象
        data_path: 数据集配置文件路径
    """
    print("\n" + "=" * 50)
    print("开始模型验证")
    print("=" * 50)
    
    results = model.val(
        data=data_path,
        imgsz=640,
        conf=0.25,
        iou=0.72,
        verbose=True
    )
    
    print("\n验证结果:")
    print(f"mAP@0.5: {results.box.map:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map50_95:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    
    return results


def create_directories(output_dir):
    """自动创建检测脚本所需的目录"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"已创建/确认目录: {output_dir}")


def main():
    """
    主函数: 解析命令行参数并执行相应的检测或验证任务
    """
    parser = argparse.ArgumentParser(description='YOLOv26 检测与验证脚本')
    
    parser.add_argument(
        '--model', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detect/best.pt'),
        help='模型路径，支持预训练模型名称或本地权重文件路径'
    )
    
    parser.add_argument(
        '--data', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.yaml'),
        help='数据集配置文件路径（用于模型验证）'
    )
    
    parser.add_argument(
        '--source', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img_raw'),
        help='检测源：图像路径、图像目录、视频路径或摄像头ID（数字）'
    )
    
    parser.add_argument(
        '--mode', type=str, choices=['detect', 'validate'], default='detect',
        help='运行模式：detect（检测）或 validate（验证）'
    )
    
    parser.add_argument(
        '--output', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs/detect'),
        help='检测结果输出目录'
    )
    
    parser.add_argument(
        '--name', type=str, default='exp',
        help='实验名称，用于创建子目录'
    )
    
    parser.add_argument(
        '--txt-dir', type=str, default='txt',
        help='识别结果txt文件保存目录'
    )
    
    args = parser.parse_args()
    
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    create_directories(args.output)
    os.makedirs(args.txt_dir, exist_ok=True)
    
    model_path = args.model
    if not os.path.exists(model_path):
        pretrained_dir = os.path.join(PROJECT_ROOT, 'pretrained_models')
        os.makedirs(pretrained_dir, exist_ok=True)
        model_path_in_dir = os.path.join(pretrained_dir, os.path.basename(args.model))
        if os.path.exists(model_path_in_dir):
            model_path = model_path_in_dir
    
    print("=" * 50)
    print("YOLOv26 检测与验证脚本")
    print("=" * 50)
    print(f"模型: {model_path}")
    print(f"模式: {args.mode}")
    print(f"txt输出目录: {args.txt_dir}")
    
    try:
        model = YOLO(model_path)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)
    
    if args.mode == 'validate':
        validate_model(model, args.data)
    else:
        if args.source is None:
            print("错误: 检测模式需要指定 --source 参数")
            print("使用示例:")
            print("  python detect.py --source dataset/images/val/")
            print("  python detect.py --source ./test_video.mp4")
            print("  python detect.py --source 0")
            sys.exit(1)
        
        if args.source.isdigit():
            detect_camera(model, int(args.source), args.output, args.name, args.txt_dir)
        elif os.path.isdir(args.source):
            detect_image(model, args.source, args.output, args.name, args.txt_dir)
        elif os.path.isfile(args.source):
            if args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                detect_image(model, args.source, args.output, args.name, args.txt_dir)
            elif args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                detect_video(model, args.source, args.output, args.name, args.txt_dir)
            else:
                print(f"不支持的文件格式: {args.source}")
                sys.exit(1)
        else:
            print(f"无效的源路径: {args.source}")
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("任务完成！")
    print("=" * 50)
    print(f"识别结果txt文件已保存到: {args.txt_dir}")
    print("可使用 labelImg 工具对这些识别结果进行微调")


if __name__ == '__main__':
    main()