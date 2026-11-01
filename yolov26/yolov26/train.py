# ============================================
# YOLOv26 训练脚本
# 功能: 基于 Ultralytics 框架进行目标检测模型训练
# 包含数据集划分、模型训练、模型验证完整流程
# 支持内存管理辅助脚本的状态保存与恢复
# ============================================

import os
import sys
import warnings
import yaml
import shutil
from datetime import datetime

warnings.filterwarnings('ignore')

from ultralytics import YOLO
import torch

# TensorBoard 相关导入
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_WRITER_AVAILABLE = True
except ImportError:
    TB_WRITER_AVAILABLE = False


def _sanitize_path(path):
    """
    清理路径，处理含空格的路径问题
    通过创建符号链接到无空格路径，解决 TensorBoard 写入问题
    """
    if ' ' not in path:
        return path
    
    safe_dir = os.path.join(os.path.dirname(path), '_safe_path')
    os.makedirs(os.path.dirname(safe_dir), exist_ok=True)
    
    if not os.path.exists(safe_dir):
        try:
            os.symlink(path, safe_dir)
        except (OSError, NotImplementedError):
            safe_dir = path
    
    return safe_dir


def setup_safe_project_path(project_dir):
    """
    为含空格路径创建符号链接，解决 TensorBoard 写入失败问题
    返回无空格的安全路径
    """
    if ' ' not in project_dir:
        return project_dir
    
    safe_dir = "/home/fst/yolov26_safe_output"
    
    # 如果符号链接已存在且指向正确，直接返回
    if os.path.islink(safe_dir):
        target = os.path.realpath(safe_dir)
        if target == os.path.realpath(project_dir):
            return safe_dir
        else:
            os.remove(safe_dir)
    
    # 创建符号链接
    try:
        os.symlink(project_dir, safe_dir)
        print(f"🔗 路径含空格，已创建符号链接: {safe_dir} → {project_dir}")
        return safe_dir
    except (OSError, PermissionError):
        print(f"⚠️  无法创建符号链接，TensorBoard 可能无法正常写入")
        print(f"   请手动执行: ln -s '{project_dir}' {safe_dir}")
        return project_dir


def verify_tensorboard_events(log_dir):
    """验证 TensorBoard event 文件是否已生成"""
    # 同时检查原始路径和真实路径（处理符号链接）
    search_paths = [log_dir]
    try:
        real_path = os.path.realpath(log_dir)
        if real_path != log_dir:
            search_paths.append(real_path)
    except:
        pass
    
    for search_dir in search_paths:
        if not os.path.exists(search_dir):
            continue
        event_files = [f for f in os.listdir(search_dir)
                       if f.startswith('events.out.tfevents')]
        if event_files:
            print(f"✅ 找到 {len(event_files)} 个 event 文件 (目录: {search_dir})")
            for ef in event_files:
                file_size = os.path.getsize(os.path.join(search_dir, ef))
                print(f"   {ef} ({file_size:,} bytes)")
            return True
    
    print(f"⚠️  未找到 event 文件，已检查目录: {log_dir}")
    print("   可能原因:")
    print("   1. tensorboard 包未正确安装")
    print("   2. 路径含空格导致 TensorBoard 写入失败")
    print("   3. 训练尚未开始或已中断")
    print(f"   4. 检查 save_dir 路径: {log_dir}")
    return False


def check_environment():
    """检查运行环境，确保依赖和设备可用"""
    print("=" * 50)
    print("环境检查")
    print("=" * 50)
    
    print(f"PyTorch 版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA 可用，设备数量: {torch.cuda.device_count()}")
        print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        device = torch.device("cuda:0")
    else:
        print("CUDA 不可用，使用 CPU 训练")
        device = torch.device("cpu")
    
    return device


def save_training_state(model, epoch, results, save_dir):
    """
    保存训练状态，便于后续恢复
    
    参数:
        model: YOLO 模型对象
        epoch: 当前训练轮次
        results: 训练结果对象
        save_dir: 保存目录
    """
    state_dir = os.path.join(save_dir, 'training_state')
    os.makedirs(state_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(state_dir, f'checkpoint_epoch_{epoch}.pt')
    
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.model.state_dict(),
            'optimizer_state_dict': model.trainer.optimizer.state_dict() if hasattr(model.trainer, 'optimizer') else None,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        latest_link = os.path.join(state_dir, 'latest.pt')
        if os.path.exists(latest_link):
            os.remove(latest_link)
        try:
            os.symlink(checkpoint_path, latest_link)
        except:
            shutil.copy2(checkpoint_path, latest_link)
        
        print(f"\n训练状态已保存: {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        print(f"警告: 保存训练状态失败 - {e}")
        return None


def load_training_state(model, state_dir):
    """
    加载训练状态，恢复训练
    
    参数:
        model: YOLO 模型对象
        state_dir: 状态目录
    
    返回:
        恢复的轮次，如果没有找到状态文件返回 None
    """
    latest_path = os.path.join(state_dir, 'latest.pt')
    
    if os.path.exists(latest_path):
        try:
            checkpoint = torch.load(latest_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                model.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"已恢复模型权重")
            
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
                if hasattr(model.trainer, 'optimizer'):
                    model.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print(f"已恢复优化器状态")
            
            epoch = checkpoint.get('epoch', 0)
            print(f"训练状态恢复成功，从第 {epoch + 1} 轮继续")
            
            return epoch
        except Exception as e:
            print(f"警告: 加载训练状态失败 - {e}")
            return None
    else:
        print("未找到训练状态文件，从头开始训练")
        return None


def train_model(model, config_path, resume_epoch=None):
    """
    训练模型
    
    参数:
        model: YOLO 模型对象
        config_path: 训练配置文件路径（args.yaml）
        resume_epoch: 从第几轮恢复训练
    """
    print("\n" + "=" * 50)
    print("开始模型训练")
    print("=" * 50)
    
    print(f"\n加载配置文件: {config_path}")
    print("训练参数将从配置文件中读取...")
    
    if resume_epoch is not None:
        print(f"从第 {resume_epoch + 1} 轮恢复训练")
    
    try:
        results = model.train(cfg=config_path)
        
        print("\n训练完成！")
        print(f"训练结果保存目录: {results.save_dir}")
        
        return results
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"\n❌ CUDA out of memory 错误: {e}")
            raise
        else:
            raise
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        raise


def validate_model(model_path, data_path):
    """
    验证模型性能
    
    参数:
        model_path: 训练好的模型权重路径
        data_path: 数据集配置文件路径
    """
    print("\n" + "=" * 50)
    print("开始模型验证")
    print("=" * 50)
    
    model = YOLO(model_path)
    
    print(f"\n加载模型: {model_path}")
    print(f"验证数据集: {data_path}")
    
    results = model.val(data=data_path)
    
    print("\n验证结果:")
    print(f"mAP@0.5: {results.box.map:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map50_95:.4f}")
    
    return results


def main():
    """主函数: 执行完整的训练流程"""
    
    PROJECT_ROOT = "/home/fst/zzy work/yolov26"
    print(f"项目根目录: {PROJECT_ROOT}")
    
    config_path = os.path.join(PROJECT_ROOT, 'args.yaml')
    data_path = os.path.join(PROJECT_ROOT, 'data.yaml')
    
    # 环境检查
    device = check_environment()
    
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在 - {config_path}")
        sys.exit(1)
    
    if not os.path.exists(data_path):
        print(f"错误: 数据集配置文件不存在 - {data_path}")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    model_name = cfg.get('model', 'yolo26s.pt')
    model = YOLO(model_name)
    
    print(f"\n加载模型: {model_name}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    
    # 确保输出目录存在
    project_dir = cfg.get('project', os.path.join(PROJECT_ROOT, 'output'))
    os.makedirs(project_dir, exist_ok=True)
    
    # 为含空格路径创建符号链接，解决 TensorBoard 写入问题
    safe_project = setup_safe_project_path(project_dir)
    
    # 开始训练，使用无空格路径确保 TensorBoard 正常工作
    if safe_project != project_dir:
        print(f"使用安全路径进行训练: {safe_project}")
        train_results = model.train(cfg=config_path, project=safe_project)
    else:
        train_results = model.train(cfg=config_path)
    
    # 验证 TensorBoard event 文件
    if train_results:
        save_dir = train_results.save_dir
        print(f"\n训练结果保存目录: {save_dir}")
        verify_tensorboard_events(save_dir)
    
    # 模型验证
    best_model_path = os.path.join(train_results.save_dir, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        validate_model(best_model_path, data_path)
    else:
        print(f"警告: 最佳模型权重文件不存在 - {best_model_path}")
    
    print("\n" + "=" * 50)
    print("训练流程全部完成！")
    print("=" * 50)


if __name__ == '__main__':
    main()