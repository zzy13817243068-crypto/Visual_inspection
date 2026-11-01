# ============================================
# YOLOv26 内存管理辅助脚本
# 功能: 自动监测CUDA内存溢出错误，智能调整训练参数，重新启动训练
# 核心特性:
#   1. 实时错误监测
#   2. 优先级参数调整策略
#   3. 渐进式参数降低逻辑
#   4. 训练状态保存与恢复
#   5. 详细日志记录
#   6. 版本控制与自动恢复机制
# ============================================

import os
import sys
import subprocess
import time
import yaml
import shutil
import logging
import hashlib
from datetime import datetime
from pathlib import Path


class FileVersionController:
    """文件版本控制器"""
    
    def __init__(self, file_path, backup_dir=None):
        self.file_path = file_path
        self.backup_dir = backup_dir if backup_dir else os.path.join(os.path.dirname(file_path), 'backups')
        os.makedirs(self.backup_dir, exist_ok=True)
        
        self.original_content = None
        self.original_hash = None
        self.modification_history = []
        self.version_counter = 0
        
        self._create_initial_backup()
    
    def _create_initial_backup(self):
        """创建初始备份，保存原始内容和哈希值"""
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.original_content = f.read()
            
            self.original_hash = self._compute_hash(self.original_content)
            self._save_backup(self.original_content, 'original')
            
            print(f"[版本控制] 已创建初始备份")
            print(f"[版本控制] 原始文件哈希: {self.original_hash}")
        else:
            print(f"[版本控制] 警告: 文件不存在 - {self.file_path}")
    
    def _compute_hash(self, content):
        """计算内容的MD5哈希值"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _save_backup(self, content, version_tag):
        """保存备份文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{os.path.basename(self.file_path)}_{version_tag}_{timestamp}.bak"
        backup_path = os.path.join(self.backup_dir, backup_filename)
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return backup_path
    
    def save_modification(self, new_content, reason="参数调整"):
        """保存修改内容并记录历史"""
        self.version_counter += 1
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            current_content = f.read()
        
        modification_record = {
            'version': self.version_counter,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'reason': reason,
            'previous_hash': self._compute_hash(current_content),
            'new_hash': self._compute_hash(new_content),
            'backup_path': self._save_backup(new_content, f'v{self.version_counter}')
        }
        
        self.modification_history.append(modification_record)
        
        print(f"[版本控制] 已保存版本 {self.version_counter}")
        print(f"[版本控制] 修改原因: {reason}")
    
    def restore_original(self):
        """恢复到原始状态"""
        if self.original_content is not None:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(self.original_content)
            
            current_hash = self._compute_hash(self.original_content)
            if current_hash == self.original_hash:
                print(f"[版本控制] ✅ 已成功恢复到原始状态")
                print(f"[版本控制] 恢复后哈希: {current_hash}")
                return True
            else:
                print(f"[版本控制] ❌ 恢复失败，哈希不匹配")
                return False
        else:
            print(f"[版本控制] 警告: 没有原始备份可恢复")
            return False
    
    def generate_change_record(self, output_path=None):
        """生成详细的变更记录文件"""
        if output_path is None:
            record_dir = os.path.join(os.path.dirname(self.file_path), 'reports')
            os.makedirs(record_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(record_dir, f'change_record_{timestamp}.txt')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("YOLOv26 内存管理 - 参数变更记录\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"记录生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"目标文件: {self.file_path}\n")
            f.write(f"原始文件哈希: {self.original_hash}\n")
            f.write(f"总修改次数: {self.version_counter}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("【初始状态】- 修改前的原始配置\n")
            f.write("=" * 80 + "\n")
            f.write(self.original_content)
            f.write("\n\n")
            
            if self.modification_history:
                for i, record in enumerate(self.modification_history, 1):
                    f.write("=" * 80 + "\n")
                    f.write(f"【修改 #{i}】版本 {record['version']}\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"时间: {record['timestamp']}\n")
                    f.write(f"原因: {record['reason']}\n")
                    f.write(f"修改前哈希: {record['previous_hash']}\n")
                    f.write(f"修改后哈希: {record['new_hash']}\n")
                    f.write(f"备份路径: {record['backup_path']}\n\n")
                    
                    with open(record['backup_path'], 'r', encoding='utf-8') as bf:
                        f.write("修改后的内容:\n")
                        f.write("-" * 40 + "\n")
                        f.write(bf.read())
                        f.write("-" * 40 + "\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("【最终状态】- 训练完成时的配置\n")
            f.write("=" * 80 + "\n")
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r', encoding='utf-8') as f2:
                    f.write(f2.read())
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("记录结束\n")
            f.write("=" * 80 + "\n")
        
        print(f"[版本控制] 变更记录已生成: {output_path}")
        return output_path


class MemoryManager:
    """内存管理辅助类"""
    
    def __init__(self, project_root, config_path='args.yaml', train_script='train.py'):
        self.project_root = project_root
        self.config_path = os.path.join(project_root, config_path)
        self.train_script = os.path.join(project_root, train_script)
        
        self.max_retries = 10
        self.current_retry = 0
        
        self.original_config = {}
        self.current_config = {}
        self.adjustment_history = []
        
        self.version_controller = None
        
        self._init_logging()
        self._init_version_control()
        self._load_original_config()
        
        self.param_adjustment_order = [
            {'name': 'cache', 'type': 'bool', 'priority': 1, 'action': 'disable', 'desc': '禁用数据缓存', 'adjustable': True},
            {'name': 'batch', 'type': 'int', 'priority': 2, 'action': 'reduce', 'factor': 0.7, 'min_value': 1, 'desc': '批次大小降低30%', 'adjustable': True, 'repeatable': True},
            {'name': 'imgsz', 'type': 'int', 'priority': 3, 'action': 'reduce', 'factor': 0.875, 'steps': [640, 576, 512, 448, 384, 320], 'desc': '图像尺寸按阶梯减小', 'adjustable': True},
            {'name': 'workers', 'type': 'int', 'priority': 4, 'action': 'reduce', 'factor': 0.5, 'min_value': 0, 'desc': '数据加载线程数减半', 'adjustable': True, 'repeatable': True},
            {'name': 'amp', 'type': 'bool', 'priority': 5, 'action': 'force_enable', 'desc': '强制启用混合精度', 'adjustable': True},
            {'name': 'close_mosaic', 'type': 'int', 'priority': 6, 'action': 'reduce', 'factor': 1.0, 'steps': [10, 5, 0], 'desc': '提前关闭马赛克增强', 'adjustable': True},
            {'name': 'mosaic', 'type': 'float', 'priority': 7, 'action': 'reduce', 'factor': 0.5, 'min_value': 0.0, 'desc': '马赛克增强概率降低50%', 'adjustable': True, 'repeatable': True},
            {'name': 'copy_paste', 'type': 'float', 'priority': 8, 'action': 'reduce', 'factor': 0.5, 'min_value': 0.0, 'desc': '复制粘贴增强概率降低50%', 'adjustable': True, 'repeatable': True},
            {'name': 'mixup', 'type': 'float', 'priority': 9, 'action': 'reduce', 'factor': 0.5, 'min_value': 0.0, 'desc': 'MixUp增强概率降低50%', 'adjustable': True, 'repeatable': True},
            {'name': 'augment', 'type': 'bool', 'priority': 10, 'action': 'disable', 'desc': '禁用数据增强', 'adjustable': True},
            {'name': 'rect', 'type': 'bool', 'priority': 11, 'action': 'enable', 'desc': '启用矩形训练', 'adjustable': True},
            {'name': 'hsv_h', 'type': 'float', 'priority': 12, 'action': 'reduce', 'factor': 0.5, 'min_value': 0.0, 'desc': 'HSV色调扰动降低50%', 'adjustable': True, 'repeatable': True},
            {'name': 'hsv_s', 'type': 'float', 'priority': 13, 'action': 'reduce', 'factor': 0.5, 'min_value': 0.0, 'desc': 'HSV饱和度扰动降低50%', 'adjustable': True, 'repeatable': True},
            {'name': 'hsv_v', 'type': 'float', 'priority': 14, 'action': 'reduce', 'factor': 0.5, 'min_value': 0.0, 'desc': 'HSV亮度扰动降低50%', 'adjustable': True, 'repeatable': True},
            {'name': 'degrees', 'type': 'float', 'priority': 15, 'action': 'reduce', 'factor': 0.5, 'min_value': 0.0, 'desc': '旋转角度降低50%', 'adjustable': True, 'repeatable': True},
            {'name': 'translate', 'type': 'float', 'priority': 16, 'action': 'reduce', 'factor': 0.5, 'min_value': 0.0, 'desc': '平移比例降低50%', 'adjustable': True, 'repeatable': True},
            {'name': 'scale', 'type': 'float', 'priority': 17, 'action': 'reduce', 'factor': 0.5, 'min_value': 0.0, 'desc': '缩放比例降低50%', 'adjustable': True, 'repeatable': True},
            {'name': 'shear', 'type': 'float', 'priority': 18, 'action': 'reduce', 'factor': 0.5, 'min_value': 0.0, 'desc': '剪切角度降低50%', 'adjustable': True, 'repeatable': True},
            {'name': 'perspective', 'type': 'float', 'priority': 19, 'action': 'reduce', 'factor': 0.5, 'min_value': 0.0, 'desc': '透视变换强度降低50%', 'adjustable': True, 'repeatable': True},
            {'name': 'dropout', 'type': 'float', 'priority': 20, 'action': 'reduce', 'factor': 0.5, 'min_value': 0.0, 'desc': 'Dropout比例降低50%', 'adjustable': True, 'repeatable': True},
            {'name': 'model', 'type': 'str', 'priority': 21, 'action': 'downgrade', 'models': ['yolo26m.pt', 'yolo26s.pt', 'yolo26n.pt'], 'desc': '降级到更小的模型', 'adjustable': True},
            {'name': 'freeze', 'type': 'int', 'priority': 22, 'action': 'increase', 'factor': 1.5, 'min_value': 10, 'max_value': 50, 'desc': '冻结更多层', 'adjustable': True, 'repeatable': True},
            {'name': 'nbs', 'type': 'int', 'priority': 23, 'action': 'reduce', 'factor': 0.5, 'min_value': 8, 'desc': '名义批次大小降低50%', 'adjustable': True, 'repeatable': True},
        ]
        
        self.last_optimized_param_index = -1
        self.adjustment_counts = {}
        
        self.log.info("=" * 60)
        self.log.info("Memory Manager 初始化完成")
        self.log.info(f"项目根目录: {self.project_root}")
        self.log.info(f"配置文件: {self.config_path}")
        self.log.info(f"训练脚本: {self.train_script}")
        self.log.info(f"最大重试次数: {self.max_retries}")
        self.log.info("=" * 60)
    
    def _init_version_control(self):
        """初始化版本控制"""
        self.version_controller = FileVersionController(self.config_path)
        self.log.info(f"版本控制已初始化，备份目录: {self.version_controller.backup_dir}")
    
    def _init_logging(self):
        """初始化日志系统"""
        log_dir = os.path.join(self.project_root, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'memory_manager_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        self.log = logging.getLogger('MemoryManager')
        self.log.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        self.log.addHandler(file_handler)
        self.log.addHandler(console_handler)
        
        self.log_file_path = log_file
    
    def _load_original_config(self):
        """加载原始配置文件"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.original_config = yaml.safe_load(f)
            self.current_config = self.original_config.copy()
            self.log.info(f"已加载原始配置文件")
        else:
            self.log.error(f"配置文件不存在: {self.config_path}")
            sys.exit(1)
    
    def _save_config(self):
        """保存当前配置到文件"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.current_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    def _adjust_parameter(self):
        """按照优先级调整参数，支持重复调整"""
        total_strategies = len(self.param_adjustment_order)
        search_count = 0
        
        while search_count < total_strategies * 2:
            self.last_optimized_param_index += 1
            if self.last_optimized_param_index >= total_strategies:
                self.last_optimized_param_index = 0
            
            param_info = self.param_adjustment_order[self.last_optimized_param_index]
            param_name = param_info['name']
            action = param_info['action']
            
            if param_name not in self.current_config:
                search_count += 1
                continue
            
            original_value = self.current_config[param_name]
            new_value = original_value
            
            adjustment_count = self.adjustment_counts.get(param_name, 0)
            is_repeatable = param_info.get('repeatable', False)
            
            if not is_repeatable and adjustment_count > 0:
                search_count += 1
                continue
            
            if action == 'reduce':
                if 'steps' in param_info:
                    steps = param_info['steps']
                    if original_value in steps:
                        current_idx = steps.index(original_value)
                        if current_idx < len(steps) - 1:
                            new_value = steps[current_idx + 1]
                        else:
                            search_count += 1
                            continue
                    else:
                        new_value = steps[0]
                else:
                    factor = param_info['factor']
                    if param_info['type'] == 'int':
                        new_value = int(original_value * factor)
                    else:
                        new_value = round(original_value * factor, 4)
                    
                    if 'min_value' in param_info and new_value < param_info['min_value']:
                        if is_repeatable:
                            new_value = param_info['min_value']
                        else:
                            search_count += 1
                            continue
            
            elif action == 'increase':
                factor = param_info['factor']
                if param_info['type'] == 'int':
                    new_value = int(original_value * factor)
                else:
                    new_value = round(original_value * factor, 4)
                
                if 'max_value' in param_info and new_value > param_info['max_value']:
                    if is_repeatable:
                        new_value = param_info['max_value']
                    else:
                        search_count += 1
                        continue
            
            elif action == 'disable':
                if original_value is not False:
                    new_value = False
                else:
                    search_count += 1
                    continue
            
            elif action == 'enable':
                if original_value is not True:
                    new_value = True
                else:
                    search_count += 1
                    continue
            
            elif action == 'force_enable':
                if original_value is not True:
                    new_value = True
                else:
                    search_count += 1
                    continue
            
            elif action == 'downgrade':
                models = param_info['models']
                current_model = os.path.basename(original_value)
                if current_model in models:
                    idx = models.index(current_model)
                    if idx < len(models) - 1:
                        new_value = os.path.join(self.project_root, models[idx + 1])
                    else:
                        search_count += 1
                        continue
                else:
                    search_count += 1
                    continue
            
            if new_value == original_value:
                search_count += 1
                continue
            
            self.current_config[param_name] = new_value
            self._save_config()
            
            self.adjustment_counts[param_name] = adjustment_count + 1
            
            self.version_controller.save_modification(
                new_content=self._get_current_config_content(),
                reason=f"{param_info['desc']}: {param_name} 从 {original_value} 调整为 {new_value} (第{adjustment_count + 1}次调整)"
            )
            
            adjustment_record = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'param_name': param_name,
                'original_value': original_value,
                'new_value': new_value,
                'adjustment_reason': param_info['desc'],
                'priority': param_info['priority'],
                'retry_count': self.current_retry,
                'adjustment_count': adjustment_count + 1
            }
            self.adjustment_history.append(adjustment_record)
            
            self.log.info("=" * 60)
            self.log.info(f"参数调整: {param_name}")
            self.log.info(f"  原始值: {original_value}")
            self.log.info(f"  新值: {new_value}")
            self.log.info(f"  调整原因: {param_info['desc']}")
            self.log.info(f"  参数优先级: {param_info['priority']}")
            self.log.info(f"  当前重试次数: {self.current_retry}")
            self.log.info(f"  该参数已调整次数: {adjustment_count + 1}")
            self.log.info("=" * 60)
            
            return True
        
        self.log.error("所有参数调整策略已用尽，无法继续优化")
        return False
    
    def _get_current_config_content(self):
        """获取当前配置文件的完整内容"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _detect_oom_error(self, output):
        """检测输出中是否包含CUDA内存溢出错误"""
        oom_patterns = [
            'CUDA out of memory',
            'Out of memory',
            'RuntimeError: CUDA out of memory',
            'CudaError: out of memory',
            'torch.cuda.OutOfMemoryError',
        ]
        
        for pattern in oom_patterns:
            if pattern.lower() in output.lower():
                return True
        return False
    
    def _get_gpu_info(self):
        """获取GPU信息"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    info = lines[0].strip().split(',')
                    gpu_name = info[0].strip()
                    total_mem = info[1].strip()
                    used_mem = info[2].strip()
                    return f"{gpu_name} | 显存总量: {total_mem} | 当前使用: {used_mem}"
        except Exception as e:
            self.log.warning(f"获取GPU信息失败: {e}")
        
        return "GPU信息获取失败"
    
    def _launch_training(self):
        """启动训练进程"""
        self.log.info(f"\n启动训练进程...")
        self.log.info(f"GPU状态: {self._get_gpu_info()}")
        
        cmd = [sys.executable, self.train_script]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.project_root,
                bufsize=1
            )
            
            full_output = []
            while True:
                line = process.stdout.readline()
                if line == '' and process.poll() is not None:
                    break
                if line:
                    print(line, end='')
                    full_output.append(line)
                    sys.stdout.flush()
            
            return_code = process.wait()
            return return_code, ''.join(full_output)
        
        except Exception as e:
            self.log.error(f"启动训练进程失败: {e}")
            return -1, str(e)
    
    def _cleanup_and_restore(self, success):
        """清理并恢复原始配置"""
        print("\n" + "=" * 60)
        print("[版本控制] 执行清理和恢复操作")
        print("=" * 60)
        
        if success:
            print("[版本控制] 训练成功，生成变更记录...")
            self.version_controller.generate_change_record()
        else:
            print("[版本控制] 训练失败或终止，生成变更记录...")
            self.version_controller.generate_change_record()
        
        print("[版本控制] 恢复原始配置文件...")
        restore_success = self.version_controller.restore_original()
        
        if restore_success:
            print("[版本控制] ✅ 所有操作完成")
        else:
            print("[版本控制] ❌ 恢复失败，请手动检查")
    
    def run(self):
        """主运行逻辑"""
        self.log.info("\n" + "=" * 60)
        self.log.info("开始内存管理训练流程")
        self.log.info("=" * 60)
        
        try:
            while self.current_retry < self.max_retries:
                self.current_retry += 1
                
                self.log.info(f"\n--- 第 {self.current_retry} 次训练尝试 ---")
                
                return_code, output = self._launch_training()
                
                if self._detect_oom_error(output):
                    self.log.error("检测到 CUDA out of memory 错误！")
                    self.log.error(f"训练进程返回码: {return_code}")
                    
                    if self.current_retry < self.max_retries:
                        self.log.info("正在调整训练参数...")
                        success = self._adjust_parameter()
                        if not success:
                            self.log.error("无法继续调整参数，训练终止")
                            break
                        
                        time.sleep(2)
                        self.log.info("清理GPU缓存...")
                        self._clear_gpu_cache()
                    else:
                        self.log.error("已达到最大重试次数，训练终止")
                        break
                
                elif return_code == 0:
                    self.log.info("训练成功完成！")
                    self._save_adjustment_report()
                    self._cleanup_and_restore(success=True)
                    return True
                
                else:
                    self.log.error(f"训练异常终止，返回码: {return_code}")
                    self.log.error(f"输出信息: {output[-2000:] if len(output) > 2000 else output}")
                    
                    if self._detect_oom_error(output):
                        continue
                    else:
                        self.log.error("非内存错误导致训练失败，终止进程")
                        self._save_adjustment_report()
                        self._cleanup_and_restore(success=False)
                        return False
            
            self._save_adjustment_report()
            self._cleanup_and_restore(success=False)
            return False
            
        except Exception as e:
            self.log.error(f"内存管理脚本运行异常: {e}")
            import traceback
            self.log.error(f"异常堆栈: {traceback.format_exc()}")
            self._cleanup_and_restore(success=False)
            return False
    
    def _clear_gpu_cache(self):
        """清理GPU缓存"""
        try:
            subprocess.run(
                [sys.executable, '-c', 'import torch; torch.cuda.empty_cache()'],
                capture_output=True,
                timeout=10
            )
            self.log.info("GPU缓存已清理")
        except Exception as e:
            self.log.warning(f"清理GPU缓存失败: {e}")
    
    def _save_adjustment_report(self):
        """保存参数调整报告"""
        report_dir = os.path.join(self.project_root, 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        report_file = os.path.join(report_dir, f'adjustment_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# YOLOv26 内存管理参数调整报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**项目根目录**: {self.project_root}\n")
            f.write(f"**训练尝试次数**: {self.current_retry}\n")
            f.write(f"**GPU信息**: {self._get_gpu_info()}\n\n")
            
            f.write("## 原始配置\n")
            f.write("```yaml\n")
            yaml.dump(self.original_config, f, default_flow_style=False, allow_unicode=True)
            f.write("```\n\n")
            
            f.write("## 当前配置\n")
            f.write("```yaml\n")
            yaml.dump(self.current_config, f, default_flow_style=False, allow_unicode=True)
            f.write("```\n\n")
            
            f.write("## 参数调整历史\n")
            if self.adjustment_history:
                f.write("| 时间 | 参数名 | 原始值 | 新值 | 调整原因 | 优先级 | 重试次数 | 调整次数 |\n")
                f.write("|------|--------|--------|------|----------|--------|----------|----------|\n")
                for record in self.adjustment_history:
                    adj_count = record.get('adjustment_count', 1)
                    f.write(f"| {record['timestamp']} | {record['param_name']} | {record['original_value']} | {record['new_value']} | {record['adjustment_reason']} | {record['priority']} | {record['retry_count']} | {adj_count} |\n")
            else:
                f.write("无参数调整记录\n")
            
            f.write("\n## 参数调整优先级说明\n")
            f.write("| 优先级 | 参数名 | 调整方式 | 影响说明 |\n")
            f.write("|--------|--------|----------|----------|\n")
            for param in self.param_adjustment_order:
                f.write(f"| {param['priority']} | {param['name']} | {param['action']} | {param['desc']} |\n")
        
        self.log.info(f"参数调整报告已保存: {report_file}")
    
    def print_adjustment_summary(self):
        """打印参数调整摘要"""
        print("\n" + "=" * 60)
        print("参数调整摘要")
        print("=" * 60)
        
        if self.adjustment_history:
            print(f"\n共进行了 {len(self.adjustment_history)} 次参数调整")
            print("\n调整记录:")
            for i, record in enumerate(self.adjustment_history, 1):
                print(f"  {i}. {record['param_name']}: {record['original_value']} -> {record['new_value']}")
                print(f"     原因: {record['adjustment_reason']}")
        else:
            print("未进行任何参数调整")
        
        print("\n" + "=" * 60)


def main():
    """主函数"""
    PROJECT_ROOT = "/home/fst/zzy work/yolov26"
    
    manager = MemoryManager(PROJECT_ROOT)
    
    success = manager.run()
    
    manager.print_adjustment_summary()
    
    if success:
        print("\n✅ 训练成功完成！")
        sys.exit(0)
    else:
        print("\n❌ 训练失败，请查看日志文件了解详情")
        sys.exit(1)


if __name__ == '__main__':
    main()