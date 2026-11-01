#!/bin/bash
# ============================================
# YOLOv26 内存管理训练启动脚本
# 功能: 使用内存管理辅助脚本启动训练，自动处理CUDA OOM错误
# ============================================

PROJECT_DIR="/home/fst/zzy work/yolov26"
VENV_DIR="$PROJECT_DIR/venv"

echo "============================================"
echo "YOLOv26 内存管理训练启动脚本"
echo "============================================"
echo "项目目录: $PROJECT_DIR"
echo ""

# 检查虚拟环境
if [ -d "$VENV_DIR" ]; then
    echo "激活虚拟环境..."
    source "$VENV_DIR/bin/activate"
else
    echo "警告: 未找到虚拟环境，请先创建"
    echo "创建命令: python3.10 -m venv venv"
fi

echo ""
echo "启动内存管理训练..."
echo ""

# 启动内存管理脚本
python "$PROJECT_DIR/memory_manager.py"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "============================================"
    echo "✅ 训练成功完成！"
    echo "============================================"
    exit 0
else
    echo "============================================"
    echo "❌ 训练失败，请查看日志文件了解详情"
    echo "日志文件: $PROJECT_DIR/logs/"
    echo "报告文件: $PROJECT_DIR/reports/"
    echo "============================================"
    exit $EXIT_CODE
fi