#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目录初始化脚本
创建 detect.py 和 train.py 所需的所有目录
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

directories = [
    os.path.join(PROJECT_ROOT, 'pretrained_models'),
    os.path.join(PROJECT_ROOT, 'output'),
    os.path.join(PROJECT_ROOT, 'runs', 'detect'),
    os.path.join(PROJECT_ROOT, 'runs', 'train'),
    os.path.join(PROJECT_ROOT, 'runs', 'val'),
    os.path.join(PROJECT_ROOT, 'dataset', 'images', 'train'),
    os.path.join(PROJECT_ROOT, 'dataset', 'images', 'val'),
    os.path.join(PROJECT_ROOT, 'dataset', 'images', 'test'),
    os.path.join(PROJECT_ROOT, 'dataset', 'labels', 'train'),
    os.path.join(PROJECT_ROOT, 'dataset', 'labels', 'val'),
    os.path.join(PROJECT_ROOT, 'dataset', 'labels', 'test'),
    os.path.join(PROJECT_ROOT, 'dataset', 'images_raw'),
    os.path.join(PROJECT_ROOT, 'dataset', 'labels_raw'),
]

print("=" * 60)
print("目录初始化脚本")
print("=" * 60)

created_count = 0
existing_count = 0

for dir_path in directories:
    if os.path.exists(dir_path):
        print(f"[已存在] {dir_path}")
        existing_count += 1
    else:
        os.makedirs(dir_path)
        print(f"[已创建] {dir_path}")
        created_count += 1

print("=" * 60)
print(f"完成！已创建: {created_count} 个目录，已存在: {existing_count} 个目录")
print("=" * 60)