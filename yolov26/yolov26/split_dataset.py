# -*- coding: utf-8 -*-
# 将图片文件及标签文件按比例划分为训练集、验证集、测试集
import os
import shutil
import random

PROJECT_ROOT = "/home/fst/zzy work/yolov26"

label_file_dir = os.path.join(PROJECT_ROOT, 'dataset/labels_raw/')
images_file_dir = os.path.join(PROJECT_ROOT, 'dataset/images_raw/')

train_percent = 0.85
val_percent = 0.15
test_percent = 0

IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']


def create_dir():
    images_dir = os.path.join(PROJECT_ROOT, 'dataset/images/')
    labels_dir = os.path.join(PROJECT_ROOT, 'dataset/labels/')
    
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    for subdir in ['train', 'val', 'test']:
        if not os.path.exists(os.path.join(images_dir, subdir)):
            os.makedirs(os.path.join(images_dir, subdir))
        if not os.path.exists(os.path.join(labels_dir, subdir)):
            os.makedirs(os.path.join(labels_dir, subdir))


def get_image_extension(filename):
    for ext in IMAGE_EXTENSIONS:
        if filename.lower().endswith(ext):
            return ext
    return None


def main():
    create_dir()
    
    print("=" * 60)
    print("数据集划分诊断")
    print("=" * 60)
    
    all_images = []
    all_labels = []
    
    for file in os.listdir(images_file_dir):
        ext = get_image_extension(file)
        if ext:
            all_images.append(file)
    
    for file in os.listdir(label_file_dir):
        if file.lower().endswith('.txt'):
            all_labels.append(file)
    
    print(f"\n图片目录: {images_file_dir}")
    print(f"标签目录: {label_file_dir}")
    print(f"\n总的图片数目：{len(all_images)}")
    print(f"总的标签数目：{len(all_labels)}")
    
    if len(all_images) == 0:
        print("\n❌ 错误：图片目录中没有找到任何图片文件！")
        print(f"支持的格式: {IMAGE_EXTENSIONS}")
        return
    
    if len(all_labels) == 0:
        print("\n❌ 错误：标签目录中没有找到任何标签文件！")
        return
    
    label_basenames = set()
    for label_file in all_labels:
        basename = os.path.splitext(label_file)[0]
        label_basenames.add(basename)
    
    print(f"\n去重后的标签基名数目：{len(label_basenames)}")
    
    print("\n--- 检测图片格式分布 ---")
    format_counts = {}
    for img in all_images:
        ext = get_image_extension(img)
        format_counts[ext] = format_counts.get(ext, 0) + 1
    for ext, count in format_counts.items():
        print(f"  {ext}: {count} 张")
    
    print("\n--- 检测标签匹配情况 ---")
    valid_images = []
    matched_labels = []
    unmatched_images = []
    unmatched_labels = []
    
    label_set = set(label_basenames)
    
    for img_file in all_images:
        ext = get_image_extension(img_file)
        basename = os.path.splitext(img_file)[0]
        
        if basename in label_set:
            valid_images.append(img_file)
            matched_labels.append(basename + '.txt')
            label_set.remove(basename)
        else:
            unmatched_images.append(img_file)
    
    unmatched_labels = [lb + '.txt' for lb in label_set]
    
    print(f"\n✅ 有标签的图片：{len(valid_images)}")
    print(f"❌ 无标签的图片：{len(unmatched_images)}")
    print(f"❌ 无对应图片的标签：{len(unmatched_labels)}")
    
    if len(valid_images) == 0:
        print("\n" + "=" * 60)
        print("❌ 严重错误：没有找到任何匹配的图片和标签！")
        print("=" * 60)
        
        if unmatched_images and unmatched_labels:
            print("\n可能的原因：")
            print("1. 图片和标签的文件名前缀不一致")
            print("2. 大小写不匹配（如 image001.png vs Image001.txt）")
            print("3. 文件扩展名不匹配")
            
            print("\n--- 示例对比 ---")
            print("图片文件名示例：")
            for img in unmatched_images[:5]:
                print(f"  {img}")
            print("\n标签文件名示例：")
            for lbl in unmatched_labels[:5]:
                print(f"  {lbl}")
        
        return
    
    print("\n" + "=" * 60)
    print("开始数据集划分")
    print("=" * 60)
    
    total_valid = len(valid_images)
    num_train = int(total_valid * train_percent)
    num_val = int(total_valid * val_percent)
    num_test = total_valid - num_train - num_val
    
    print(f"\n训练集：{num_train} ({train_percent*100:.0f}%)")
    print(f"验证集：{num_val} ({val_percent*100:.0f}%)")
    print(f"测试集：{num_test} ({test_percent*100:.0f}%)")
    
    indices = list(range(total_valid))
    random.shuffle(indices)
    
    list_trainIndex = set(indices[:num_train])
    list_valIndex = set(indices[num_train:num_train + num_val])
    list_testIndex = set(indices[num_train + num_val:])
    
    train_count = 0
    val_count = 0
    test_count = 0
    
    images_dir = os.path.join(PROJECT_ROOT, 'dataset/images/')
    labels_dir = os.path.join(PROJECT_ROOT, 'dataset/labels/')
    
    for i, img_file in enumerate(valid_images):
        img_path = os.path.join(images_file_dir, img_file)
        basename = os.path.splitext(img_file)[0]
        label_file = basename + '.txt'
        label_path = os.path.join(label_file_dir, label_file)
        
        if i in list_trainIndex:
            dst_img = os.path.join(images_dir, 'train', img_file)
            dst_label = os.path.join(labels_dir, 'train', label_file)
            train_count += 1
        elif i in list_valIndex:
            dst_img = os.path.join(images_dir, 'val', img_file)
            dst_label = os.path.join(labels_dir, 'val', label_file)
            val_count += 1
        else:
            dst_img = os.path.join(images_dir, 'test', img_file)
            dst_label = os.path.join(labels_dir, 'test', label_file)
            test_count += 1
        
        shutil.copyfile(img_path, dst_img)
        shutil.copyfile(label_path, dst_label)
    
    print("\n" + "=" * 60)
    print("数据集划分完成！")
    print("=" * 60)
    print(f"\n训练集：{train_count}")
    print(f"验证集：{val_count}")
    print(f"测试集：{test_count}")
    print(f"\n无标签的图片（已跳过）：{len(unmatched_images)}")
    print(f"无对应图片的标签（已跳过）：{len(unmatched_labels)}")
    
    if unmatched_images:
        print("\n无标签图片列表（前10个）：")
        for img in unmatched_images[:10]:
            print(f"  {img}")
        if len(unmatched_images) > 10:
            print(f"  ... 还有 {len(unmatched_images) - 10} 个无标签图片")
    
    if unmatched_labels:
        print("\n无对应图片的标签列表（前10个）：")
        for lbl in unmatched_labels[:10]:
            print(f"  {lbl}")
        if len(unmatched_labels) > 10:
            print(f"  ... 还有 {len(unmatched_labels) - 10} 个无对应图片的标签")


if __name__ == '__main__':
    main()