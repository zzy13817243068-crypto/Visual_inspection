# -*- coding: utf-8 -*-
# ============================================
# 数据集清洗脚本
# 功能: 识别并删除无标签图片，保留所有标签，将清洗后的数据保存到test文件夹
# 基于 split_dataset.py 改造
# ============================================

import os
import shutil
from datetime import datetime

# 项目根目录（Linux路径，部署到服务器后自动适配）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 原始数据目录
IMAGES_RAW_DIR = os.path.join(PROJECT_ROOT, 'dataset/images_raw/')
LABELS_RAW_DIR = os.path.join(PROJECT_ROOT, 'dataset/labels_raw/')
# 清洗后数据保存目录（test文件夹）
TEST_DIR = os.path.join(PROJECT_ROOT, 'clean/')
CLEAN_IMAGES_DIR = os.path.join(TEST_DIR, 'clean/images_raw/')
CLEAN_LABELS_DIR = os.path.join(TEST_DIR, 'clean/labels_raw/')

# 支持的图片格式
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']


def get_image_extension(filename):
    """获取图片文件的扩展名（不区分大小写）"""
    for ext in IMAGE_EXTENSIONS:
        if filename.lower().endswith(ext):
            return ext
    return None


def create_output_dirs():
    """创建输出目录结构"""
    os.makedirs(CLEAN_IMAGES_DIR, exist_ok=True)
    os.makedirs(CLEAN_LABELS_DIR, exist_ok=True)
    print(f"已创建输出目录:")
    print(f"  图片: {CLEAN_IMAGES_DIR}")
    print(f"  标签: {CLEAN_LABELS_DIR}")


def scan_files():
    """
    扫描原始数据目录，收集所有图片和标签文件
    
    返回:
        all_images: 所有图片文件名列表
        all_labels: 所有标签文件名列表
    """
    all_images = []
    all_labels = []

    if os.path.exists(IMAGES_RAW_DIR):
        for file in os.listdir(IMAGES_RAW_DIR):
            if get_image_extension(file):
                all_images.append(file)

    if os.path.exists(LABELS_RAW_DIR):
        for file in os.listdir(LABELS_RAW_DIR):
            if file.lower().endswith('.txt'):
                all_labels.append(file)

    return all_images, all_labels


def match_files(all_images, all_labels):
    """
    匹配图片和标签文件
    
    返回:
        matched_pairs: 匹配成功的 (图片文件名, 标签文件名) 列表
        orphan_images: 无标签的图片文件名列表
        orphan_labels: 无对应图片的标签文件名列表
    """
    # 构建标签基名集合（去掉.txt后缀）
    label_basenames = {}
    for label_file in all_labels:
        basename = os.path.splitext(label_file)[0]
        label_basenames[basename] = label_file

    # 构建图片基名集合（去掉扩展名）
    image_basenames = {}
    for img_file in all_images:
        basename = os.path.splitext(img_file)[0]
        image_basenames[basename] = img_file

    matched_pairs = []
    orphan_images = []
    orphan_labels = []

    # 匹配图片
    matched_label_basenames = set()
    for basename, img_file in image_basenames.items():
        if basename in label_basenames:
            matched_pairs.append((img_file, label_basenames[basename]))
            matched_label_basenames.add(basename)
        else:
            orphan_images.append(img_file)

    # 找出无对应图片的标签
    for basename, label_file in label_basenames.items():
        if basename not in matched_label_basenames:
            orphan_labels.append(label_file)

    return matched_pairs, orphan_images, orphan_labels


def save_clean_data(matched_pairs, all_labels):
    """
    将清洗后的数据复制到test文件夹
    图片: 只保存有对应标签的图片（删除无标签图片）
    标签: 保存所有标签（保留全部，不删除任何标签）
    
    返回:
        saved_image_count: 成功保存的图片数量
        saved_label_count: 成功保存的标签数量
    """
    saved_image_count = 0
    # 保存匹配的图片（有对应标签的图片）
    for img_file, label_file in matched_pairs:
        src_img = os.path.join(IMAGES_RAW_DIR, img_file)
        dst_img = os.path.join(CLEAN_IMAGES_DIR, img_file)
        shutil.copyfile(src_img, dst_img)
        saved_image_count += 1

    # 保存所有标签（保留全部标签，不删除无图片对应的标签）
    saved_label_count = 0
    for label_file in all_labels:
        src_label = os.path.join(LABELS_RAW_DIR, label_file)
        dst_label = os.path.join(CLEAN_LABELS_DIR, label_file)
        shutil.copyfile(src_label, dst_label)
        saved_label_count += 1

    return saved_image_count, saved_label_count


def generate_report(all_images, all_labels, matched_pairs, orphan_images,
                    orphan_labels, saved_image_count, saved_label_count):
    """生成并打印详细的处理报告，同时保存到txt文件"""
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("数据集清洗处理报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)

    report_lines.append("")
    report_lines.append("一、原始数据统计")
    report_lines.append("-" * 40)
    report_lines.append(f"图片目录: {IMAGES_RAW_DIR}")
    report_lines.append(f"标签目录: {LABELS_RAW_DIR}")
    report_lines.append(f"原始图片总数: {len(all_images)}")
    report_lines.append(f"原始标签总数: {len(all_labels)}")

    # 图片格式分布
    report_lines.append("")
    report_lines.append("二、图片格式分布")
    report_lines.append("-" * 40)
    format_counts = {}
    for img in all_images:
        ext = get_image_extension(img)
        format_counts[ext] = format_counts.get(ext, 0) + 1
    for ext, count in format_counts.items():
        report_lines.append(f"  {ext}: {count} 张")

    report_lines.append("")
    report_lines.append("三、匹配结果")
    report_lines.append("-" * 40)
    report_lines.append(f"✅ 成功匹配（图片+标签）: {len(matched_pairs)} 对")
    report_lines.append(f"❌ 无标签的图片: {len(orphan_images)} 个")
    report_lines.append(f"ℹ️ 无对应图片的标签（已保留）: {len(orphan_labels)} 个")

    report_lines.append("")
    report_lines.append("四、处理操作")
    report_lines.append("-" * 40)
    report_lines.append(f"已删除（排除）无标签图片: {len(orphan_images)} 个")
    report_lines.append(f"已保留所有标签（不删除）: {saved_label_count} 个")
    report_lines.append(f"已保存的图片: {saved_image_count} 个")

    report_lines.append("")
    report_lines.append("五、保存路径")
    report_lines.append("-" * 40)
    report_lines.append(f"清洗后图片保存目录: {CLEAN_IMAGES_DIR}")
    report_lines.append(f"清洗后标签保存目录: {CLEAN_LABELS_DIR}")

    # 列出被删除的文件
    if orphan_images:
        report_lines.append("")
        report_lines.append("六、被删除的无标签图片列表")
        report_lines.append("-" * 40)
        for i, img in enumerate(orphan_images):
            report_lines.append(f"  {i + 1}. {img}")
        if len(orphan_images) > 20:
            report_lines.append(f"  ... 共 {len(orphan_images)} 个")

    if orphan_labels:
        report_lines.append("")
        report_lines.append("七、无对应图片的标签列表（已保留，未删除）")
        report_lines.append("-" * 40)
        for i, lbl in enumerate(orphan_labels):
            report_lines.append(f"  {i + 1}. {lbl}")
        if len(orphan_labels) > 20:
            report_lines.append(f"  ... 共 {len(orphan_labels)} 个")

    # 保留的文件列表
    if matched_pairs:
        report_lines.append("")
        report_lines.append("八、保留的文件列表（前20个）")
        report_lines.append("-" * 40)
        for i, (img, lbl) in enumerate(matched_pairs[:20]):
            report_lines.append(f"  {i + 1}. {img} <-> {lbl}")
        if len(matched_pairs) > 20:
            report_lines.append(f"  ... 共 {len(matched_pairs)} 对")

    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("清洗完成！")
    report_lines.append("=" * 60)

    # 打印报告
    report_text = "\n".join(report_lines)
    print(report_text)

    # 保存报告到文件
    report_file = os.path.join(TEST_DIR, f'clean_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n报告已保存到: {report_file}")

    return report_file


def main():
    """主函数：执行数据集清洗流程"""
    print("=" * 60)
    print("数据集清洗脚本")
    print("功能: 删除无标签图片，保留所有标签，保存清洗后数据到test文件夹")
    print("=" * 60)

    # 步骤1：创建输出目录
    print("\n[步骤1] 创建输出目录...")
    create_output_dirs()

    # 步骤2：扫描原始文件
    print("\n[步骤2] 扫描原始数据...")
    all_images, all_labels = scan_files()

    print(f"  找到图片: {len(all_images)} 个")
    print(f"  找到标签: {len(all_labels)} 个")

    if len(all_images) == 0:
        print("\n❌ 错误：图片目录中没有找到任何图片文件！")
        print(f"  图片目录: {IMAGES_RAW_DIR}")
        print(f"  支持的格式: {IMAGE_EXTENSIONS}")
        return

    if len(all_labels) == 0:
        print("\n❌ 错误：标签目录中没有找到任何标签文件！")
        print(f"  标签目录: {LABELS_RAW_DIR}")
        return

    # 步骤3：匹配文件
    print("\n[步骤3] 匹配图片和标签...")
    matched_pairs, orphan_images, orphan_labels = match_files(all_images, all_labels)

    print(f"  ✅ 匹配成功: {len(matched_pairs)} 对")
    print(f"  ❌ 无标签图片（将删除）: {len(orphan_images)} 个")
    print(f"  ℹ️ 无图片标签（将保留）: {len(orphan_labels)} 个")

    if len(matched_pairs) == 0:
        print("\n❌ 严重错误：没有找到任何匹配的图片和标签！")
        print("\n可能的原因：")
        print("  1. 图片和标签的文件名前缀不一致")
        print("  2. 大小写不匹配（如 image001.jpg vs Image001.txt）")
        print("\n--- 示例对比 ---")
        print("图片文件名示例：")
        for img in all_images[:5]:
            print(f"  {img}")
        print("标签文件名示例：")
        for lbl in all_labels[:5]:
            print(f"  {lbl}")
        return

    # 步骤4：保存清洗后的数据
    print("\n[步骤4] 保存清洗后的数据到test文件夹...")
    saved_image_count, saved_label_count = save_clean_data(matched_pairs, all_labels)
    print(f"  已保存图片: {saved_image_count} 个")
    print(f"  已保存标签: {saved_label_count} 个（全部保留）")

    # 步骤5：生成处理报告
    print("\n[步骤5] 生成处理报告...")
    generate_report(all_images, all_labels, matched_pairs, orphan_images,
                    orphan_labels, saved_image_count, saved_label_count)


if __name__ == '__main__':
    main()
