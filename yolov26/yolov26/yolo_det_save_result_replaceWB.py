import shutil
import cv2
import os
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import math
import logging
import random
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 全局变量
img_count = 0
frame_idx = 0
consecutive_failures = 0
max_consecutive_failures = 10

# ===================== 【已移除】原固定裁剪约束 =====================
# 现改为视频动态裁剪：宽度=视频宽，高度在 [视频高/2, 视频高] 随机，Y随机
# ================================================================

# 水平随机裁剪开关（水平固定，开关无效，保留但不再使用）
enable_random_crop = True  # 该开关已不再影响裁剪逻辑（宽度固定）

def mk_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def get_subfolder_name(img_count, max_per_folder=10000):
    folder_index = math.floor(img_count / max_per_folder)
    return f"batch_{folder_index:03d}"

def yolo_detection_and_save(
        image,
        output_img_dir,
        output_label_dir,
        model,
        save_result=False,
        save_txt=False,
        resave_source_img=False,
        show_result=False,
        max_per_folder=10000,
        is_video_frame=False,
        frame_idx=None,
        crop_config_name=None,
        full_frame=None,
        full_frame_size=None
):
    global img_count

    if is_video_frame:
        crop_img = image
        if crop_img is None or crop_img.size == 0:
            logger.warning(f"空裁剪帧，跳过 frame_idx={frame_idx}")
            return
        filename = f"frame_{frame_idx:08d}_{crop_config_name}"
        save_img = crop_img
        infer_img = crop_img
    else:
        try:
            img_path = Path(image).resolve()
            if not img_path.exists():
                logger.error(f"图片不存在: {img_path}")
                return
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                logger.error(f"读取图片失败: {img_path}")
                return
            filename = img_path.stem
            save_img = img
            infer_img = img
        except Exception as e:
            logger.error(f"处理图片异常 {image}: {str(e)}")
            return

    # 模型推理
    try:
        results = model.predict(source=infer_img, conf=0.5, verbose=False)
    except Exception as e:
        logger.error(f"模型推理失败 {filename}: {str(e)}")
        return

    # 检测结果判断
    has_detection = False
    car_count = 0
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            has_detection = True
            car_count = len(result.boxes)

    # 保存效果图
    if save_result:
        try:
            output_path = os.path.join(output_img_dir, f"{filename}_det.jpg")
            annotated = results[0].plot()
            cv2.imwrite(output_path, annotated)
        except Exception as e:
            logger.warning(f"保存效果图失败 {output_path}: {str(e)}")

    # 保存图片（无dict分级，统一batch保存）
    need_save_txt = save_txt
    current_txt_name = filename
    current_subfolder = ""

    if resave_source_img:
        try:
            subfolder = get_subfolder_name(img_count, max_per_folder)
            current_subfolder = subfolder

            if has_detection:
                base_dir = os.path.join(output_img_dir, "detected")
                new_name = f"{img_count}_car_{car_count}_{crop_config_name}.jpg"
                src_dir = mk_dir(os.path.join(base_dir, subfolder))
                current_txt_name = os.path.splitext(new_name)[0]
                save_path = os.path.join(src_dir, new_name)
                cv2.imwrite(save_path, save_img)
            else:
                base_dir = os.path.join(output_img_dir, "no_detected")
                new_name = f"{img_count}_no_car_{crop_config_name}.jpg"
                src_dir = mk_dir(os.path.join(base_dir, subfolder))
                current_txt_name = os.path.splitext(new_name)[0]
                save_path = os.path.join(src_dir, new_name)
                cv2.imwrite(save_path, save_img)

            img_count += 1
        except Exception as e:
            logger.error(f"保存文件失败 {filename}: {str(e)}")
            need_save_txt = False

    # 保存标签（统一保存）
    if need_save_txt and has_detection:
        try:
            label_dir = mk_dir(os.path.join(output_label_dir, "detected", current_subfolder))
            txt_path = os.path.join(label_dir, f"{current_txt_name}.txt")
            orig_h, orig_w = save_img.shape[:2]

            with open(txt_path, "w") as f:
                for box in results[0].boxes:
                    class_id = 0
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2 / orig_w
                    cy = (y1 + y2) / 2 / orig_h
                    w = (x2 - x1) / orig_w
                    h = (y2 - y1) / orig_h
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        except Exception as e:
            logger.error(f"保存标签失败 {txt_path}: {str(e)}")

    # 显示结果
    if show_result:
        try:
            show_img = results[0].plot()
            cv2.imshow(f"Detection Result", show_img)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                return
        except Exception as e:
            logger.warning(f"显示结果失败: {str(e)}")


def process_images_batch(image_paths, output_img_dir, output_label_dir, model,
                         save_result=False, save_txt=False, resave_source_img=False,
                         show_result=False, batch_size=1, max_per_folder=10000):
    # 创建基础目录
    mk_dir(os.path.join(output_img_dir, "detected"))
    mk_dir(os.path.join(output_img_dir, "no_detected"))
    mk_dir(os.path.join(output_label_dir, "detected"))

    # 过滤有效图片
    valid_imgs = []
    for p in image_paths:
        if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and os.path.exists(p):
            valid_imgs.append(p)
    logger.info(f"待处理图片数: {len(valid_imgs)}")

    for i in tqdm(range(0, len(valid_imgs), batch_size), desc="处理图片", unit="张", mininterval=1.0, smoothing=0.05):
        for img_path in valid_imgs[i:i + batch_size]:
            yolo_detection_and_save(
                image=img_path,
                output_img_dir=output_img_dir,
                output_label_dir=output_label_dir,
                model=model,
                save_result=save_result,
                save_txt=save_txt,
                resave_source_img=resave_source_img,
                show_result=show_result,
                max_per_folder=max_per_folder,
                is_video_frame=False,
                crop_config_name="default"
            )


def process_video(video_path, output_img_dir, output_label_dir, model,
                  save_result=False, save_txt=False, resave_source_img=False,
                  show_result=False, max_per_folder=10000, frame_interval=1):
    global frame_idx, consecutive_failures

    if not os.path.exists(video_path):
        logger.error(f"视频不存在: {video_path}")
        return False

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            return False
    except Exception as e:
        logger.error(f"打开视频失败: {str(e)}")
        return False

    # 获取视频信息
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"视频分辨率: {video_w}x{video_h} | 总帧数: {total_frames}")

    # ===================== 【修改】动态生成裁剪参数 =====================
    # 裁剪宽度固定为视频宽度
    crop_width = video_w
    # 裁剪高度在 [video_h/2, video_h] 随机
    min_crop_height = int(video_h * 0.7)
    # min_crop_height = int(video_h)
    max_crop_height = video_h
    if max_crop_height > min_crop_height:
        crop_height = random.randint(min_crop_height, max_crop_height)
    else:
        crop_height = max_crop_height  # 视频高度为1或0时，取全高

    # Y方向随机（确保不超出）
    y1 = random.randint(0, video_h - crop_height) if video_h - crop_height >= 0 else 0
    x1 = 0
    x2 = video_w
    y2 = y1 + crop_height

    # 安全校验（理论上不会越界，但保留）
    x1 = max(0, x1)
    x2 = min(video_w, x2)
    y1 = max(0, y1)
    y2 = min(video_h, y2)

    # 兜底：若区域无效则使用整帧
    if x1 >= x2 or y1 >= y2:
        logger.warning(f"裁剪区域无效，使用整个视频画面 | 视频尺寸: {video_w}x{video_h}")
        x1, y1, x2, y2 = 0, 0, video_w, video_h
        crop_height = video_h
        crop_width = video_w

    crop_config_name = f"fullw_h{crop_height}_y{y1}"
    logger.info(
        f"动态裁剪参数 | 最终裁剪区域: [{x1}, {y1}, {x2}, {y2}] "
        f"| 裁剪尺寸: {x2 - x1}x{y2 - y1}"
    )
    # ==================================================================

    # 创建基础目录
    mk_dir(os.path.join(output_img_dir, "detected"))
    mk_dir(os.path.join(output_img_dir, "no_detected"))
    mk_dir(os.path.join(output_label_dir, "detected"))

    frame_idx = 0
    consecutive_failures = 0
    pbar = tqdm(total=total_frames, desc=f"处理视频 {Path(video_path).name}", unit="帧", mininterval=1.0)

    while True:
        ret, full_frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                break
            frame_idx += 1
            pbar.update(1)
            continue
        consecutive_failures = 0

        if frame_idx % frame_interval == 0:
            try:
                # 裁剪并处理
                crop_frame = full_frame[y1:y2, x1:x2]
                yolo_detection_and_save(
                    image=crop_frame,
                    output_img_dir=output_img_dir,
                    output_label_dir=output_label_dir,
                    model=model,
                    save_result=save_result,
                    save_txt=save_txt,
                    resave_source_img=resave_source_img,
                    show_result=show_result,
                    max_per_folder=max_per_folder,
                    is_video_frame=True,
                    frame_idx=frame_idx,
                    crop_config_name=crop_config_name
                )
            except Exception as e:
                logger.error(f"处理帧 {frame_idx} 失败: {str(e)}")

        frame_idx += 1
        pbar.update(1)
        if frame_idx >= total_frames:
            break

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"视频处理完成 | 裁剪配置: {crop_config_name}")
    return True


if __name__ == "__main__":
    # 配置参数
    show_result = False
    save_result = False
    save_txt = False
    resave_source_img = True
    max_per_folder = 5000
    frame_interval = 1

    # 路径配置（基于脚本所在目录的相对路径，跨平台兼容）
    BASE_DIR = Path(__file__).resolve().parent
    input_path = str(BASE_DIR / 'yolo_det_save_result_replaceWB')
    output_img_dir = str(BASE_DIR / 'result')
    output_label_dir = str(BASE_DIR / 'datasets' / '2026-2027-replace_WB' / 'labels')
    model_path = str(BASE_DIR / 'output' / 'exp-det-12' / 'weights' / 'best.pt')

    # 初始化
    img_count = 0
    frame_idx = 0

    if not os.path.exists(input_path):
        logger.error(f"输入路径不存在: {input_path}")
        exit(1)

    # 加载模型（全局仅一次）
    try:
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            exit(1)
        model = YOLO(model_path)
        logger.info(f"模型加载成功: {model_path}")
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        exit(1)

    if os.path.isdir(input_path):
        all_files = []
        video_files = []
        for root, _, files in os.walk(input_path):
            for file in files:
                fp = os.path.join(root, file)
                all_files.append(fp)
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
                    video_files.append(fp)

        logger.info(f"发现视频文件数: {len(video_files)}")

        # 处理图片（不变，仍使用原图）
        logger.info("=== 开始处理图片 ===")
        process_images_batch(
            image_paths=all_files,
            output_img_dir=output_img_dir,
            output_label_dir=output_label_dir,
            model=model,
            save_result=save_result,
            save_txt=save_txt,
            resave_source_img=resave_source_img,
            show_result=show_result,
            max_per_folder=max_per_folder
        )

        # 处理视频 - 随机抽取直到全部完成
        logger.info("=== 开始处理视频（随机抽取模式） ===")
        remaining_videos = video_files.copy()
        total_videos = len(remaining_videos)
        processed_count = 0

        while remaining_videos:
            random_idx = random.randint(0, len(remaining_videos) - 1)
            current_video = remaining_videos.pop(random_idx)
            processed_count += 1

            logger.info(f"\n【随机抽取】处理第 {processed_count}/{total_videos} 个视频")
            logger.info(f"当前视频路径: {current_video}")

            process_video(
                video_path=current_video,
                output_img_dir=output_img_dir,
                output_label_dir=output_label_dir,
                model=model,
                save_result=save_result,
                save_txt=save_txt,
                resave_source_img=resave_source_img,
                show_result=show_result,
                max_per_folder=max_per_folder,
                frame_interval=frame_interval
            )

    logger.info("=== 所有任务处理完成 ===")