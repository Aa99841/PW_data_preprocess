# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
from matplotlib.pyplot import hsv

def apply_clahe(gray):

    # 創建 CLAHE 物件
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    enhanced = clahe.apply(gray)

    return enhanced

def remove_green_red(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ========= 綠色 =========
    lower_green = np.array([40, 80, 80])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # ========= 紅色 (兩段) =========
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 80, 80])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # 合併綠色和紅色的 mask
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask_green, mask_red)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    result = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)

    return result

def resize_with_padding(img, size):

    h, w = img.shape[:2]

    # 計算縮放比例
    scale = size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    # resize
    resized = cv2.resize(img, (new_w, new_h))

    # padding
    pad_top = (size - new_h) // 2
    pad_bottom = size - new_h - pad_top
    pad_left = (size - new_w) // 2
    pad_right = size - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0
    )

    return padded

def resize(img, size):

    h, w = img.shape[:2]

    # 計算縮放比例
    scale = size / min(h, w)

    if scale > 1.0:
        scale = 1.0
    new_w = int(w * scale)
    new_h = int(h * scale)

    # resize
    resized = cv2.resize(img, (new_w, new_h))

    return resized

def center_crop(img, size):

    h, w = img.shape[:2]

    # size = min(h, w)

    start_x = (w - size) // 2
    start_y = (h - size) // 2

    crop = img[start_y:start_y+size, start_x:start_x+size]

    return crop

def save_and_classify_frame(frame_to_save, original_frame, frame_name, base_output_path):

    with_ui_dir = os.path.join(base_output_path, "with_ui")
    no_ui_dir = os.path.join(base_output_path, "no_ui")

    os.makedirs(with_ui_dir, exist_ok=True)
    os.makedirs(no_ui_dir, exist_ok=True)

    h, w = original_frame.shape[:2]

    # 只抓右上角區域
    roi = original_frame[0:int(h*0.15), int(w*0.75):w]

    if roi.size == 0:
        return

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # ========= 1. 偵測白色文字 =========
    _, white_mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    white_pixels = cv2.countNonZero(white_mask)

    # ========= 2. 偵測黑色背景 =========
    _, black_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    black_pixels = cv2.countNonZero(black_mask)

    # ========= 3. 找文字輪廓 =========
    contours, _ = cv2.findContours(
        white_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    text_like_contours = 0

    for cnt in contours:
        x,y,wc,hc = cv2.boundingRect(cnt)

        # 過濾太小的 noise
        if 5 < wc < 80 and 5 < hc < 40:
            text_like_contours += 1

    # ========= Debug =========
    # print(frame_name, white_pixels, black_pixels, text_like_contours)

    # ========= 判斷 UI =========
    if (
        white_pixels > 80 and
        black_pixels > 300 and
        text_like_contours >= 2
    ):
        save_path = os.path.join(with_ui_dir, frame_name)
        statue = "with_ui"
    else:
        save_path = os.path.join(no_ui_dir, frame_name)
        statue = "no_ui"

    cv2.imwrite(save_path, frame_to_save)
    return statue

def extract_frames_with_timestamp(video_path, output_dir):
    # 建立輸出資料夾
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 開啟影片檔案
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟影片: {video_path}")
        print("無法開啟影片")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)  # 每秒幀數
    frame_count = -1
    no_ui_count = 0
    
    while True:
        ret, frame = cap.read() # 讀取一幀
        if not ret:
            break  # 讀到結尾
        
        # 900 * 700
        x1, y1 = 80, 100
        x2, y2 = 980, 800
        
        # 650 * 650
        # x1, y1 = 80, 150
        # x2, y2 = 730, 800
        cropped = frame[y1:y2, x1:x2]
        
        frame_count += 1
        
        # 每 10 幀處理一次
        if frame_count % 25 != 0:
            continue

        # 計算當前幀對應的時間戳（秒）
        time_in_sec = frame_count / fps
        timestamp_str = f"frame_{time_in_sec:.2f}s.png"
        filename = os.path.join(output_dir, timestamp_str)
        
        result = remove_green_red(cropped)
        result = resize(result, 224)
        # result = resize_with_padding(result, 224)
        # result = resize_with_padding(result, 512)
        # result = center_crop(result, 224)
        # result = center_crop(result, 512)
        enhanced = apply_clahe(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))         

        # 儲存圖片
        # cv2.imwrite(filename, enhanced)
        statue = save_and_classify_frame(enhanced, cropped, timestamp_str, output_dir)
        # if statue == "no_ui":
        #     no_ui_count += 1

    cap.release()
    print("所有幀都已成功擷取完成！")
    print(f"size: {result.shape}")
    return no_ui_count
    
    
# --- 主程式 ---
total_no_ui_count = 0
# input_folder = "extracted_clips/data2/new"
# for filename in os.listdir(input_folder):
#     output_folder = "output/measure_remove_green/classify_test/data2/new/data2_" + filename.split(".")[0]
#     os.makedirs(output_folder, exist_ok=True)
#     video_path = os.path.join(input_folder, filename)
#     total_no_ui_count += extract_frames_with_timestamp(video_path, output_folder)

input_folder = "extracted_clips/data1"
for filename in os.listdir(input_folder):
    output_folder = "output/measure_remove_green/classify_test/data1/data1_" + filename.split(".")[0]
    os.makedirs(output_folder, exist_ok=True)
    video_path = os.path.join(input_folder, filename)
    total_no_ui_count += extract_frames_with_timestamp(video_path, output_folder)
    
input_folder = "extracted_clips/data2"
for filename in os.listdir(input_folder):
    output_folder = "output/measure_remove_green/classify_test/data2/data2_" + filename.split(".")[0]
    os.makedirs(output_folder, exist_ok=True)
    video_path = os.path.join(input_folder, filename)
    total_no_ui_count += extract_frames_with_timestamp(video_path, output_folder)
    
input_folder = "extracted_clips/data3"
for filename in os.listdir(input_folder):
    output_folder = "output/measure_remove_green/classify_test/data3/data3_" + filename.split(".")[0]
    os.makedirs(output_folder, exist_ok=True)
    video_path = os.path.join(input_folder, filename)
    total_no_ui_count += extract_frames_with_timestamp(video_path, output_folder)
    
input_folder = "extracted_clips/data4"
for filename in os.listdir(input_folder):
    output_folder = "output/measure_remove_green/classify_test/data4/data4_" + filename.split(".")[0]
    os.makedirs(output_folder, exist_ok=True)
    video_path = os.path.join(input_folder, filename)
    total_no_ui_count += extract_frames_with_timestamp(video_path, output_folder)
    
input_folder = "extracted_clips/data5"
for filename in os.listdir(input_folder):
    output_folder = "output/measure_remove_green/classify_test/data5/data5_" + filename.split(".")[0]
    os.makedirs(output_folder, exist_ok=True)
    video_path = os.path.join(input_folder, filename)
    total_no_ui_count += extract_frames_with_timestamp(video_path, output_folder)
    
input_folder = "extracted_clips/data6"
for filename in os.listdir(input_folder):
    output_folder = "output/measure_remove_green/classify_test/data6/data6_" + filename.split(".")[0]
    os.makedirs(output_folder, exist_ok=True)
    video_path = os.path.join(input_folder, filename)
    total_no_ui_count += extract_frames_with_timestamp(video_path, output_folder)
    
print(f"總共擷取了 {total_no_ui_count} 張無 UI 的圖片！")
