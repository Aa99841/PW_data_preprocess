import math
import cv2
import os
import numpy as np
from skimage.morphology import skeletonize

def resize(img, size):

    h, w = img.shape[:2]

    # 計算縮放比例
    scale = size / min(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    # resize
    resized = cv2.resize(img, (new_w, new_h))

    return resized

def crop(img):

    x1, y1 = 0, 100
    x2, y2 = 900, 800
    
    cropped = img[y1:y2, x1:x2]

    return cropped

def detected_line(img, img_ori, filename):

    # ====== 建立彩色範圍 ======
    lower_green = np.array([30, 130, 30])
    upper_green = np.array([95, 255, 255])

    # ====== 抓取符合色彩範圍內的像素成為新影像 ======
    mask = cv2.inRange(img, lower_green, upper_green) 

    mask = cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # ====== 形態學處理 ====== 
    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)

    # ====== 偵測所有線段 ====== 
    # rho=1, theta=1度, threshold=20 (因為片段可能很短，門檻設低一點)
    lines = cv2.HoughLinesP(
        mask_dilated, 
        rho=1, 
        theta=np.pi/180, 
        threshold=20, 
        minLineLength=10, 
        maxLineGap=50  # 這裡設大一點，讓 OpenCV 嘗試自己連線
    )

    if lines is not None:
        all_points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            all_points.append((x1, y1))
            all_points.append((x2, y2))

        #  ====== 關鍵步驟：尋找「最遠兩端點」：找 Y 座標最小（最上方）與最大（最下方）的點 ====== 
        p_top = min(all_points, key=lambda p: p[1])
        p_bottom = max(all_points, key=lambda p: p[1])

        # ====== 計算長直線資訊 ====== 
        x1, y1 = p_top
        x2, y2 = p_bottom
        
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        # ====== 繪製結果 ====== 
        result_img = np.zeros_like(img_ori)
        cv2.line(result_img, p_top, p_bottom, (0, 0, 255), 2) 

        # print(f"偵測完成！")
        # print(f"頂點: {p_top}, 底點: {p_bottom}")
        # print(f"連線總長度: {length:.2f}")
        # print(f"連線角度: {angle:.2f}")

        return result_img
    else:
        print(f"{filename}未能偵測到線段，請檢查 mask 是否有內容。")
        
    return img
    

def floder_deal(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ======= 遍歷資料夾 =======
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    print(f"找到 {len(image_files)} 張 Mask 圖片，開始對應原圖處理...")

    for filename in image_files:
        # ======= 讀取圖像 ======= 
        mask_path = os.path.join(input_dir, filename)
        img_mask = cv2.imread(mask_path)
        
        # # ======= 讀取血管圖像 ======= 
        ori_filename = filename.replace("_frame", "") 
        vis_img = detected_line(img_mask, img_mask, filename)

        # ======= 儲存結果 =======
        save_path = os.path.join(output_dir, ori_filename)
        cv2.imwrite(save_path, vis_img)


# ======= 設定路徑 =======
input_dir_ori = r"C:\collega\Project\post_precessor\images"

input_dir = 'image_green/data1'
output_dir =  'line/data1'
floder_deal(input_dir, output_dir)
print(f"--- 所有檔案處理完畢！結果儲存於 {output_dir} ---")

input_dir = 'image_green/data2'
output_dir =  'line/data2'
floder_deal(input_dir, output_dir)
print(f"--- 所有檔案處理完畢！結果儲存於 {output_dir} ---")

input_dir = 'image_green/data3'
output_dir =  'line/data3'
floder_deal(input_dir, output_dir)
print(f"--- 所有檔案處理完畢！結果儲存於 {output_dir} ---")

input_dir = 'image_green/data4'
output_dir =  'line/data4'
floder_deal(input_dir, output_dir)
print(f"--- 所有檔案處理完畢！結果儲存於 {output_dir} ---")

input_dir = 'image_green/data5'
output_dir =  'line/data5'
floder_deal(input_dir, output_dir)
print(f"--- 所有檔案處理完畢！結果儲存於 {output_dir} ---")

input_dir = 'image_green/data6'
output_dir =  'line/data6'
floder_deal(input_dir, output_dir)
print(f"--- 所有檔案處理完畢！結果儲存於 {output_dir} ---")
