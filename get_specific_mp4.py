import cv2
import os

# data2 [422, 607] => 13 : 3
# other => 7 : 1
def extract_clips(video_path, output_dir,seconds_list, window_sec=7):
    """
    video_path: 原始影片路徑
    seconds_list: 你提供的秒數清單
    window_sec: 截取的時間範圍（前後各幾秒）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("無法開啟影片")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 設定編碼器

    # 建立輸出資料夾
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for sec in seconds_list:
        # 計算開始與結束的 frame
        start_frame = max(0, int((sec - window_sec) * fps))
        end_frame = int((sec + 1) * fps)
        
        # 跳轉到開始位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        output_name = os.path.join(output_dir, f"clip_{sec}s.mp4")
        out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
        
        print(f"正在擷取 {sec}s 附近的片段...")
        
        for i in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            
        out.release()
        print(f"完成：{output_name}")

    cap.release()
    print("所有片段擷取完畢！")

# ====================== 執行 ===========================
my_seconds = [141, 163, 248, 294, 352, 423, 504, 560, 617, 643, 716, 756, 793, 818]
extract_clips('data/data1.mp4','extracted_clips/data1', my_seconds)    

my_seconds = [147,185,216,257,345,422,481,492,535,571,599,607,637,676,696,722,750]
extract_clips('data/data2.mp4', 'extracted_clips/data2', my_seconds)   

# my_seconds = [422, 607]
# extract_clips('data/data2.mp4', 'extracted_clips/data2/new', my_seconds) 

my_seconds = [31,36,98,154,177,289,312,358,394,405]
extract_clips('data/data3.mp4', 'extracted_clips/data3', my_seconds) 
