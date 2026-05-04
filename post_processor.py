import math
import cv2
import os
import time
import numpy as np
from pathlib import Path
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from skimage.feature import hessian_matrix
from skimage.morphology import medial_axis
from scipy.ndimage import distance_transform_edt, label
from scipy.interpolate import splprep, splev

def resize(img, size):

    h, w = img.shape[:2]

    # 計算縮放比例
    scale = size / min(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    # resize
    resized = cv2.resize(img, (new_w, new_h))

    return resized

def crop(img, x1, x2, y1, y2):

    # x1, y1 = 0, 100
    # x2, y2 = 900, 800
    
    cropped = img[y1:y2, x1:x2]

    return cropped

def padding_Replication(img):    
    padded = cv2.copyMakeBorder(
        img,
        3, 3, 3, 3,
        cv2.BORDER_REPLICATE,
        value=0
    )
    
    return padded

def pruning_skeleton(skeleton_cv):
    if not np.any(skeleton_cv > 0):
        return skeleton_cv

    try:
        # 1. 將二值化影像轉化為一個圖數據結構，識別端點、分岔點
        skel = Skeleton(skeleton_cv > 0)
        
        # 2. 列出圖中每一條「分支」的資訊
        stats = summarize(skel, separator='_') 
        if stats.empty:
            return skeleton_cv
        
        pruned_skeleton = skeleton_cv.copy()
        
        junctions = stats[stats['branch_type'] == 2] # 2 代表 節點到節點 (通常主幹中間)
        tips = stats[stats['branch_type'] == 1]      # 1 代表 端點到節點 (分岔)

        # 找到較短分支剪掉
        if not tips.empty:
            tips_sorted = tips.sort_values(by='branch_distance', ascending=False) # 將所有端點分支按長度由長到短排序
            
            to_prune = tips_sorted.iloc[2:] # 將第三個頭（分岔）剪掉
            # to_prune += tips[tips['branch_distance'] < 50]

            for idx in to_prune.index:
                coords = skel.path_coordinates(idx) # 取得該分支所有像素的座標
                for r, c in coords.astype(int):
                    pruned_skeleton[r, c] = 0 
                    
        return pruned_skeleton

    except Exception as e:
        print(f"剪枝失敗: {e}")
        return skeleton_cv
    
def pruning_skeleton_small(skeleton_cv):
    skeleton_bool = (skeleton_cv > 0)

    skel = Skeleton(skeleton_bool)
    stats = summarize(skel, separator='_')  

    pruned_skeleton = skeleton_cv.copy()
    height, width = skeleton_cv.shape

    for _ in range(3): 
        skel = Skeleton(pruned_skeleton > 0)
        stats = summarize(skel, separator='_')
        
        for i in range(len(stats)):
            # 這裡記得用你剛修正過的底線符號
            branch_type = stats.loc[i, 'branch_type']
            branch_dist = stats.loc[i, 'branch_distance']
            
            if branch_type == 1:  # 端點到分岔點
                coords = skel.path_coordinates(i)
                endpoint = coords[-1]
                
                # 邊界保護邏輯
                is_near_border = (endpoint[1] < 5) or (endpoint[1] > width - 5)
                
                # 如果不是邊界附近的短毛刺，就把它抹除
                if branch_dist < 30 and not is_near_border:
                    for r, c in coords.astype(int):
                        pruned_skeleton[r, c] = 0
    return pruned_skeleton

def detected_line(img):

    # ====== 建立彩色範圍 ======
    lower_green = np.array([0, 0, 200])
    upper_green = np.array([0, 0, 255])

    # ====== 抓取符合色彩範圍內的像素成為新影像 ======
    mask = cv2.inRange(img, lower_green, upper_green) 
    
    green_points = np.where(mask == 255)
    
    if len(green_points[0]) > 0:
        # 將座標轉換成 (x, y) 的列表
        all_points = list(zip(green_points[1], green_points[0]))
        
        # 找出最上方的點（Y最小）和最下方的點（Y最大）
        p_top = min(all_points, key=lambda p: p[1])
        p_bottom = max(all_points, key=lambda p: p[1])
        
        return p_top, p_bottom
    
    return None, None

def separate_white_regions_advanced(image, min_area=100, 
                                     distance_threshold=10,
                                     save_separate=False, 
                                     output_dir=None,
                                     visualize=False):
    """
    Parameters:
        image: 輸入的二值圖像
        min_area: 最小區域面積
        distance_threshold: 距離閾值，小於此距離的區域可能被視為同一區域
        save_separate: 是否儲存分離後的圖片
        output_dir: 輸出目錄
        visualize: 是否顯示視覺化結果
    
    Returns:
        white_regions: 分離的白色區域列表
        region_info: 區域資訊
        is_separated: 是否被隔斷
    """
    if len(image.shape) == 3:
        binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        binary = image.copy()
    
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    
    # 方法1：使用連通組件找出初始區域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # 分析區域之間的關係
    regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            regions.append({
                'label': i,
                'area': area,
                'centroid': centroids[i],
                'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]),
                'mask': (labels == i).astype(np.uint8) * 255
            })
    
    # 判斷是否有黑色隔斷（多個區域）
    is_separated = len(regions) > 1
    
    return regions, is_separated

def smooth_vessel_mask(mask, method, iterations):
    """
    平滑血管 mask 的邊緣
    
    Parameters:
        mask: 輸入的二值遮罩
        method: 'morphological', 'gaussian', 'bilateral'
        iterations: 迭代次數
    """
    if method == 'morphological':
        # 形態學平滑（先閉運算再開運算）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # # 閉運算：填補小洞
        # closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        # 開運算：移除小突起
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        
    elif method == 'gaussian':
        # 高斯模糊後二值化
        blurred = cv2.GaussianBlur(mask.astype(np.float32), (20, 20), 0)
        smoothed = (blurred > 127).astype(np.uint8) * 255
        
    elif method == 'bilateral':
        # 雙邊濾波（保留邊緣）
        bilateral = cv2.bilateralFilter(mask.astype(np.uint8), 9, 75, 75)
        smoothed = bilateral
        
    return smoothed

def centerline(mask, w, x1, x2, image, method='opencv'):
    if method == 'opencv':
        skeleton_cv = cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        # skeleton_cv = pruning_skeleton(skeleton_cv)
        skeleton_cv = crop(skeleton_cv, 3, 903, 3, 703)

        result_skeleton = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        # result_skeleton[0:700, x1:x2] = skeleton_cv
    
    return skeleton_cv

def is_valid_centerline(x_pts, y_pts, img_w=224):
    """ 判定中心線合法性並過濾雜訊 """
    if len(x_pts) < 15: return False
    
    width = np.max(x_pts) - np.min(x_pts)
    y_variability = np.sum(np.abs(np.diff(y_pts))) / (width + 1e-6)
    
    if width < 20: return False 
    if y_variability > 1.5: return False

    return True

def process_single_centerline(img_orig, mask_224):
    """ 處理單一張影像並生成中心線 Mask """
    
    # 讀取影像
    # img_orig = cv2.imread(img_path)
    # mask_224 = cv2.imread(label_path, 0)
    
    # if img_orig is None or mask_224 is None:
    #     print(f"Error: 無法讀取影像或標籤。檢查路徑: \n{img_path}\n{label_path}")
    #     return

    h_orig, w_orig = img_orig.shape[:2]
    scale_x, scale_y = w_orig / 224.0, h_orig / 224.0

    # 建立純黑畫布
    centerline_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

    # 二值化與標記連通域
    _, binary_mask = cv2.threshold(mask_224, 127, 255, cv2.THRESH_BINARY)
    labeled_array, num_features = label(binary_mask > 0)
    
    for i in range(1, num_features + 1):
        single_region = (labeled_array == i)
        if np.sum(single_region) < 20: continue

        dist_map = distance_transform_edt(single_region)
        skel_region = medial_axis(single_region)
        y_idx, x_idx = np.where(skel_region)
        
        region_best_pts = {}
        for x, y in zip(x_idx, y_idx):
            val = dist_map[y, x]
            if x not in region_best_pts or val > region_best_pts[x][1]:
                region_best_pts[x] = (y, val)
        
        sorted_x = np.array(sorted(region_best_pts.keys()))
        sorted_y = np.array([region_best_pts[x][0] for x in sorted_x])

        if not is_valid_centerline(sorted_x, sorted_y):
            continue

        # 座標縮放
        pts_scaled = np.column_stack((sorted_x * scale_x, sorted_y * scale_y))

        try:
            k_val = min(3, len(pts_scaled)-1)
            tck, u = splprep([pts_scaled[:, 0], pts_scaled[:, 1]], s=len(pts_scaled)*12, k=k_val)
            u_fine = np.linspace(0, 1, 2000) 

            x_fine, y_fine = splev(u_fine, tck)
            
            # 物理級細線繪製
            for x, y in zip(x_fine, y_fine):
                ix, iy = int(round(x)), int(round(y))
                if 0 <= ix < w_orig and 0 <= iy < h_orig:
                    centerline_mask[iy, ix] = 255
        except Exception as e:
            print(f"Spline fitting error: {e}")
            continue

    # 儲存結果
    # cv2.imwrite(output_path, centerline_mask)
    # print(f"成功儲存中心線 Mask 至: {output_path}")
    return centerline_mask

def draw_perpendicular_line(image, line_p1, line_p2, point, length=50, color=(0, 255, 0), thickness=2):
    """
    在指定點畫出與直線垂直的線
    
    Parameters:
        image: 輸入圖像
        line_p1: 原直線的第一個點 (x, y)
        line_p2: 原直線的第二個點 (x, y)
        point: 要畫垂直線的點 (x, y)
        length: 垂直線的長度（兩側各 length/2）
        color: 線條顏色 (B, G, R)
        thickness: 線條粗細
    """
    # 計算原直線的方向向量
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]
    
    # 計算原直線的長度
    line_length = math.sqrt(dx**2 + dy**2)
    
    if line_length == 0:
        return None
    
    # 計算單位方向向量
    ux = dx / line_length
    uy = dy / line_length
    
    # 垂直向量（旋轉90度）
    perp_x = -uy
    perp_y = ux
    
    # 計算垂直線的兩個端點
    half_length = length / 2
    p1 = (int(point[0] + perp_x * half_length), int(point[1] + perp_y * half_length))
    p2 = (int(point[0] - perp_x * half_length), int(point[1] - perp_y * half_length))
    
    # 繪製垂直線
    cv2.line(image, p1, p2, color, thickness)
    
    return p1, p2

def find_RangeGate(start_pt, target_pt, img):
    """
    沿著 start_pt->p_top 和 start_pt->p_bottom 尋找最近的黑白交界
    :param start_pt: 起始點 (x, y)
    :param target_pt: 線段一端 (x, y)
    :param img: 二值化影像 (255為血管內部, 0為背景)
    :return: (top_intersection, bottom_intersection)
    """
    h, w = img.shape
    start_x, start_y = start_pt
    target_x, target_y = target_pt
        
    # 1. 計算方向向量
    dx = target_x - start_x
    dy = target_y - start_y
    distance = np.sqrt(dx**2 + dy**2)
        
    if distance == 0:
        return start_pt
        
    # 2. 單位向量 (每次移動 1 像素)
    ux = dx / distance
    uy = dy / distance
        
    # 3. 沿線搜尋
    last_pt = start_pt
    # 搜尋的長度不超過到目標點的距離
    for d in np.arange(0, distance, 1.0):
        curr_x = int(start_x + d * ux) 
        curr_y = int(start_y + d * uy)
            
        # 檢查是否超出影像邊界
        if not (0 <= curr_y < h and 0 <= curr_x < w):
            return last_pt
            
        # 偵測交界點
        if img[curr_y, curr_x] == 0:
            final_x = int(round(curr_x - ux))
            final_y = int(round(curr_y - uy))
            
            # 點查是否還在影像內
            final_x = max(0, min(w - 1, final_x))
            final_y = max(0, min(h - 1, final_y))
            return (final_x, final_y)
            
        last_pt = (curr_x, curr_y)
            
    return last_pt

def get_boundary_intersection_direct(mask_shape, center_pt, angle_deg):
    """
    計算指定角度射線與圖片邊界的交點
    :param mask_shape: (height, width)
    :param center_pt: (x, y)
    :param angle_deg: 絕對角度 (度)
    :return: (pt_far_1, pt_far_2) 兩個在圖片邊緣的點
    """
    h, w = mask_shape
    cx, cy = center_pt
    
    # 1. 取得方向向量
    rad = np.radians(angle_deg)
    dx = np.cos(rad)
    dy = np.sin(rad)
    
    # 2. 計算到達四個邊界需要的距離 (d = delta_pos / direction)
    # 我們只考慮正向 (t > 0)
    distances = []
    
    # 避免除以零
    epsilon = 1e-9
    
    # 檢查與垂直邊界相交 (左右)
    if abs(dx) > epsilon:
        distances.append((0 - cx) / dx)      # 左邊界
        distances.append((w - 1 - cx) / dx)  # 右邊界
        
    # 檢查與水平邊界相交 (上下)
    if abs(dy) > epsilon:
        distances.append((0 - cy) / dy)      # 上邊界
        distances.append((h - 1 - cy) / dy)  # 下邊界
        
    # 3. 找出正向最靠近的距離，以及負向最靠近的距離
    # 正向距離中最小的正數，就是射線撞到邊框的位置
    t_pos = min([t for t in distances if t > 0])
    # 負向距離中最大的負數 (絕對值最小)，就是反向射線撞到邊框的位置
    t_neg = max([t for t in distances if t < 0])
    
    # 4. 根據距離回推座標
    pt_edge_1 = (int(cx + t_pos * dx), int(cy + t_pos * dy))
    pt_edge_2 = (int(cx + t_neg * dx), int(cy + t_neg * dy))
    
    if pt_edge_1[1] > pt_edge_2[1]:
        p_bottom = pt_edge_1
        p_top = pt_edge_2
    else:
        p_bottom = pt_edge_2
        p_top = pt_edge_1
    
    return p_top, p_bottom

def get_absolute_angle(v, relative_angle_deg):
    """
    計算旋轉後的向量相對於圖片水平軸(X軸)的絕對角度
    :param v: 基準向量 (dx, dy)
    :param relative_angle_deg: 相對於基準向量的夾角 (度)
    :return: 絕對角度 (0~360 度)
    """
    # 1. 取得基準向量本身的絕對角度 (弧度)
    # np.arctan2 參數順序是 (y, x)
    base_angle_rad = np.arctan2(v[1], v[0])
    
    # 2. 加上相對夾角 (轉為弧度)
    relative_angle_rad = np.radians(relative_angle_deg)
    final_angle_rad = base_angle_rad + relative_angle_rad
    
    # 3. 轉回角度制
    final_angle_deg = np.degrees(final_angle_rad)
    
    # 4. 正規化到 0~360 度之間 (選用，方便閱讀)
    final_angle_deg = final_angle_deg % 360
    
    return final_angle_deg

def get_direction_by_skan(skeleton_cv, target_point):
    """
    利用 Skan 找到離目標點最近的路徑，並回傳方向向量 (dx, dy)
    """
    try:
        # 建立 Skan 骨架物件
        skel = Skeleton(skeleton_cv > 0)
        stats = summarize(skel, separator='_')
        
        if stats.empty:
            print("Skan 沒有找到任何路徑，回傳水平向量")
            return np.array([1, 0]) # 預防萬一回傳水平向量

        # 1. 獲取整段 skel 的所有路徑座標
        all_coords = []
        for path_idx in stats.index:
            coords = skel.path_coordinates(path_idx)
            all_coords.extend(coords)
        path_coords = np.array(all_coords)  # [row, col] 格式

        # 2. 找到路徑上距離 target_point 最近的點的索引
        p_target = np.array([target_point[1], target_point[0]])
        dists = np.linalg.norm(path_coords - p_target, axis=1)
        nearest_idx = np.argmin(dists)

        # 3. 取鄰域點計算向量 (前後各 5 點以平滑鋸齒)
        idx_start = max(0, nearest_idx - 5)
        idx_end = min(len(path_coords) - 1, nearest_idx + 5)

        p_start = path_coords[idx_start]
        p_end = path_coords[idx_end]
        
        # 4. 計算向量
        # dr = y 變化量, dc = x 變化量
        dr = p_end[0] - p_start[0]
        dc = p_end[1] - p_start[1]
        
        # 正規化為單位向量 (dx, dy)
        magnitude = np.sqrt(dr**2 + dc**2)
        if magnitude == 0:
            return np.array([1, 0])
            
        # 注意順序：我們回傳 (dx, dy)，對應 (dc, dr)
        direction = np.array([dc / magnitude, dr / magnitude])
        
        if direction[0] > 0:
            return -direction
        
        return direction 

    except Exception as e:
        print(f"Skan 方向計算失敗: {e}")
        return np.array([1, 0])
    
def get_direction_by_hessian(skeleton_cv, target_point):
        
    # 1. 稍微模糊以獲得連續的曲率
    smoothed = cv2.GaussianBlur(skeleton_cv.astype(np.float32), (5, 5), 0)
    
    # 2. 計算 Hessian 矩陣元素 (Hrr, Hrc, Hcc)
    H = hessian_matrix(smoothed, sigma=1.5, order='rc')
    Hrr, Hrc, Hcc = H
    
    x, y = int(target_point[0]), int(target_point[1])
    h, w = skeleton_cv.shape
    y = max(0, min(h - 1, y))
    x = max(0, min(w - 1, x))
    
    # 3. 取得目標點位置的 Hessian 矩陣數值
    # 矩陣形式為: [[Hrr, Hrc], 
    #             [Hrc, Hcc]]
    matrix = np.array([
        [Hrr[y, x], Hrc[y, x]],
        [Hrc[y, x], Hcc[y, x]]
    ])
    
    # 4. 計算特徵值與特徵向量
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        
        idx = np.argmin(np.abs(eigenvalues))
        v = eigenvectors[:, idx]
        
        dy, dx = v[0], v[1]
        
        # 5. 正規化向量
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude < 1e-6:
            print("Hessian 方向向量過小，回傳水平向量")
            return np.array([1.0, 0.0])
            
        return np.array([dx / magnitude, dy / magnitude])
        
    except np.linalg.LinAlgError:
        print("Hessian 矩陣特徵值計算失敗，回傳水平向量")
        return np.array([1.0, 0.0])

def calculate_angle_between_vectors(p1, p2, v_given, absolute=False):
    """
    計算兩點(p1, p2)連線向量與給定向量(v_given)之間的夾角
    :param p1: 點1 (x1, y1)
    :param p2: 點2 (x2, y2)
    :param v_given: 給定向量 (dx, dy)
    :param absolute: 若為 True，回傳 0~90 度的最小夾角 (適合算 Range Gate 與血管夾角)
    :return: 角度 (Degree)
    """
    # 1. 算出兩點連線的向量 A
    vec_a = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    
    # 2. 準備給定向量 B
    vec_b = np.array(v_given)
    
    # 3. 計算長度 (Norm)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    # 防止除以零
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    # 4. 計算點積並求出 cos(theta)
    dot_product = np.dot(vec_a, vec_b)
    cos_theta = dot_product / (norm_a * norm_b)
    
    # 限制 cos 範圍在 -1 到 1 之間，避免浮點數誤差導致 NaN
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # 5. 轉為角度
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    if absolute:
        # 將角度轉為 -90 ~ 90 度
        if angle_deg > 90:
            angle_deg = angle_deg - 180
            
    return angle_deg

def draw_tangent(img, point, direction, length=20):
    x0, y0 = point
    dx, dy = direction

    # 正規化
    norm = np.sqrt(dx**2 + dy**2)
    dx /= norm
    dy /= norm

    x1 = int(x0 - dx * length)
    y1 = int(y0 - dy * length)
    x2 = int(x0 + dx * length)
    y2 = int(y0 + dy * length)

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

def post_process(base_line_dir, base_mask_dir, base_ori_dir):
    output_dir = "result/line"
    os.makedirs(output_dir, exist_ok=True)


    files = [f for f in os.listdir(base_line_dir) if f.endswith('.png')]
    print(f"{base_line_dir} 總共有 {len(files)} 張圖片需要後處理...")

    for filename in files:
        start = time.perf_counter()

        # ===== 路徑 =====
        line_path = os.path.join(base_line_dir, filename)
        # mask_path = os.path.join(base_mask_dir, filename.replace("_line.png", ".png"))
        # ori_path  = os.path.join(base_ori_dir, filename.replace("_line.png", ".png"))
        mask_path = os.path.join(base_mask_dir, filename.replace(".png", "_label.png"))
        ori_path  = os.path.join(base_ori_dir, filename)
    
        # ===== 讀圖 =====
        line = cv2.imread(line_path)
        mask = cv2.imread(mask_path, 0)
        img_ori = cv2.imread(ori_path, 0)

        if line is None or mask is None:
            print(f"跳過 {filename}")
            continue

        if img_ori is None:
            print(f"找不到 {ori_path}")
            img_ori = mask.copy()

        #  ======= 224 -> 900*700 ======= 
        image = resize(mask, 900)
        image = crop(image, 0, 900, 100, 800)
        img_ori = resize(img_ori, 900)
        # img_ori = crop(img_ori, 0, 900, 100, 800)

        #  ======= 找線的端點 ======= 
        p_top, p_bottom = detected_line(line)
        if p_top is None or p_bottom is None:
            print(f"{filename} 沒有找到線段")
            continue

        if p_top[0] > p_bottom[0]:
            x1 = p_bottom[0] - 20
            x2 = p_top[0] + 20
            w = x2 - x1
        else :
            x1 = p_top[0] - 20
            x2 = p_bottom[0] + 20
            w = x2 - x1

        #  ======= 畫線 ======= 
        lines = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.line(lines, p_top, p_bottom, 255, 1) 

        #  ======= 選定骨架化範圍 ======= 
        # image_crop = crop(image, x1, x2, 0, 700)
        # image_crop = image.copy()
            
        # region, is_separated = separate_white_regions_advanced(image_crop, min_area=100, save_separate=False, visualize=False)
        # if is_separated:
        #     image_crop = region[0]['mask']

        # blur = smooth_vessel_mask(image_crop, method='morphological', iterations=10)

        # #  ======= padding ======= 
        # bin_padR = padding_Replication(blur)

        #  ======= 中心線 =======
        # centerLine = centerline(bin_padR, w, x1, x2, image)
        centerLine = process_single_centerline(img_ori, mask)
        img_ori = crop(img_ori, 0, 900, 100, 800)
        centerLine = crop(centerLine, 0, 900, 100, 800)
        
        if np.sum(centerLine > 0) == 0:
            print(f"Warning: {filename}Skeleton is empty, skipping this image.")
            continue 

        #  ======= 找到中心線與綠線交點 =======
        # print(f"交點數量: {lines.shape} | centerLine :{centerLine.shape}")
        intersection_mask = cv2.bitwise_and(lines, centerLine)
        points = np.where(intersection_mask == 255)
        points_list = list(zip(points[1], points[0]))

        # ======== 處理交點 ==========
        if len(points_list) >= 1:
            mask = cv2.bitwise_and(lines, image)
            green_points = np.where(mask == 255)
            
            center = points_list[0]
            
            # ======== Angle ========== 
            direction = get_direction_by_skan(centerLine, center)
            
            # ======== range gate ==========
            if len(green_points[0]) > 0:
                all_points = list(zip(green_points[1], green_points[0]))
                
                in_top = min(all_points, key=lambda p: p[1])
                in_bottom = max(all_points, key=lambda p: p[1])
                intersection_top = find_RangeGate(center, in_top, image)
                intersection_bottom = find_RangeGate(center, in_bottom, image)
        
        # ======== 以原圖處理 ========
        if len(points_list) < 1:
            # # ======== 切割多塊血管 ========
            # region, is_separated = separate_white_regions_advanced(image, min_area=100, save_separate=False, visualize=False)
            # if is_separated:
            #     image = region[0]['mask']

            # blur = smooth_vessel_mask(image, method='morphological', iterations=10)

            # #  ======= padding ======= 
            # bin_padR = padding_Replication(blur)
            
            # # ======== 中心線 ========
            # centerLine = centerline(bin_padR, 900, 0, 900, image)
            
            # ======== 找中心線中點 ========
            skel = Skeleton(centerLine > 0)
            stats = summarize(skel, separator='_')
            
            main_path_idx = stats['branch_distance'].idxmax()
            path_coords = skel.path_coordinates(main_path_idx) # [row, col]
            
            target_idx = int(len(path_coords) * 0.5)
            center_r, center_c = path_coords[target_idx]
            center = (int(center_c), int(center_r))
            
            # ======== Angle ========
            direction = get_direction_by_skan(centerLine, center)
            
            # ======== 預設線段(-75) ========
            p_top, p_bottom = get_boundary_intersection_direct(image.shape, center, get_absolute_angle(direction, 60))
            
            lines = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.line(lines, p_top, p_bottom, 255, 1) 
                    
            mask = cv2.bitwise_and(lines, image)
            green_points = np.where(mask == 255)
    
            # ======== range gate ==========
            if len(green_points[0]) > 0:
                all_points = list(zip(green_points[1], green_points[0]))
                
                in_top = min(all_points, key=lambda p: p[1])
                in_bottom = max(all_points, key=lambda p: p[1])
                intersection_top = find_RangeGate(center, in_top, image)
                intersection_bottom = find_RangeGate(center, in_bottom, image)

            
        #  ======= 視覺化 =======
        result = img_ori.copy()
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        # 畫骨架
        skeleton_color = cv2.cvtColor(centerLine, cv2.COLOR_GRAY2BGR)
        skeleton_color[centerLine == 255] = [0, 255, 0]
        result = cv2.addWeighted(result, 0.7, skeleton_color, 0.3, 0)
        
        # 畫 Range Gate
        draw_perpendicular_line(result, p_top, p_bottom, intersection_top, length=50, color=(0, 0, 255), thickness=2)
        draw_perpendicular_line(result, p_top, p_bottom, intersection_bottom, length=50, color=(0, 0, 255), thickness=2)
        
        # 畫 Angle
        if direction is not None:
            draw_tangent(result, center, direction)
        
        # 畫 beam
        cv2.line(result, p_top, intersection_top, (0, 255, 0), 1)
        cv2.line(result, intersection_bottom, p_bottom, (0, 255, 0), 1)
        angle = calculate_angle_between_vectors(intersection_bottom, intersection_top, direction, absolute=True)

        # ===== 存檔 =====
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, result)
        
        # print(f"{filename} | {time.perf_counter()-start:.3f}s | intersection: {len(points_list)}")
        print(f"{filename} | Angle:  angle: {angle:.2f} degree")
        # print(f"p: {p_top}, {p_bottom} | intersection: {intersection_top}, {intersection_bottom} | center: {center} | len(points_list): {len(points_list)}")

    print("\n=== 全部處理完成 ===")
    
    
    
    
# base_line_dir = r"C:\collega\Project\data\dealData_post\line\data5"
# base_mask_dir = r"C:\collega\Project\data\dealData_post\masks_cleaned"
# base_ori_dir = r"C:\collega\Project\data\dealData_post\images"
# post_process(base_line_dir, base_mask_dir, base_ori_dir)
