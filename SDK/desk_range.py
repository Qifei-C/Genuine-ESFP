import cv2
import cv2.aruco as aruco
import numpy as np

def order_points(pts):
    """
    对四个点进行排序，返回按顺序排列的点：
    [左上, 右上, 右下, 左下]
    """
    rect = np.zeros((4, 2), dtype="float32")
    # 按 x 坐标排序，分为左右两部分
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    
    # 左侧点按 y 坐标排序得到 tl, bl
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    
    # 右侧点按 y 坐标排序得到 tr, br
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost
    
    rect[0] = tl
    rect[1] = tr
    rect[2] = br
    rect[3] = bl
    return rect

def corners_are_similar(c1, c2, threshold=5):
    """
    判断两组角点是否足够接近（逐点欧几里得距离小于 threshold）
    c1, c2: shape=(4,2) 的坐标数组
    threshold: 允许的像素距离阈值
    """
    diffs = np.linalg.norm(c1 - c2, axis=1)  # 计算每个对应点的距离
    return np.all(diffs < threshold)

def perspective_correction(frame, frozen_corners):
    """
    对桌面进行透视矫正，返回正方形（鸟瞰图）结果。
    
    参数：
    - frame: 原始图像
    - frozen_corners: 形状为 (4,2) 的数组，按照 [左上, 右上, 右下, 左下] 顺序排列
    """
    # 拆分角点
    (tl, tr, br, bl) = frozen_corners
    
    # 计算宽度：左右两边
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    # 计算高度：上下两边
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    # 既然桌面是正方形，则取较大的边长作为目标尺寸
    side = max(maxWidth, maxHeight)
    
    # 构建目标正方形的四个角点：[左上, 右上, 右下, 左下]
    dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ], dtype="float32")
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(frozen_corners, dst)
    
    # 对整个图像进行透视变换
    warped = cv2.warpPerspective(frame, M, (side, side))
    return warped

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# 记录是否已经冻结坐标
frozen = False
frozen_corners = None

# 用于连续检测成功的计数
success_count = 0
# 需要连续多少帧检测稳定才算冻结
MIN_STABLE_FRAMES = 2
# 用于比较当前帧和上一帧的角点
last_corners = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 如果已经冻结坐标，则直接在画面上绘制并进行透视矫正
    if frozen and frozen_corners is not None:
        pts_int = frozen_corners.astype(int)
        cv2.polylines(frame, [pts_int], True, (0, 255, 0), 2)
        for i, p in enumerate(pts_int):
            cv2.circle(frame, tuple(p), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"pt{i}", tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, "Table coords frozen", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # 进行透视矫正，得到正方形（鸟瞰图）
        warped = perspective_correction(frame, frozen_corners)
        cv2.imshow("Warped Table", warped)
        
        cv2.imshow("Table Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # 未冻结时进行标记检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    marker_centers = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            c = corners[i][0]
            center = c.mean(axis=0)
            marker_centers[marker_id] = center
        aruco.drawDetectedMarkers(frame, corners, ids)
    
    expected_ids = [0, 1, 2, 3]
    missing_ids = [eid for eid in expected_ids if eid not in marker_centers]
    
    if len(missing_ids) == 0:
        # 提取四个角的坐标并排序
        pts = np.array([marker_centers[k] for k in expected_ids], dtype="float32")
        ordered_pts = order_points(pts)
        
        # 在画面上绘制当前检测到的轮廓
        ordered_pts_int = ordered_pts.astype(int)
        cv2.polylines(frame, [ordered_pts_int], True, (0, 255, 0), 2)
        for i, point in enumerate(ordered_pts_int):
            cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"pt{i}", tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 判断与上一帧是否接近，如果接近则 success_count + 1，否则重置
        if last_corners is not None:
            if corners_are_similar(ordered_pts, last_corners):
                success_count += 1
            else:
                success_count = 0
        else:
            # 第一次有检测结果时，初始化计数
            success_count = 1
        
        last_corners = ordered_pts.copy()
        
        # 如果连续检测 MIN_STABLE_FRAMES 帧都相似，认为稳定，冻结坐标
        if success_count >= MIN_STABLE_FRAMES:
            frozen = True
            frozen_corners = ordered_pts
            print("桌面坐标已冻结：", frozen_corners)
    else:
        # 没有检测到所有标记时重置计数
        success_count = 0
        print(f"桌面未检测到！缺失的标记ID: {missing_ids}")
    
    cv2.imshow("Table Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()