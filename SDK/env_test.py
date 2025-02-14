import cv2
import cv2.aruco as aruco
import numpy as np
from uarm.wrapper import SwiftAPI
import time

# ======= 常用函数 =======

def order_points(pts):
    """
    对四个点进行排序，返回顺序为 [左上, 右上, 右下, 左下] 的点坐标。
    """
    rect = np.zeros((4, 2), dtype="float32")
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost
    
    rect[0] = tl
    rect[1] = tr
    rect[2] = br
    rect[3] = bl
    return rect

def corners_are_similar(c1, c2, threshold=5):
    """
    判断两组角点是否足够接近（逐点欧几里得距离小于 threshold）。
    """
    diffs = np.linalg.norm(c1 - c2, axis=1)
    return np.all(diffs < threshold)

def perspective_correction(frame, frozen_corners):
    """
    对桌面进行透视矫正，返回正方形（鸟瞰图）结果。
    
    frozen_corners: 按 [左上, 右上, 右下, 左下] 顺序排列的桌面角点。
    """
    (tl, tr, br, bl) = frozen_corners
    
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    side = max(maxWidth, maxHeight)
    
    dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(frozen_corners, dst)
    warped = cv2.warpPerspective(frame, M, (side, side))
    return warped, side

def pixel_to_table_coords(u, v, warped_side, table_real_width, table_real_height):
    """
    将透视矫正后图像的像素坐标 (u, v) 转换为桌面坐标（单位 mm），
    桌面坐标系的原点固定为桌面左上角 (0,0)。
    """
    scale_x = table_real_width / warped_side
    scale_y = table_real_height / warped_side
    table_x = u * scale_x
    table_y = v * scale_y
    return table_x, table_y

def table_to_robot_with_calibration(u, v, warped_side, table_real_width, table_real_height, A):
    """
    先将 (u,v) 像素坐标转换为桌面坐标，再利用仿射矩阵 A 得到机械臂坐标。
    """
    table_coord = np.array(pixel_to_table_coords(u, v, warped_side, table_real_width, table_real_height))
    table_coord_aug = np.hstack([table_coord, 1.0])  # 形成 [x, y, 1]
    robot_coord = np.dot(A, table_coord_aug)
    return robot_coord[0], robot_coord[1]

# ======= 初始化部分 =======

# 初始化机械臂（请确保机械臂 SDK 正常安装）
arm = SwiftAPI()
time.sleep(1)
if not arm.connected:
    print("机械臂未连接，正在尝试连接...")
    arm.connect()

# 摄像头初始化
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# 桌面角点冻结相关变量（利用标记 0,1,2,3）
frozen = False
frozen_corners = None
success_count = 0
MIN_STABLE_FRAMES = 2
last_corners = None

# 桌面实际尺寸（单位 mm），例如 510×510
TABLE_REAL_WIDTH = 510
TABLE_REAL_HEIGHT = 510

# 标记校准状态和校准矩阵（2×3 仿射矩阵）
calibration_done = False
calibration_matrix = None

# ======= 效应器校准函数 =======

def run_effector_calibration(arm, frozen_corners, cap, calibration_positions, table_real_width, table_real_height):
    """
    通过机械臂移动到多个校准点并检测效应器标签（id==4），建立
    桌面坐标（以桌面左上角为原点）与机械臂坐标之间的仿射变换矩阵。
    
    参数：
      arm: 机械臂对象
      frozen_corners: 桌面透视矫正的角点（已冻结）
      cap: 摄像头捕获对象
      calibration_positions: 机械臂应移动到的目标位置列表，每项为 (x, y, z)
      table_real_width, table_real_height: 桌面实际尺寸（mm）
    
    返回：
      2×3 仿射矩阵 A，使得：
         [robot_x, robot_y]^T = A * [table_x, table_y, 1]^T
      如果有效点不足，则返回 None。
    """
    detected_points = []  # 从图像检测到的桌面坐标（mm，原点在桌面左上角）
    robot_points = []     # 对应的机械臂坐标（x, y）

    for pos in calibration_positions:
        print(f"\n[校准] 正在移动到校准点: {pos}")
        # 移动到预设校准点（确保 z 轴高度保证摄像头能看到效应器标签）
        arm.set_position(x=pos[0], y=pos[1], z=pos[2], wait=True, timeout=5)
        time.sleep(2)  # 等待稳定

        # 捕获图像
        ret, frame = cap.read()
        if not ret:
            print("校准过程中捕获图像失败。")
            continue

        # 利用冻结的桌面角点进行透视矫正
        warped, warped_side = perspective_correction(frame, frozen_corners)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is None or 4 not in ids:
            print("在该位置未检测到效应器标签（id==4）。")
            continue

        # 找到 id==4 的标签，计算其中心像素坐标
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == 4:
                effector_corners = corners[i][0]
                effector_center = effector_corners.mean(axis=0)
                break

        # 显示校准图像（可选）
        cv2.circle(warped, (int(effector_center[0]), int(effector_center[1])), 5, (0, 0, 255), -1)
        cv2.putText(warped, "id4", (int(effector_center[0]) + 10, int(effector_center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.imshow("Calibration Warped", warped)
        cv2.waitKey(500)

        # 将鸟瞰图像素坐标转换为桌面坐标（以桌面左上角为 (0,0)）
        detected_table_x, detected_table_y = pixel_to_table_coords(
            effector_center[0], effector_center[1], warped_side, table_real_width, table_real_height)
        print(f"检测到的桌面坐标: ({detected_table_x:.1f}, {detected_table_y:.1f})")
        detected_points.append([detected_table_x, detected_table_y])
        # 使用机械臂运动指令的 x,y（你设定的目标坐标）
        robot_points.append([pos[0], pos[1]])

    if len(detected_points) < 3:
        print("校准点不足，至少需要 3 个有效点进行仿射校准。")
        return None

    detected_points = np.array(detected_points, dtype=np.float32)
    robot_points = np.array(robot_points, dtype=np.float32)

    # 计算从桌面坐标到机械臂坐标的仿射变换矩阵
    A, inliers = cv2.estimateAffinePartial2D(detected_points, robot_points)
    print("\n=== 校准完成 ===")
    print("仿射变换矩阵 A：\n", A)
    return A

# ======= 主循环 =======

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 如果桌面角点已冻结，则显示相关信息
    if frozen and frozen_corners is not None:
        pts_int = frozen_corners.astype(int)
        cv2.polylines(frame, [pts_int], True, (0, 255, 0), 2)
        for i, p in enumerate(pts_int):
            cv2.circle(frame, tuple(p), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"pt{i}", tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, "Table coords frozen", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow("Table Detection", frame)

        # 按下 'c' 键触发校准流程（如果还未校准）
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and not calibration_done:
            # 预设校准点，确保在这些位置效应器标签（id==4）均能被检测到
            calibration_positions = [
                (150, 0, 100),
                (175, 0, 100),
                (200, 0, 100),
                (225, 0, 100),
                (250, 0, 100)
            ]
            calibration_matrix = run_effector_calibration(arm, frozen_corners, cap,
                                                            calibration_positions,
                                                            TABLE_REAL_WIDTH, TABLE_REAL_HEIGHT)
            if calibration_matrix is not None:
                calibration_done = True
            else:
                print("校准未成功，请重新尝试。")
        elif key == ord('q'):
            break
        continue

    # 若桌面未冻结，通过检测标记 0,1,2,3 确定桌面角点
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
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
        pts = np.array([marker_centers[k] for k in expected_ids], dtype="float32")
        ordered_pts = order_points(pts)
        
        ordered_pts_int = ordered_pts.astype(int)
        cv2.polylines(frame, [ordered_pts_int], True, (0, 255, 0), 2)
        for i, point in enumerate(ordered_pts_int):
            cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"pt{i}", tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        if last_corners is not None:
            if corners_are_similar(ordered_pts, last_corners):
                success_count += 1
            else:
                success_count = 0
        else:
            success_count = 1
        
        last_corners = ordered_pts.copy()
        
        if success_count >= MIN_STABLE_FRAMES:
            frozen = True
            frozen_corners = ordered_pts
            print("桌面坐标已冻结：", frozen_corners)
    else:
        success_count = 0
        print(f"桌面未检测到！缺失的标记ID: {missing_ids}")
    
    cv2.imshow("Table Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()