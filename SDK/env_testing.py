import cv2
import cv2.aruco as aruco
import numpy as np
from uarm.wrapper import SwiftAPI
import time
import threading

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

'''
def table_to_robot_with_calibration(u, v, warped_side, table_real_width, table_real_height, A):
    """
    先将 (u,v) 像素坐标转换为桌面坐标，再利用仿射矩阵 A 得到机械臂坐标。
    """
    table_coord = np.array(pixel_to_table_coords(u, v, warped_side, table_real_width, table_real_height))
    table_coord_aug = np.hstack([table_coord, 1.0])  # 形成 [x, y, 1]
    robot_coord = np.dot(A, table_coord_aug)
    return robot_coord[0], robot_coord[1]

'''

def table_to_robot_with_calibration(u, v, warped_side, table_real_width, table_real_height, A):
    """
    先将 (u, v) 像素坐标转换为桌面坐标，再利用仿射矩阵 A 得到机械臂坐标。
    这里对 x 坐标进行镜像翻转以校正左右方向。
    """
    # 得到桌面坐标（原点在桌面左上角）
    table_coord = np.array(pixel_to_table_coords(u, v, warped_side, table_real_width, table_real_height))
    # 对 x 坐标进行镜像处理：将其转换为以桌面右边为原点的坐标值
    table_coord[0] = table_real_width - table_coord[0]
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

# 全局变量
frozen = False
frozen_corners = None
success_count = 0
MIN_STABLE_FRAMES = 2
last_corners = None

# 桌面实际尺寸（单位 mm），例如 510×510
TABLE_REAL_WIDTH = 510
TABLE_REAL_HEIGHT = 510

# 校准状态和仿射矩阵（2×3 矩阵）
calibration_done = False
calibration_matrix = None

# 用于标记机械臂是否正在移动
arm_moving = False

# ======= 效应器校准函数 =======

def run_effector_calibration(arm, frozen_corners, cap, calibration_positions, table_real_width, table_real_height):
    """
    通过机械臂移动到多个校准点并检测效应器标签（id==4），建立桌面坐标（以桌面左上角为原点）
    与机械臂坐标之间的仿射变换矩阵。
    """
    detected_points = []  # 图像检测到的桌面坐标（mm，原点在桌面左上角）
    robot_points = []     # 对应的机械臂坐标（x, y）

    for pos in calibration_positions:
        print(f"\n[校准] 正在移动到校准点: {pos}")
        arm.set_position(x=pos[0], y=pos[1], z=pos[2], wait=True, timeout=5)
        time.sleep(2)  # 等待稳定

        ret, frame = cap.read()
        if not ret:
            print("校准过程中捕获图像失败。")
            continue

        warped, warped_side = perspective_correction(frame, frozen_corners)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is None or 4 not in ids:
            print("在该位置未检测到效应器标签（id==4）。")
            continue

        # 找到 id==4 标签并计算中心坐标
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

        detected_table_x, detected_table_y = pixel_to_table_coords(
            effector_center[0], effector_center[1], warped_side, table_real_width, table_real_height)
        print(f"检测到的桌面坐标: ({detected_table_x:.1f}, {detected_table_y:.1f})")
        detected_points.append([detected_table_x, detected_table_y])
        robot_points.append([pos[0], pos[1]])

    if len(detected_points) < 3:
        print("校准点不足，至少需要 3 个有效点进行仿射校准。")
        return None

    detected_points = np.array(detected_points, dtype=np.float32)
    robot_points = np.array(robot_points, dtype=np.float32)
    A, inliers = cv2.estimateAffinePartial2D(detected_points, robot_points)
    print("\n=== 校准完成 ===")
    print("仿射变换矩阵 A：\n", A)
    return A

# ======= 主流程 =======

print("等待桌面角点冻结（请确保桌面上放置了aruco标签 0, 1, 2, 3）……")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 若桌面已冻结，则显示冻结信息并等待校准命令
    if frozen and frozen_corners is not None:
        pts_int = frozen_corners.astype(int)
        cv2.polylines(frame, [pts_int], True, (0, 255, 0), 2)
        for i, p in enumerate(pts_int):
            cv2.circle(frame, tuple(p), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"pt{i}", tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, "Table coords frozen", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow("Table Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and not calibration_done:
            # time.sleep(15)  # Pause execution for 15 seconds
            # 预设校准点（建议使用 y=0 以确保贴纸识别效果好）
            calibration_positions = [
                (150, 50, 100),
                (160, 40, 100),
                (175, 25, 100),
                (200, 0, 100),
                (225, -25, 100),
                (240, -40, 100),
                (250, -50, 100)
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

    # 若桌面未冻结，通过检测aruco标签 0,1,2,3 来确定桌面角点
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

# ======= 验证模式 =======
# 利用桌面透视矫正后检测aruco标签 id==5，并将其转换为机械臂坐标后控制机械臂移动
print("验证程序启动……")
print("请确保桌面角点 (frozen_corners) 与仿射矩阵 (calibration_matrix) 已经通过校准获得。")
print("将aruco标签 id==5 放置在桌面上后，程序将控制机械臂将效应器移动到标签对应的位置。")

desired_z = 100  # 目标 z 轴高度

# 重新打开摄像头（如果需要）
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 用于标记机械臂是否正在移动，避免重复下发指令
arm_moving = False

def move_arm_to(target_robot_x, target_robot_y, desired_z):
    global arm_moving
    print("正在控制机械臂移动到目标位置……")
    try:
        arm.set_position(x=target_robot_x, y=target_robot_y, z=desired_z, wait=True, timeout=5)
    except Exception as e:
        print("移动过程中发生错误:", e)
    print("机械臂移动完成")
    arm_moving = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    warped, warped_side = perspective_correction(frame, frozen_corners)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # 检测aruco标签 id==5
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if ids is not None and 5 in ids:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == 5:
                marker_corners = corners[i][0]
                marker_center = marker_corners.mean(axis=0)
                break
        
        cv2.circle(warped, (int(marker_center[0]), int(marker_center[1])), 5, (0, 0, 255), -1)
        cv2.putText(warped, "id5", (int(marker_center[0]) + 10, int(marker_center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 将检测到的标签中心转换为机械臂坐标
        target_robot_x, target_robot_y = table_to_robot_with_calibration(
            marker_center[0], marker_center[1],
            warped_side, TABLE_REAL_WIDTH, TABLE_REAL_HEIGHT,
            calibration_matrix)
        
        print("检测到aruco==5标签，计算目标机械臂坐标：X={:.1f}, Y={:.1f}".format(target_robot_x, target_robot_y))
        
        # 如果机械臂未在运动状态，则启动新线程执行运动命令
        if not arm_moving:
            arm_moving = True
            t = threading.Thread(target=move_arm_to, args=(target_robot_x, target_robot_y, desired_z))
            t.start()
    else:
        print("未检测到aruco==5标签，请检查标签是否在视野内。")
    
    cv2.imshow("Verification Warped", warped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()