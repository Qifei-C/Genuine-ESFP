import matplotlib
matplotlib.use('TkAgg')  # 强制使用支持交互的后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
import mediapipe as mp
import math
import numpy as np

from vision.keypoint_converter import extract_coco_17_from_mediapipe
from vision.pose3d_estimator import Pose3DEstimator

# --------- 初始化 MediaPipe Hands 和 Pose 模块 ------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def is_valid_landmark(lm):
    return 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0

def calculate_distance(lm1, lm2):
    return math.hypot(lm2.x - lm1.x, lm2.y - lm1.y)
 
def is_hand_open(hand_landmarks):
    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return calculate_distance(thumb, index) > 0.05

# --------- 初始化 3D 姿态估计器 ----------
pose3d_estimator = Pose3DEstimator('model/pretrained_model.bin')

# --------- 初始化 Matplotlib 视图 ----------
plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=80, subplot_kw={'projection': '3d'})

# --------- 初始化 Matplotlib 视图 ---------- 
fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=80)  # 使用普通的 2D 图，而不是 3D 图

# 设置每个子图的标题
axs[0, 0].set_title("XY Plane")
axs[0, 1].set_title("XZ Plane")
axs[1, 0].set_title("YZ Plane")
axs[1, 1].set_title("Arm Pose (Top View)")

# 设置轴范围
for ax in axs.flat:
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

plt.show(block=False)  # 弹出非阻塞窗口

# --------- 打开摄像头 ----------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2D 姿态 & 手势 检测
    hand_results = hands.process(frame_rgb)
    pose_results = pose.process(frame_rgb)

    height, width = frame.shape[:2]

    # —— 2D 可视化 & 坐标打印 —— 
    if (pose_results.pose_landmarks and
        pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.7):

        # 绘制骨架
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 提取肩部、肘部、手腕的坐标
        s = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        e = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        wpt = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        # 画圆标注
        cv2.circle(frame, (int(s.x * width), int(s.y * height)), 5, (255, 255, 0), -1)
        cv2.circle(frame, (int(e.x * width), int(e.y * height)), 5, (255, 255, 0), -1)
        cv2.circle(frame, (int(wpt.x * width), int(wpt.y * height)), 5, (255, 255, 0), -1)

        # —— 计算空间差值并使用肩部到肘部的长度进行归一化 —— 
        shoulder_to_elbow_length = calculate_distance(s, e)
        delta_elbow = np.array([e.x - s.x, e.y - s.y, e.z - s.z])
        delta_wrist = np.array([wpt.x - s.x, wpt.y - s.y, wpt.z - s.z])

        # 归一化这些差值
        delta_elbow /= shoulder_to_elbow_length
        delta_wrist /= shoulder_to_elbow_length

        # 更新 3D 坐标
        ls = np.array([0, 0, 0])  # Shoulder
        le = ls + delta_elbow      # Elbow
        lw = ls + delta_wrist      # Wrist

        print(f"[3D] Shoulder: ({ls[0]:.3f}, {ls[1]:.3f}, {ls[2]:.3f})")
        print(f"[3D] Elbow:    ({le[0]:.3f}, {le[1]:.3f}, {le[2]:.3f})")
        print(f"[3D] Wrist:    ({lw[0]:.3f}, {lw[1]:.3f}, {lw[2]:.3f})")

        # —— 绘制不同视角的 2D 投影 —— 
        # XY Plane (Z=0)
        axs[0, 0].cla()
        axs[0, 0].scatter(ls[0], ls[1], color='blue', s=100)  # Shoulder
        axs[0, 0].scatter(le[0], le[1], color='blue', s=100)  # Elbow
        axs[0, 0].scatter(lw[0], lw[1], color='blue', s=100)  # Wrist
        axs[0, 0].plot([ls[0], le[0], lw[0]], [ls[1], le[1], lw[1]], color='blue', linewidth=2)

        # XZ Plane (Y=0)
        axs[0, 1].cla()
        axs[0, 1].scatter(ls[0], ls[2], color='blue', s=100)  # Shoulder
        axs[0, 1].scatter(le[0], le[2], color='blue', s=100)  # Elbow
        axs[0, 1].scatter(lw[0], lw[2], color='blue', s=100)  # Wrist
        # 将y=0固定，使得投影在XZ平面上
        axs[0, 1].plot([ls[0], le[0], lw[0]], [0, 0, 0], [ls[2], le[2], lw[2]], color='blue', linewidth=2)

        # YZ Plane (X=0)
        axs[1, 0].cla()
        axs[1, 0].scatter(ls[1], ls[2], color='blue', s=100)  # Shoulder
        axs[1, 0].scatter(le[1], le[2], color='blue', s=100)  # Elbow
        axs[1, 0].scatter(lw[1], lw[2], color='blue', s=100)  # Wrist
        # 将x=0固定，使得投影在YZ平面上
        axs[1, 0].plot([0, 0, 0], [ls[1], le[1], lw[1]], [ls[2], le[2], lw[2]], color='blue', linewidth=2)

        # 3D Arm Pose (Top view)
        axs[1, 1].cla()
        axs[1, 1].scatter(ls[0], ls[1], color='blue', s=100)  # Shoulder
        axs[1, 1].scatter(le[0], le[1], color='blue', s=100)  # Elbow
        axs[1, 1].scatter(lw[0], lw[1], color='blue', s=100)  # Wrist
        axs[1, 1].plot([ls[0], le[0], lw[0]], [ls[1], le[1], lw[1]], color='blue', linewidth=2)

        # 更新 Matplotlib 实时显示
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

    # —— 手势识别 —— 
    if hand_results.multi_hand_landmarks:
        for hl in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            txt = "Hand Open" if is_hand_open(hl) else "Fist"
            color = (0, 255, 0) if txt == "Hand Open" else (0, 0, 255)
            cv2.putText(frame, txt, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Hand and Pose Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
