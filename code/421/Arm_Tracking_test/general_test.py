import cv2
import mediapipe as mp
import math

# 初始化 MediaPipe Hands 和 Pose 模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(0)

def is_valid_landmark(landmark):
    return 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1

def calculate_distance(landmark1, landmark2):
    # 计算两点之间的欧几里得距离
    x1, y1, z1 = landmark1.x, landmark1.y, landmark1.z
    x2, y2, z2 = landmark2.x, landmark2.y, landmark2.z
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def is_hand_open(hand_landmarks):
    # 判断手势是否为张开手
    # 我们通过比对握拳时，指尖和掌心之间的距离来判断手是否张开
    # 例如：大拇指和食指之间的距离
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    distance = calculate_distance(thumb_tip, index_tip)

    # 若距离大于某个阈值，则认为是张开手
    return distance > 0.05


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # 转换为 RGB 格式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 进行手部姿态估计
    hand_results = hands.process(frame_rgb)

    # 进行全身姿态估计
    pose_results = pose.process(frame_rgb)

    # 绘制全身姿态关键点
    # if pose_results.pose_landmarks:
    if pose_results.pose_landmarks and pose_results.pose_landmarks.landmark[
        mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.7:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 获取上肢的关节坐标
        shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        # 在画面上标注关节位置
        cv2.circle(frame, (int(shoulder.x * frame.shape[1]), int(shoulder.y * frame.shape[0])), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(elbow.x * frame.shape[1]), int(elbow.y * frame.shape[0])), 5, (0, 255, 0), -1)
        cv2.circle(frame, (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])), 5, (0, 0, 255), -1)

        if is_valid_landmark(shoulder) and is_valid_landmark(elbow) and is_valid_landmark(wrist):
            print(f"Shoulder: ({shoulder.x}, {shoulder.y}, {shoulder.z})")
            print(f"Elbow: ({elbow.x}, {elbow.y}, {elbow.z})")
            print(f"Wrist: ({wrist.x}, {wrist.y}, {wrist.z})")
        else:
            print("⚠️坐标越界，检测不可靠，跳过本帧")

        # # 打印上肢的3D坐标
        # print(f"Shoulder: ({shoulder.x}, {shoulder.y}, {shoulder.z})")
        # print(f"Elbow: ({elbow.x}, {elbow.y}, {elbow.z})")
        # print(f"Wrist: ({wrist.x}, {wrist.y}, {wrist.z})")

    # 如果检测到手部
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # 绘制手部关键点
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 判断手势是否张开
            if is_hand_open(hand_landmarks):
                cv2.putText(frame, "Hand is Open", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Hand is Fist", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示帧
    cv2.imshow("Hand and Pose Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
