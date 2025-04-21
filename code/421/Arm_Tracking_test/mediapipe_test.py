import cv2
import mediapipe as mp

# —— Initialize MediaPipe Pose module ——
mp_pose    = mp.solutions.pose
pose       = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# —— Open Camera ——
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("WARNING: Unable to open camera.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # BGR → RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 送入模型
    results = pose.process(img_rgb)

    # 如果检测到人体姿态
    if results.pose_landmarks:
        # 在原图上绘制 33 个关键点和骨架连线
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow('Pose Tracking Simple', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
