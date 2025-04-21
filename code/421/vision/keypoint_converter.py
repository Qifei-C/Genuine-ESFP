import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def extract_coco_17_from_mediapipe(pose_landmarks):
    keypoints = np.zeros((17, 2), dtype=np.float32)
    if pose_landmarks is None:
        return keypoints

    keypoints[5] = [pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                    pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    keypoints[6] = [pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                    pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
    keypoints[7] = [pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                    pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y]
    keypoints[8] = [pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                    pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
    keypoints[9] = [pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,
                   pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y]
    keypoints[10] = [pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                    pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y]

    return keypoints