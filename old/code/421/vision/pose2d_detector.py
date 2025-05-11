import mediapipe as mp

class Pose2DDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def detect(self, frame_rgb):
        pose_results = self.pose.process(frame_rgb)
        hand_results = self.hands.process(frame_rgb)
        return pose_results.pose_landmarks, hand_results.multi_hand_landmarks