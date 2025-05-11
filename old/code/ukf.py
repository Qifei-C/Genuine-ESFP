import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import cv2

# 定义运动学模型：过程模型和观测模型
def fx(x, dt):
    # 简单的运动学模型：关节角度和角速度的匀速变化
    theta = x[0]  # 关节角度
    omega = x[1]  # 角速度
    return np.array([theta + omega * dt, omega])  # 更新关节角度和角速度

def hx(x):
    # 假设观测模型是直接测量角度
    return np.array([x[0]])

# 初始化UKF
dt = 0.1  # 时间步长
state_dim = 2  # 关节角度和角速度
measurement_dim = 1  # 关节角度

# 定义UKF
sigma_points = MerweScaledSigmaPoints(state_dim, alpha=0.1, beta=2., kappa=0)
ukf = UKF(dim_x=state_dim, dim_z=measurement_dim, fx=fx, hx=hx, points=sigma_points)

# 初始状态：[关节角度，角速度]
ukf.x = np.array([0, 0])  # 假设初始关节角度为0，角速度为0
# 初始协方差
ukf.P = np.eye(state_dim) * 0.1

# 过程噪声和观测噪声协方差
ukf.Q = np.array([[0.01, 0], [0, 0.01]])  # 过程噪声
ukf.R = np.array([[0.1]])  # 观测噪声

# 臂长（上臂和前臂的长度）
upper_arm_length = 0.3  # 上臂长度 (m)
forearm_length = 0.4   # 前臂长度 (m)

# 计算前向运动学，给定关节角度计算手部位置
def forward_kinematics(theta1, theta2):
    # 计算肩部到肘部的位置（简单2D情况，假设在平面内）
    shoulder_to_elbow = np.array([upper_arm_length * np.cos(theta1), upper_arm_length * np.sin(theta1)])
    elbow_to_hand = np.array([forearm_length * np.cos(theta2), forearm_length * np.sin(theta2)])
    
    # 计算手部在空间中的位置
    hand_position = shoulder_to_elbow + elbow_to_hand
    return hand_position

# 模拟观测数据（通过视觉系统获得的关节角度）
def get_observation():
    # 模拟获取的观测数据，通常来自相机或其他传感器
    return np.array([ukf.x[0] + np.random.normal(0, 0.1)])  # 假设关节角度加噪声

# 运行UKF进行估计和前向运动学计算手部位置
for t in range(100):
    # 获取观测数据（模拟）
    z = get_observation()

    # 更新UKF
    ukf.predict(dt=dt)
    ukf.update(z)

    # 打印估计的关节角度
    print(f"Time step {t}: Estimated angle = {ukf.x[0]}")

    # 使用前向运动学计算手部位置
    hand_position = forward_kinematics(ukf.x[0], ukf.x[1])
    print(f"Hand position at step {t}: {hand_position}")
