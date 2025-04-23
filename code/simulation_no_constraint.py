import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from scipy.spatial.transform import Rotation as R

# 定义关节运动学模型
def arm_dynamics(y, t, acc_input):
    n = len(acc_input)  # 关节数量
    dydt = np.zeros(2 * n)  # 返回的微分结果（角速度和角加速度）
    
    for i in range(n):
        dydt[i] = y[i + n]  # 角速度是角度的导数
        dydt[i + n] = acc_input[i]  # 角加速度是输入
        
    return dydt

# 计算每个关节的3D位置
def calculate_joint_positions(angles, link_lengths):
    n = len(angles)
    positions = np.zeros((n, 3))  # 存储每个关节的3D位置

    # 肩部位置固定在原点 (0, 0, 0)
    current_position = np.array([0, 0, 0])  
    current_rotation = R.from_euler('z', angles[0], degrees=False)  # 初始关节绕Z轴旋转

    # 肩部到肘部（固定连接）
    positions[0] = current_position  # 肩部位置（原点）
    positions[1] = current_position + current_rotation.apply([link_lengths[0], 0, 0])  # 肩部到肘部
    current_position = positions[1]

    # 肘部到腕部（固定连接）
    current_rotation = current_rotation * R.from_euler('z', angles[1], degrees=False)  # 更新角度
    positions[2] = current_position + current_rotation.apply([link_lengths[1], 0, 0])  # 从肘部到腕部
    current_position = positions[2]

    # 腕部到手掌末端（固定连接）
    current_rotation = current_rotation * R.from_euler('z', angles[2], degrees=False)  # 更新角度
    positions[3] = current_position + current_rotation.apply([link_lengths[2], 0, 0])  # 从腕部到手掌

    return positions


# 初始化角度、角速度和加速度
n_joints = 4  # 4个关节（肩部、肘部、腕部、手掌）
initial_angles = np.zeros(n_joints)  # 初始角度
initial_angular_velocities = np.zeros(n_joints)  # 初始角速度
initial_conditions = np.concatenate([initial_angles, initial_angular_velocities])

# 假设的角加速度输入（可以根据需要进行修改）
angular_accelerations = np.array([1, 2, 5, 0])  # 每个关节的角加速度

# 假设每个关节之间的臂段长度（可以根据需要修改）
link_lengths = np.array([0.5, 0.5, 0.4, 0.3])  # 臂段的长度（上臂、前臂、手部简化为杆）

# 时间范围
time = np.linspace(0, 10, 1000)

# 求解微分方程
solution = odeint(arm_dynamics, initial_conditions, time, args=(angular_accelerations,))

# 解析结果
angles = solution[:, :n_joints]  # 角度
angular_velocities = solution[:, n_joints:]  # 角速度

# 计算每个关节的3D位置
all_positions = []
for i in range(len(time)):
    positions = calculate_joint_positions(angles[i], link_lengths)
    all_positions.append(positions)

all_positions = np.array(all_positions)

# 创建动画
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([0, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Live 3D Arm Pose')

# 绘制手臂每个关节的线段
lines = [ax.plot([], [], [], lw=2)[0] for _ in range(n_joints)]

# 绘制每个关节的球形表示
scatters = [ax.scatter([], [], [], s=100) for _ in range(n_joints)]

# 更新函数
def update(frame):
    positions = all_positions[frame]
    
    # 更新每个关节的坐标
    for i in range(n_joints):
        # 确保每个关节位置是3D坐标
        x, y, z = positions[i]
        
        # 更新关节的球体位置
        scatters[i]._offsets3d = ([x], [y], [z])  # 应该传入列表或数组
        
        if i > 0:  # 只绘制连接部分
            # 确保连线的端点为3D坐标
            x_data = [positions[i-1, 0], positions[i, 0]]
            y_data = [positions[i-1, 1], positions[i, 1]]
            z_data = [positions[i-1, 2], positions[i, 2]]
            lines[i].set_data(x_data, y_data)
            lines[i].set_3d_properties(z_data)

    return lines + scatters

# 创建动画
ani = FuncAnimation(fig, update, frames=len(time), interval=50, blit=False)

plt.show()