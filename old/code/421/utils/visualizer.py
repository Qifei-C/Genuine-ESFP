# utils/visualizer.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 上肢关节在 COCO 中的索引
SHOULDER = 5
ELBOW = 7
WRIST = 9


# 画三点连线函数
def plot_arm_3d(keypoints_3d, ax=None):
    """
    keypoints_3d: np.ndarray, shape (17,3)
    ax: matplotlib 3D 坐标轴，可选
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # 提取三点
    s = keypoints_3d[SHOULDER]
    e = keypoints_3d[ELBOW]
    w = keypoints_3d[WRIST]

    xs = [s[0], e[0], w[0]]
    ys = [s[1], e[1], w[1]]
    zs = [s[2], e[2], w[2]]

    # 画折线
    ax.plot(xs, ys, zs, '-o', linewidth=2, markersize=5)

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Arm Pose')

    # 反转Z轴（可根据相机坐标系调整）
    ax.invert_zaxis()

    plt.show()
