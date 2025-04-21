# vision/pose3d_estimator.py

import torch
import numpy as np
from collections import deque
from common.model import TemporalModel
from common.utils import wrap


class Pose3DEstimator:
    def __init__(self, checkpoint_path: str):
        # 1) 构建与 checkpoint 匹配的模型结构
        self.model = TemporalModel(
            num_joints_in=17,
            in_features=2,
            num_joints_out=17,
            filter_widths=[3, 3, 3, 3],  # 对应官方预训练的 8 层卷积
            causal=False,
            dropout=0.25,
            channels=1024,
            dense=False
        )

        # 2) 加载 checkpoint 中的 model_pos，过滤多余键
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
        sd = checkpoint['model_pos']
        model_dict = self.model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in model_dict}
        self.model.load_state_dict(filtered, strict=False)
        self.model.eval()

        # 3) 自动获取模型需要的最小序列长度（感受野）
        self.seq_len = self.model.receptive_field()

        # 4) 初始化一个循环缓冲，保存最近 seq_len 帧的 2D 关键点
        self.buffer = deque(maxlen=self.seq_len)

    def estimate(self, keypoints_2d: np.ndarray) -> np.ndarray:
        """
        从单帧 2D 关键点出发，输出这一帧的 3D 姿态。

        keypoints_2d: shape (17, 2)，COCO格式归一化2D关键点
        return: pred_3d, shape (17, 3)，对应的3D关键点
        """

        # 1) 把当前帧 keypoints 加入缓冲
        self.buffer.append(keypoints_2d)

        # 2) 如果缓冲不到 seq_len，则前面用第一帧填充
        if len(self.buffer) < self.seq_len:
            padding = [self.buffer[0]] * (self.seq_len - len(self.buffer))
            seq = padding + list(self.buffer)
        else:
            seq = list(self.buffer)

        # 3) 构造模型输入张量 (1, seq_len, 17, 2)
        x = torch.from_numpy(np.stack(seq, axis=0)).unsqueeze(0).float()

        # 4) 前向推理
        with torch.no_grad():
            out = self.model(x)  # shape (1, seq_len, 17, 3)

        # 5) 取缓冲中最新帧（即序列最后一帧）的预测结果
        pred_3d = out[:, -1].squeeze(0).numpy()  # (17,3)

        return pred_3d
