import argparse
import cv2
import glob
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from collections import deque
from threading import Thread
import time

from depth_anything_v2.dpt import DepthAnythingV2

cap = cv2.VideoCapture("udp://0.0.0.0:5000", cv2.CAP_FFMPEG)


DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

depth_anything = DepthAnythingV2(**model_configs['vitb'])
depth_anything.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitb.pth', map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()


depth_history = deque(maxlen=500)
timestamps = deque(maxlen=500)

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-')
ax.set_title("Center Depth Over Time")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Depth")
ax.set_ylim(0, 10)

start_time = time.time()

def update_plot():
    line.set_data(timestamps, depth_history)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.0001)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        continue


    depth_map = depth_anything.infer_image(frame, 256)

    h, w = depth_map.shape
    center_depth = depth_map[h // 2, w // 2]
    elapsed = time.time() - start_time
    depth_history.append(center_depth)
    timestamps.append(elapsed)

    # Resize depth map for display
    depth_display = (depth_map * 255).astype(np.uint8)
    depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_MAGMA)

    # Get center coordinates (same for both frames)
    ch, cw = frame.shape[:2]
    center_x, center_y = cw // 2, ch // 2

    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

    dh, dw = depth_display.shape[:2]
    dx, dy = dw // 2, dh // 2
    cv2.circle(depth_display, (dx, dy), 5, (0, 255, 0), -1) 

    cv2.imshow("Camera Stream", frame)
    cv2.imshow("Depth Map", depth_display)

    update_plot()

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()