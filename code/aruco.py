import cv2
import cv2.aruco as aruco
import numpy as np

def draw_aruco_marker_manual(dictionary, marker_id, pixel_size):
    """
    手动生成 ArUco Marker 图像：
    从 dictionary.bytesList 中取出 marker_id 对应的二进制数据，
    解包后只使用前 markerSize*markerSize 个比特生成 Marker 图像，
    然后使用最近邻插值放大到 pixel_size。
    """
    # 获取指定 marker_id 的二进制数据，注意 bytesList 的形状通常为 (numMarkers, 1, N)
    marker_bytes = dictionary.bytesList[marker_id, 0]
    # 解包比特数据
    marker_bits = np.unpackbits(marker_bytes)
    # 使用字典中指定的 markerSize（对于 DICT_4X4_50，markerSize=4）
    n = int(dictionary.markerSize)
    # 只使用前 n*n 个比特作为 Marker 数据
    marker_bits = marker_bits[: n * n]
    # 重塑为 n x n 的矩阵，并转换成 0/255 的图像
    marker_img_small = (marker_bits.reshape((n, n)) * 255).astype(np.uint8)
    # 使用最近邻插值将图像放大到指定尺寸
    marker_img = cv2.resize(marker_img_small, (pixel_size, pixel_size), interpolation=cv2.INTER_NEAREST)
    return marker_img

# 获取预定义的 ArUco 字典（4x4_50）
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# 定义 Marker 的显示尺寸（单位：像素）
marker_display_size = 200

# 要生成的 Marker ID 列表（四个角标）
marker_ids = [0, 1, 2, 3]

# 用于存储生成的 Marker 图像
markers = []

# 单独生成并保存每个角标图像
for m_id in marker_ids:
    marker_img = draw_aruco_marker_manual(aruco_dict, m_id, marker_display_size)
    markers.append(marker_img)
    filename = f"aruco_marker_{m_id}.png"
    cv2.imwrite(filename, marker_img)
    print(f"Marker ID {m_id} 已保存为 {filename}")

# 另外，生成一个包含所有角标的拼贴图像（2x2）
big_size = marker_display_size * 2
big_img = np.ones((big_size, big_size), dtype=np.uint8) * 255

# 将四个 Marker 分别放置在拼贴图的四个角
big_img[0:marker_display_size, 0:marker_display_size] = markers[0]                  # 左上
big_img[0:marker_display_size, marker_display_size:big_size] = markers[1]             # 右上
big_img[marker_display_size:big_size, 0:marker_display_size] = markers[2]             # 左下
big_img[marker_display_size:big_size, marker_display_size:big_size] = markers[3]        # 右下

cv2.imwrite("4_aruco_markers_collage.png", big_img)
print("所有角标拼贴图已保存为 4_aruco_markers_collage.png")