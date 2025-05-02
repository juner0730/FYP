import numpy as np
import cv2

def xywh2xyxy(x):
    """将 [x_center, y_center, width, height] 转换为 [x1, y1, x2, y2]"""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y

def xyxy2xywh(x):
    """将 [x1, y1, x2, y2] 转换为 [x_center, y_center, width, height]"""
    y = np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x_center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y_center
    y[..., 2] = x[..., 2] - x[..., 0]       # width
    y[..., 3] = x[..., 3] - x[..., 1]       # height
    return y

def xywh2mapxy(xywh, homo_matrix):
    """将检测框坐标映射到场地平面"""
    if len(xywh.shape) == 1:
        xywh = np.array([xywh])
    
    # 转换为脚部中心点 (假设脚部在框底部中心)
    feet_points = np.column_stack((
        xywh[:, 0],               # x_center
        xywh[:, 1] + xywh[:, 3]/2  # y_bottom
    ))
    
    # 添加齐次坐标
    feet_points_homo = np.column_stack((
        feet_points,
        np.ones(len(feet_points))
    ))
    
    # 应用homography变换
    mapped_points = np.dot(homo_matrix, feet_points_homo.T).T
    mapped_points = mapped_points[:, :2] / mapped_points[:, 2, np.newaxis]
    
    return mapped_points if len(mapped_points) > 1 else mapped_points[0]

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0]+box1[2], box2[0]+box2[2])
    y2 = min(box1[1]+box1[3], box2[1]+box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    
    return inter_area / (box1_area + box2_area - inter_area + 1e-6)