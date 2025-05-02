import cv2

def draw_tracks(img, tracks, names=None):
    for t in tracks:
        x, y, w, h = t.tlwh
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cls_id = int(t.cls)
        
        # 健壮性处理
        if names and cls_id < len(names):
            label = f"{names[cls_id]} ID:{t.track_id}"
        else:
            label = f"ID:{t.track_id} (Class:{cls_id})"
        
        # 颜色选择：球员绿色，球红色，其他类别蓝色
        if cls_id == 0:  # 球员
            color = (0, 255, 0)
        elif cls_id == 1:  # 球
            color = (0, 0, 255)
        else:  # 其他类别
            color = (255, 0, 0)
            
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img