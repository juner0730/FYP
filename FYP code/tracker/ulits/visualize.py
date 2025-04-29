import cv2

def draw_tracks(img, tracks, names=None):
    for t in tracks:
        x, y, w, h = t.tlwh
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cls_id = int(t.cls)
        label = f"{names[cls_id]} ID:{t.track_id}" if names else f"ID:{t.track_id}"

        color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)  # person: green, ball: red
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img
