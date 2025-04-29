import cv2
import numpy as np
from pathlib import Path
from tracker.byte_tracker import BYTETracker
from tracker.ulits.visualize import draw_tracks

def xywh2xyxy(x):
    # 将YOLO格式(x_center,y_center,w,h)转换为(x1,y1,x2,y2)
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y

def run_tracking(opt):
    detections_dir = Path(opt.detections_dir)
    tracker = BYTETracker(opt, frame_rate=opt.fps)

    cap = cv2.VideoCapture(opt.source)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 读取检测结果 (YOLO格式的.txt文件)
        det_path = detections_dir / f"{Path(opt.source).stem}_{frame_id}.txt"
        if det_path.exists():
            dets = np.loadtxt(str(det_path), delimiter=' ', ndmin=2)
            if dets.size > 0:
                # 转换格式: (cls, x_center, y_center, w, h, conf) -> (x1, y1, x2, y2, conf, cls)
                dets[:, 1:5] = xywh2xyxy(dets[:, 1:5])
                dets = dets[:, [1, 2, 3, 4, 5, 0]]  # 重新排列列顺序
        else:
            dets = np.empty((0, 6))

        # 用ByteTrack更新
        online_targets = tracker.update(dets, frame.shape[:2], frame.shape[:2])
        frame = draw_tracks(frame, online_targets, names=opt.names)

        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) == ord('q'):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='输入视频路径')
    parser.add_argument('--detections-dir', type=str, required=True, help='检测结果目录')
    parser.add_argument('--fps', type=int, default=30, help='视频帧率')
    parser.add_argument('--track_thresh', type=float, default=0.5, help='追踪阈值')
    parser.add_argument('--track_buffer', type=int, default=30, help='追踪缓冲区大小')
    parser.add_argument('--names', nargs='+', default=['person', 'ball'], help='类别名称')

    opt = parser.parse_args()
    
    # 确保路径中的中文能正确处理
    opt.source = str(Path(opt.source).resolve())
    opt.detections_dir = str(Path(opt.detections_dir).resolve())

    run_tracking(opt)