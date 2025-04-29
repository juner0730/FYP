import numpy as np
from tracker.kalman_filter import KalmanFilter
from collections import deque
import cv2

class Track:
    def __init__(self, tlwh, score, cls, track_id):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.track_id = track_id
        self.score = score
        self.cls = cls
        self.hits = 1
        self.time_since_update = 0
        self.kalman_filter = KalmanFilter()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(tlwh))
    
    def predict(self):
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        xyah = self.mean[:4]
        self.tlwh = self.xyah_to_tlwh(xyah)

    def update(self, detection):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(detection[:4])
        )
        self.tlwh = self.xyah_to_tlwh(self.mean)
        self.score = detection[4]
        self.cls = detection[5]
        self.hits += 1
        self.time_since_update = 0

    @staticmethod
    def tlwh_to_xyah(tlwh):
        x, y, w, h = tlwh
        return np.array([x + w / 2, y + h / 2, w / h, h])

    @staticmethod
    def xyah_to_tlwh(xyah):
        cx, cy, ratio, h = xyah
        w = ratio * h
        return np.array([cx - w / 2, cy - h / 2, w, h])


class BYTETracker:
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        self.tracks = []
        self.frame_id = 0
        self.track_id_count = 0
        self.track_thresh = opt.track_thresh
        self.track_buffer = opt.track_buffer
        self.max_time_lost = 30

    def update(self, detections, img_shape, img_tensor_shape):
        self.frame_id += 1

        active_tracks = []
        new_detections = []

        # 預測現有軌跡
        for track in self.tracks:
            track.predict()
            track.time_since_update += 1

        # 檢查偵測結果格式
        for det in detections:
            if det[4] >= self.track_thresh:
                new_detections.append(det)

        # 匹配並更新軌跡
        for det in new_detections:
            matched = False
            for track in self.tracks:
                iou = self._iou(track.tlwh, det[:4])
                if iou > 0.3 and track.time_since_update < self.track_buffer:
                    track.update(det)
                    matched = True
                    break
            if not matched:
                self.track_id_count += 1
                new_track = Track(det[:4], det[4], det[5], self.track_id_count)
                active_tracks.append(new_track)

        # 刪除失效軌跡
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.track_buffer]
        self.tracks.extend(active_tracks)

        return self.tracks

    @staticmethod
    def _iou(box1, box2):
        x1, y1, x2, y2 = *box1[:2], box1[0]+box1[2], box1[1]+box1[3]
        x3, y3, x4, y4 = *box2[:2], box2[2], box2[3]

        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0
