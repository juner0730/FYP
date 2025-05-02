import cv2
import torch
import numpy as np
from pathlib import Path
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from tracker.byte_tracker import BYTETracker
from tracker.ulits.visualize import draw_tracks


def run_tracking(opt):
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)
    model.eval()
    model.to(device)

    if device.type != 'cpu':
        model.half()  # to FP16

    tracker = BYTETracker(opt, frame_rate=opt.fps)

    dataset = LoadImages(opt.source, img_size=opt.imgsz, stride=32)

    for frame_id, (path, img, im0, _) in enumerate(dataset):
        img_tensor = torch.from_numpy(img).to(device)
        img_tensor = img_tensor.half() if device.type != 'cpu' else img_tensor.float()
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor)[0]
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)[0]

        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], im0.shape).round()
            dets = pred.cpu().numpy()
        else:
            dets = []

        online_targets = tracker.update(dets, im0.shape, img_tensor.shape)

        im0 = draw_tracks(im0, online_targets, names=opt.names)

        cv2.imshow("Tracking", im0)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov7.pt')
    parser.add_argument('--source', type=str, default='football.mp4')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.3)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--track_thresh', type=float, default=0.5)
    parser.add_argument('--track_buffer', type=int, default=30)
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 32])  # person + sports ball
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--names', type=list, default=[
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
        "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ])
    opt = parser.parse_args()

    run_tracking(opt)
