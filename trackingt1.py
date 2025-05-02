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

    # 打開一個文件來記錄檢測結果
    log_file = open("detection_log.txt", "w")
    
    for frame_id, (path, img, im0, _) in enumerate(dataset):
        img_tensor = torch.from_numpy(img).to(device)
        img_tensor = img_tensor.half() if device.type != 'cpu' else img_tensor.float()
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor)[0]
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)[0]

        # 記錄檢測結果
        log_file.write(f"Frame {frame_id}:\n")
        
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], im0.shape).round()
            dets = pred.cpu().numpy()
            log_file.write(f"  Detections: {len(dets)}\n")
            for i, det in enumerate(dets):
                cls = int(det[5])
                conf = det[4]
                bbox = det[:4]
                log_file.write(f"    Det {i}: class={cls}, conf={conf:.4f}, bbox={bbox}\n")
        else:
            dets = []
            log_file.write("  No detections\n")

        online_targets = tracker.update(dets, im0.shape, img_tensor.shape)
        log_file.write(f"  Tracks: {len(online_targets)}\n")
        for t in online_targets:
            log_file.write(f"    Track {t.track_id}: class={t.cls}, bbox={t.tlbr}\n")
        log_file.write("\n")
        
        # 繪製檢測框和跟蹤ID
        im0_with_dets = im0.copy()
        for det in dets:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(cls)
            label = f"{opt.names[cls]}: {conf:.2f}"
            cv2.rectangle(im0_with_dets, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im0_with_dets, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 繪製跟蹤結果
        im0 = draw_tracks(im0, online_targets, names=opt.names)

        # 顯示檢測和跟蹤結果
        combined = np.hstack((im0_with_dets, im0))
        cv2.imshow("Detection (left) vs Tracking (right)", combined)
        if cv2.waitKey(1) == ord('q'):
            break

    log_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='old_data_Add_Xian_Taiyuan4CAM_finetune_10Label_new_Data0713_new.pt')
    parser.add_argument('--source', type=str, default='football.mp4')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.3)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--track_thresh', type=float, default=0.5)
    parser.add_argument('--track_buffer', type=int, default=30)
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 1])  # person + ball (adjusted based on your labels)
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--names', type=list, default=[ "person", "sports ball"])
    opt = parser.parse_args()

    run_tracking(opt)