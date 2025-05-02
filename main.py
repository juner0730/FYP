import cv2
import numpy as np
from pathlib import Path
import argparse
from player_v2 import PlayerState, Tracker, DetectPlayer
from ball import Ball, BallState, DetectBall, ball_tracking
from transform import xywh2mapxy, xywh2feetxy, bboxc2xywh
from tracker.byte_tracker import BYTETracker
from tracker.ulits.visualize import draw_tracks
from tracker.ulits.geometry import xywh2xyxy, xywh2mapxy

class DetectionLoader:
    def __init__(self, detections_dir, class_names):
        self.detections_dir = Path(detections_dir)
        self.class_names = class_names
        self.detection_files = sorted(self.detections_dir.glob('*.txt'))
        
    def load_detections(self, frame_idx):
        if frame_idx >= len(self.detection_files):
            return np.empty((0, 6))
        
        file_path = self.detection_files[frame_idx]
        detections = []
        
        with open(file_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    conf = float(parts[5]) if len(parts) > 5 else 1.0
                    
                    x1 = x_center - width/2
                    y1 = y_center - height/2
                    x2 = x_center + width/2
                    y2 = y_center + height/2
                    
                    detections.append([x1, y1, x2, y2, conf, class_id])
        
        return np.array(detections) if detections else np.empty((0, 6))
class TrackingSystem:
    def __init__(self, opt):
        self.opt = opt
        self.byte_tracker = BYTETracker(opt, frame_rate=opt.fps)
        self.players = []
        self.football = None
        self.frame_count = 0
        self.camera_idx = 0  # 假設單一攝像頭
        self.next_player_id = 1
        
        # 初始化球場配置
        self.field_config = {
            'STRATEGY': {'W': 100, 'H': 100},  # 策略地圖尺寸
            'ORI': {'W': 68, 'H': 105},         # 標準足球場尺寸(米)
            'LEN': {
                'PlayerTraceMax': 30,           # 球員軌跡最大長度
                'RecordRange': 10,              # 速度計算範圍
                'BallTraceMax': 30              # 球軌跡最大長度
            },
            'OUT_VIDEO': {'FPS': opt.fps},      # 輸出視頻幀率
            'THRESHOLD': {
                'Same_Person_distance': 3.0,    # 同一球員匹配閾值(米)
                'Ball_Hold_Distance': 2.0       # 持球距離閾值(米)
            }
        }
        
        # 初始化homography矩陣 (需替換為實際值)
        self.homography_matrix = self.load_homography_matrix()
        
        # 檢測類別映射
        self.class_names = opt.names if opt.names else ['person', 'ball']
    
    def load_homography_matrix(self):
        """
        加載homography矩陣
        實際應用中應從校準文件加載
        """
        # 示例矩陣 - 需替換為實際校準值
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
    
    def update_players(self, detections):
        """更新球員追蹤狀態"""
        current_detections = []
        
        # 處理球員檢測 (class 0)
        player_dets = [d for d in detections if int(d[5]) == self.class_names.index('person')]
        for det in player_dets:
            xywh = det[:4]
            conf = det[4]
            current_detections.append({
                'xywh': xywh,
                'conf': conf,
                'group': 0 if det[0] < 640 else 1  # 簡單分組邏輯
            })
        
        # 更新現有球員 - 使用距離匹配
        updated_players = []
        unmatched_indices = list(range(len(current_detections)))
        
        for player in self.players:
            if player.state == PlayerState.Lost and player.lost_count > 30:
                continue  # 移除丟失太久的球員
                
            # 計算所有可能的匹配
            best_match_idx = None
            min_distance = float('inf')
            
            for i in unmatched_indices:
                det = current_detections[i]
                try:
                    mapxy = xywh2mapxy(det['xywh'], self.homography_matrix)[0]
                    distance = np.linalg.norm(np.array(player.mapxy) - np.array(mapxy))
                    
                    if distance < min_distance and distance < self.field_config['THRESHOLD']['Same_Person_distance']:
                        min_distance = distance
                        best_match_idx = i
                except Exception as e:
                    print(f"匹配計算錯誤: {e}")
                    continue
            
            if best_match_idx is not None:
                # 找到最佳匹配
                best_det = current_detections[best_match_idx]
                
                # 創建DetectPlayer對象
                detect_player = DetectPlayer(
                    cfg=self.field_config,
                    xywh=best_det['xywh'],
                    camera_idx=self.camera_idx,
                    label=best_det['group'],
                    cameras=[{'homo_matrix': self.homography_matrix}],
                    pd={'confidence': best_det['conf']}
                )
                
                player.to_steady(best_det['xywh'], [detect_player])
                unmatched_indices.remove(best_match_idx)
            else:
                # 未找到匹配
                player.to_lost()
            
            updated_players.append(player)
        
        # 為未匹配的檢測創建新球員
        for i in unmatched_indices:
            det = current_detections[i]
            try:
                mapxy = xywh2mapxy(det['xywh'], self.homography_matrix)[0]
                
                detect_player = DetectPlayer(
                    cfg=self.field_config,
                    xywh=det['xywh'],
                    camera_idx=self.camera_idx,
                    label=det['group'],
                    cameras=[{'homo_matrix': self.homography_matrix}],
                    pd={'confidence': det['conf']}
                )
                
                new_player = Tracker(
                    cfg=self.field_config,
                    mapxy=mapxy,
                    label=det['group'],
                    number=self.next_player_id,
                    detect_players=[detect_player]
                )
                self.next_player_id += 1
                updated_players.append(new_player)
            except Exception as e:
                print(f"創建新球員錯誤: {e}")
        
        self.players = updated_players
    
    def update_ball(self, detections):
        """更新球追蹤狀態"""
        ball_dets = [d for d in detections if int(d[5]) == self.class_names.index('ball')]
        
        if not ball_dets:
            if self.football:
                self.football.to_lost()
            return
        
        # 取置信度最高的球檢測
        best_ball = max(ball_dets, key=lambda x: x[4])
        ball_xywh = best_ball[:4]
        
        try:
            detect_ball = DetectBall(
                cfg=self.field_config,
                xywh=ball_xywh,
                camera_idx=self.camera_idx,
                cameras=[{'homo_matrix': self.homography_matrix}]
            )
            
            if not self.football:
                self.football = Ball(
                    cfg=self.field_config,
                    mapxy=detect_ball.mapxy,
                    detect_balls=[[detect_ball]],
                    frame_count=self.frame_count
                )
            else:
                self.football.to_in(detect_ball.mapxy, [[detect_ball]], self.frame_count)
                
                # 更新球與球員的互動
                if self.players:
                    self.football = ball_tracking(
                        self.field_config,
                        self.football,
                        [[detect_ball]],
                        self.players,
                        self.frame_count
                    )
        except Exception as e:
            print(f"更新球狀態錯誤: {e}")
    
    def run(self, video_path):
        """主運行循環"""
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 使用BYTETracker進行檢測和追蹤
            detections = self.detect_objects(frame)
            online_targets = self.byte_tracker.update(detections, frame.shape[:2], frame.shape[:2])
            
            # 轉換為標準檢測格式 [x1,y1,x2,y2,conf,cls]
            dets = []
            for t in online_targets:
                dets.append([
                    t.tlwh[0], t.tlwh[1], 
                    t.tlwh[0]+t.tlwh[2], t.tlwh[1]+t.tlwh[3],
                    t.score, t.cls
                ])
            dets = np.array(dets) if dets else np.empty((0, 6))
            
            # 更新追蹤狀態
            self.update_players(dets)
            self.update_ball(dets)
            
            # 可視化
            frame = self.visualize(frame, online_targets)
            
            # 顯示幀號和追蹤信息
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Players: {len([p for p in self.players if p.state == PlayerState.Steady])}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Tracking System', frame)
            if cv2.waitKey(1) == ord('q'):
                break
            
            self.frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
    
    def detect_objects(self, frame):
        """
        物體檢測方法
        實際應用中應替換為YOLO等檢測器
        """
        # 這裡返回空數組，實際應從文件或模型加載檢測結果
        return np.empty((0, 6))
    
    def visualize(self, frame, tracks):
        """可視化追蹤結果"""
        # 繪製BYTETracker的基礎追蹤框
        frame = draw_tracks(frame, tracks, names=self.class_names)
        
        # 繪製自定義球員信息
        for player in self.players:
            if player.state == PlayerState.Steady:
                x, y = int(player.mapxy[0]), int(player.mapxy[1])
                color = (0, 255, 0) if player.group == 0 else (255, 0, 0)  # 不同隊伍不同顏色
                
                cv2.circle(frame, (x, y), 8, color, -1)
                cv2.putText(frame, f"{player.number}", (x+10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # 顯示速度信息
                if hasattr(player, 'speed') and player.speed is not None:
                    cv2.putText(frame, f"{player.speed:.1f}km/h", (x-20, y-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 繪製球信息
        if self.football and self.football.state == BallState.InField:
            x, y = int(self.football.mapxy[0]), int(self.football.mapxy[1])
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
            
            # 顯示持球信息
            if self.football.hold_id:
                holder = next((p for p in self.players if p.number == self.football.hold_id), None)
                if holder:
                    hx, hy = int(holder.mapxy[0]), int(holder.mapxy[1])
                    cv2.line(frame, (x, y), (hx, hy), (0, 255, 255), 2)
                    cv2.putText(frame, f"Hold by {self.football.hold_id}", (x+10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame

def run_tracking(opt):
    tracking_system = TrackingSystem(opt)
    tracking_system.run(opt.source)

class SoccerTracker:
    def __init__(self, opt):
        self.opt = opt
        self.tracker = BYTETracker(opt, frame_rate=opt.fps)
        self.detection_loader = DetectionLoader(opt.detections_dir, opt.names)
        
        # 场地配置
        self.homography_matrix = np.eye(3)  # 替换为实际homography矩阵
        self.frame_count = 0
    
    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 加载检测结果
            detections = self.detection_loader.load_detections(self.frame_count)
            
            # 执行跟踪
            tracks = self.tracker.update(detections, frame.shape[:2], frame.shape[:2])
            
            # 可视化
            frame = draw_tracks(frame, tracks, names=self.opt.names)
            
            # 显示信息
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Soccer Tracking', frame)
            if cv2.waitKey(1) == ord('q'):
                break
            
            self.frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='视频文件路径')
    parser.add_argument('--detections-dir', type=str, required=True, help='检测结果目录')
    parser.add_argument('--fps', type=int, default=30, help='视频帧率')
    parser.add_argument('--track-thresh', type=float, default=0.5, help='跟踪置信度阈值')
    parser.add_argument('--track-buffer', type=int, default=30, help='跟踪缓冲区大小')
    parser.add_argument('--names', nargs='+', default=['player', 'ball'], help='类别名称')
    
    opt = parser.parse_args()
    tracker = SoccerTracker(opt)
    tracker.run(opt.source)