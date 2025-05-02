"""
    1. 建立 detection 必須先判斷是否為同一些人，確認沒重複，在放入 queue
    2. detection, steady
    3. detection, lost
"""

import math
import numpy as np

from transform import xywh2mapxy, xywh2feetxy, bboxc2xywh

class PlayerState(object):
    Steady = 0
    Lost = 1

class Player(object):
    def __init__(self, cfg, xywh, camera_idx, label, cameras, pd):
        self.cfg = cfg
        self.xywh = xywh
        mapxy = xywh2mapxy(xywh, cameras[camera_idx].homo_matrix)
        self.mapxy = mapxy[0]
        self.camera_idx = camera_idx
        self.cameras = cameras
        self.steady_count, self.lost_count = 0, 0
        self.group = label
        self.speed = None
        self.avg_speed = None
        self.max_speed = 0

        self.detect_players = {}
        self.pd = pd

        self.traces = []
        self.feetxys = [] # Queue -> speed -> ddxddy
        self.record_all_feetxy = []
        self.speeds = []


    def update(self, xywh, camera_idx, label):
        self.xywh = xywh
        mapxy = xywh2mapxy(xywh, self.cameras[camera_idx].homo_matrix)
        self.mapxy = mapxy[0]
        self.camera_idx = camera_idx
        self.steady_count += 1
        self.group = label
        self.add_traces()
        self.add_record(self.mapxy)

    def to_steady(self, xywh, camera_idx, label):
        self.update(xywh, camera_idx, label)
        self.state = PlayerState.Steady
        self.lost_count = 0

        # TODO: number

    def to_lost(self):
        self.state = PlayerState.Lost
        self.steady_count = 0
        self.lost_count += 1
        self.add_record([-1,-1])

    def add_traces(self):
        self.traces.append([self.camera_idx, self.xywh, self.mapxy, self.group])

        if len(self.traces) > self.cfg.LEN.PlayerTraceMax:
            self.traces.pop(0)

    def add_record(self, xy):
        self.feetxys.append(xy)
        self.record_all_feetxy.append(xy)

        if len(self.feetxys) > self.cfg.LEN.RecordRange:
            self.feetxys.pop(0)

        self.calculate_speed()
        self.calculate_acceleration()

    def get_trace(self):
        return self.traces

    def get_record(self):
        return self.record_all_feetxy

    def calculate_speed(self):
        if len(self.feetxys) == self.cfg.LEN.RecordRange:
            xy1, xy2 = self.feetxys[0], self.feetxys[-1]
            if ((xy1[0] != -1) and (xy1[1] != -1)) or ((xy2[0] != -1) and (xy2[1] != -1)):
                distancex = abs(xy1[0]-xy2[0]) / self.cfg.FIELD.STRATEGY.W * self.cfg.FIELD.ORI.W
                distancey = abs(xy1[1]-xy2[1]) / self.cfg.FIELD.STRATEGY.H * self.cfg.FIELD.ORI.H
                self.speed = math.sqrt(distancex**2 + distancey**2) / (self.cfg.LEN.RecordRange/self.cfg.OUT_VIDEO.FPS)*0.001*3600 # km/hr
                self.speeds.append(self.speed)

            self.avg_speed = np.mean(np.array(self.speeds))
            if self.max_speed < self.speed:
                self.max_speed = self.speed

    def calculate_acceleration(self):
        if len(self.speeds) == self.cfg.LEN.RecordRange:
            speed1, speed2 = self.speeds[0], self.speeds[-1]
            self.acceleration = abs(speed1-speed2) / self.cfg.LEN.RecordRange







