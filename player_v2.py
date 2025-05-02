
import math
import numpy as np

from transform import xywh2mapxy, xywh2feetxy, bboxc2xywh

class PlayerState(object):
    Steady = 0
    Lost = 1

class DetectPlayer(object):
    def __init__(self, cfg, xywh, camera_idx, label, cameras, pd):
        self.cfg = cfg
        self.xywh = xywh
        mapxy = xywh2mapxy(xywh, cameras[camera_idx].homo_matrix)
        self.mapxy = mapxy[0]
        self.camera_idx = camera_idx
        self.group = label
        self.pd = pd

class Tracker(object):
    def __init__(self, cfg, mapxy, label, number, detect_players):
        self.cfg = cfg
        self.steady_count, self.lost_count = 0, 0
        self.speed = None
        self.max_speed = 0
        self.acceleration = None
        self.number = number
        self.detect_players = {}

        self.traces = []
        self.feetxys = [] # Queue -> speed -> ddxddy
        self.record_all_feetxy = []
        self.speeds = []
        self.smooth_speeds = []
        self.group = label
        self.total_distance = 0

        self.try_pass = 0
        self.pass_success = 0
        self.pass_failure = 0

        self.to_steady(mapxy, detect_players)

    def camera_detect_players(self, detect_players):
        for player in detect_players:
            if player.camera_idx not in self.detect_players:
                self.detect_players[player.camera_idx] = {"trace": [], "pd": []}
            self.detect_players[player.camera_idx]["trace"].append([player.xywh, player.group])
            self.detect_players[player.camera_idx]["pd"] = player.pd

            if len(self.detect_players[player.camera_idx]["trace"]) > self.cfg.LEN.RecordRange:
                self.detect_players[player.camera_idx]["trace"].pop(0)

    def update(self, mapxy, detect_players):
        self.mapxy = mapxy
        self.steady_count += 1
        # self.group = label
        self.camera_detect_players(detect_players)
        self.add_field_traces()
        self.add_record(self.mapxy)

    def to_steady(self, mapxy, detect_players):
        self.update(mapxy, detect_players)
        self.state = PlayerState.Steady
        self.lost_count = 0

    def to_lost(self):
        self.state = PlayerState.Lost
        self.steady_count = 0
        self.lost_count += 1
        self.add_record([-1,-1])

    def add_field_traces(self):
        self.traces.append([self.mapxy, self.group])

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
                moving = math.sqrt(distancex**2 + distancey**2)
                if moving < 3:
                    self.total_distance += (moving / 1000) # km
                    speed = (moving / (self.cfg.LEN.RecordRange/self.cfg.OUT_VIDEO.FPS))*0.001*3600 # km/hr
                    self.speeds.append(speed)

                    # self.speed = (moving / (self.cfg.LEN.RecordRange/self.cfg.OUT_VIDEO.FPS))*0.001*3600 # km/hr
                    # self.speeds.append(self.speed)
                    # if self.max_speed < self.speed:
                    #     self.max_speed = self.speed


                    if len(self.speeds) > 3:
                        self.speeds.pop(0)
                    if len(self.speeds) == 3:
                        avg_speed = np.sum(np.array(self.speeds)) / 3
                        self.speed = avg_speed
                        self.smooth_speeds.append(avg_speed)

                        if self.max_speed < self.speed:
                            self.max_speed = self.speed

    def calculate_acceleration(self):
        if len(self.smooth_speeds) == self.cfg.LEN.RecordRange:
            speed1, speed2 = self.smooth_speeds[0], self.smooth_speeds[-1]
            self.acceleration = (speed1-speed2) / self.cfg.LEN.RecordRange





