
import math
import numpy as np

from transform import xywh2mapxy
from player import PlayerState

class BallState(object):
    Undefined = 0
    InField = 1


class DetectBall(object):
    def __init__(self, cfg, xywh, camera_idx, cameras):
        self.cfg = cfg
        self.xywh = xywh
        mapxy = xywh2mapxy(xywh, cameras[camera_idx].homo_matrix)
        self.mapxy = mapxy[0]
        self.camera_idx = camera_idx
        self.group = "ball"

class Ball(object):
    def __init__(self, cfg, mapxy, detect_balls, frame_count):
        self.state = BallState.InField
        self.cfg = cfg
        self.group = "ball"
        self.detect_balls = {}
        self.last_appear_count = 0
        self.in_count = 0
        self.lost_count = 0
        # self.highest = 0
        # self.hold_count = 5
        self.pass_count = 0

        self.distance = 0

        self.number_of_consecutive_passes = {
            0:{
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0,
                8: 0,
                9: 0,
                10: 0,
                11: 0

            },
            1:{
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0,
                8: 0,
                9: 0,
                10: 0,
                11: 0
            }

        }
        self.hold_id = None
        self.hold_group = None

        self.traces = []
        self.to_in(mapxy, detect_balls, frame_count)

        ###
        self.disturbed = False
        self.intercept_num = {
            0: 0,
            1: 0
        }
        ###
    def to_lost(self):
        self.state = BallState.Undefined
        self.lost_count += 1
        self.in_count = 0
        self.speed = 0
        # if self.traces:
        #     self.traces.pop(0)

    def camera_detect_balls(self, detect_balls):
        for ball in detect_balls:
            if ball.camera_idx not in self.detect_balls:
                self.detect_balls[ball.camera_idx] = {"trace": []}
            self.detect_balls[ball.camera_idx]["trace"].append([ball.xywh])

            if len(self.detect_balls[ball.camera_idx]["trace"]) > self.cfg.LEN.BallTraceMax:
                self.detect_balls[ball.camera_idx]["trace"].pop(0)

    def update(self, mapxy, detect_balls, frame_count):
        self.mapxy = mapxy
        self.in_count += 1
        self.last_appear_count = frame_count
        self.camera_detect_balls(detect_balls)
        # self.add_trace(self.mapxy)

    def to_in(self, mapxy, detect_balls, frame_count):
        self.state = BallState.InField
        self.lost_count = 0
        self.update(mapxy, detect_balls, frame_count)


    def add_trace(self):
        self.traces.append([self.mapxy])

        if len(self.traces) > self.cfg.LEN.BallTraceMax:
            self.traces.pop(0)

    def get_trace(self):
        return self.traces


def first_frame_balls(cfg, detect_balls_set):
    new_ball_set = []

    for balls in detect_balls_set:
        for ball in balls:
            SamePersonFlag = False
            for idx, compare_balls in enumerate(new_ball_set):
                if any(math.sqrt((ball.mapxy[0] - cmp_ball.mapxy[0])**2 + (ball.mapxy[1] - cmp_ball.mapxy[1])**2) < cfg.THRESHOLD.Same_Person_distance for cmp_ball in compare_balls):
                    SamePersonFlag = True
                    new_ball_set[idx].append(ball)
                    break
            if not SamePersonFlag:
                new_ball_set.append([ball])

    return new_ball_set

def ball_tracking(cfg, football, detect_balls, players, frame_count):
    # There is only one ball being tracked

    if football is None: # init
        detect_balls_set = first_frame_balls(cfg, detect_balls)
        for detect_balls in detect_balls_set:
            mapxys = np.array([ball.mapxy for ball in detect_balls])
            avg_mapxy = np.mean(mapxys, axis=0)
            football = Ball(cfg, avg_mapxy, detect_balls, frame_count)
        return football

    camera_detect_ball = []

    for dtc_balls in detect_balls:
        for ball in dtc_balls:
            dx, dy = ball.mapxy[0] - football.mapxy[0], ball.mapxy[1] - football.mapxy[1]
            distance = math.pow((math.pow(dx, 2) + math.pow(dy, 2)), 0.5)
            if (0 < frame_count - football.last_appear_count < 15) and distance >= 50:
                continue
            camera_detect_ball.append(ball)
    if camera_detect_ball == []:
        football.to_lost()
        if football.lost_count >= 100:
            if football.hold_group != None:
                if football.pass_count != 0 and football.pass_count in football.number_of_consecutive_passes[football.hold_group]:
                    football.number_of_consecutive_passes[football.hold_group][football.pass_count] += 1
                    player = next((player for player in players if player.number == football.hold_id), None)
                    player.try_pass += 1
                    player.pass_failure += 1
            print(football.hold_group)
            if football.traces:
                football.traces.pop(0)
            football.hold_group = None
            football.hold_id = None
            football.pass_count = 0

    else:
        mapxys = np.array([ball.mapxy for ball in camera_detect_ball])
        avg_mapxy = np.mean(mapxys, axis=0)
        football.to_in(avg_mapxy, camera_detect_ball, frame_count)

    if football.state == BallState.InField:
        ball_distances = []
        steady_players = []
        player_idx = 0
        for player in players:
            if player.state == PlayerState.Steady:
                dx, dy = player.mapxy[0] - football.mapxy[0], player.mapxy[1] - football.mapxy[1]
                distance = math.pow((math.pow(dx, 2) + math.pow(dy, 2)), 0.5)
                steady_players.append(player)
                ball_distances.append((player_idx, player.number, distance))
                player_idx += 1

        if ball_distances:
            hold_group = None
            hold_id = None
            if football.hold_id != None:
                ball_distances = sorted(ball_distances, key = lambda ball_distances: ball_distances[2])
                ball_distances = np.asarray(ball_distances, dtype=np.float32)
                # np.set_printoptions(suppress=True)

                people_nearby = np.where(ball_distances[:3, 2] < 25)[0]
                print(people_nearby)
                if len(people_nearby) > 1:
                    football.disturbed = True

                last_hold_exist = np.where(ball_distances[:3, 1] == football.hold_id)[0]

                if last_hold_exist.size:
                    hold_idx, _, distance = ball_distances[last_hold_exist[0]]
                    if distance > 45:
                        hold_idx, _, distance = ball_distances[0]
                else:
                    hold_idx, _, distance = ball_distances[0]

                if distance < 15:
                    hold_group = steady_players[int(hold_idx)].group
                    hold_id = steady_players[int(hold_idx)].number
                    football.add_trace()
                    football.distance = distance

            else:
                min_distance_idx = np.array(ball_distances)[:, 2].argmin()
                if ball_distances[min_distance_idx][2] < 15:
                    hold_group = steady_players[ball_distances[min_distance_idx][0]].group
                    hold_id = steady_players[ball_distances[min_distance_idx][0]].number
                    football.add_trace()
                    football.distance = ball_distances[min_distance_idx][2]


            if football.hold_group == None:
                football.hold_group = hold_group
                football.hold_id = hold_id
                return football

            if hold_group != None:
                if football.hold_group != hold_group:
                    if football.disturbed:
                        football.intercept_num[hold_group] += 1
                        football.disturbed = False
                        if football.pass_count != 0 and football.pass_count in football.number_of_consecutive_passes[football.hold_group]:
                            football.number_of_consecutive_passes[football.hold_group][football.pass_count] += 1

                    elif football.pass_count != 0 and football.pass_count in football.number_of_consecutive_passes[football.hold_group]:
                        football.number_of_consecutive_passes[football.hold_group][football.pass_count] += 1
                        player = next((player for player in players if player.number == football.hold_id), None)
                        player.try_pass += 1
                        player.pass_failure += 1

                    football.pass_count = 0
                elif football.hold_group == hold_group and football.hold_id != hold_id:
                    if football.disturbed:
                        football.disturbed = False

                    football.pass_count += 1
                    player = next((player for player in players if player.number == football.hold_id), None)
                    player.try_pass += 1
                    player.pass_success += 1
                football.hold_group = hold_group
                football.hold_id = hold_id
            else:
                football.to_lost()
                if football.lost_count > 300:
                    if football.hold_group != None:
                        if football.pass_count != 0 and football.pass_count in football.number_of_consecutive_passes[football.hold_group]:
                            football.number_of_consecutive_passes[football.hold_group][football.pass_count] += 1
                            player = next((player for player in players if player.number == football.hold_id), None)
                            player.try_pass += 1
                            player.pass_success += 1
                    if football.traces:
                        football.traces.pop(0)
                    football.hold_group = None
                    football.hold_id = None
                    football.pass_count = 0

    return football