# track.py
import math  # 用於計算距離
import numpy as np  # 用於數值計算與陣列操作
import lap  # 匈牙利算法 LAPJV，用於最佳匹配
from scipy.spatial.distance import cdist  # 可選距離計算（未使用）

from player import PlayerState  # 玩家狀態的定義

def same_people_process(cfg, players_set):
    new_player_set = []  # 初始化新的玩家群組集合

    for players in players_set:
        for player in players:
            SamePersonFlag = False  # 判斷是否屬於相同人的旗標
            for idx, compare_players in enumerate(new_player_set):
                if any(math.sqrt((player.mapxy[0] - cmp_player.mapxy[0])**2 + 
                                (player.mapxy[1] - cmp_player.mapxy[1])**2) < cfg.THRESHOLD.Same_Person_distance 
                       for cmp_player in compare_players):
                    SamePersonFlag = True
                    new_player_set[idx].append(player)  # 加入現有群組
                    break
            if not SamePersonFlag:
                new_player_set.append([player])  # 新增新的人群組

    return new_player_set

def create_distance_array(players:list, detections:list):
    distance_array = np.zeros((len(players), len(detections)))  # 建立距離矩陣
    for player_count, player in enumerate(players):
        player_mapx, player_mapy = player.mapxy[0], player.mapxy[1]  # 取得玩家位置
        for detection_count, detection in enumerate(detections):
            detection_mapx, detection_mapy = detection.mapxy[0], detection.mapxy[1]  # 偵測位置
            dx, dy = player_mapx - detection_mapx, player_mapy - detection_mapy
            distance = math.hypot(dx, dy)  # 計算歐式距離
            distance_array[player_count, detection_count] = distance
    return distance_array

def get_match(matrix, threshold):
    _, x, y = lap.lapjv(matrix, extend_cost=True, cost_limit=threshold)  # 使用 LAPJV 匹配
    match_pairs = np.array([[ix, mx] for ix, mx in enumerate(x)])  # 組成匹配對
    un_player_idx = np.where(x < 0)[0]  # 未匹配的玩家索引
    un_detection_idx = np.where(y < 0)[0]  # 未匹配的偵測索引
    return match_pairs, un_player_idx, un_detection_idx

def matching(players:list, detections:list, match_pairs:list):
    for player_index, detection_index in match_pairs:
        if player_index < 0 or detection_index < 0:
            continue  # 跳過無效匹配
        new_xywh = detections[detection_index].xywh
        new_camera_idx = detections[detection_index].camera_idx
        new_label = detections[detection_index].group
        new_pd = detections[detection_index].pd

        players[player_index].to_steady(new_xywh, new_camera_idx, new_label, new_pd)  # 更新玩家資訊
    return players

def tracking(cfg, players, detect_players):
    lost_players, steady_players, new_players, unmatch_detections = [], [], [], []

    if not players:  # 若尚無玩家（初始化）
        for player in detect_players:
            player.to_steady(player.xywh, player.camera_idx, player.group, player.pd)
            new_players.append(player)
        return new_players

    for player in players:
        if player.state == PlayerState.Lost:
            lost_players.append(player)  # 已丟失玩家
        else:
            steady_players.append(player)  # 穩定追蹤玩家

    distance_array = create_distance_array(steady_players, detect_players)  # 產生距離矩陣

    if distance_array.size == 0:
        for detection in detect_players:
            if (len(players + new_players)) < cfg.MaxPlayerCount:
                detection.to_steady(detection.xywh, detection.camera_idx, detection.group, detection.pd)
                new_players.append(detection)
        return lost_players + steady_players + new_players

    match_pairs, unmatch_player_idx, unmatch_detections_idx = get_match(distance_array, threshold=60)  # 主匹配
    steady_players = matching(steady_players, detect_players, match_pairs)
    _steady_players = []

    for sid in range(len(steady_players)):
        if sid in unmatch_player_idx:
            steady_players[sid].to_lost()  # 若無匹配，標記為 Lost
            lost_players.append(steady_players[sid])
        else:
            _steady_players.append(steady_players[sid])

    steady_players = _steady_players
    unmatch_detections = [detect_players[idx] for idx in unmatch_detections_idx]

    # 第二階段匹配：Lost vs 未匹配偵測
    re_distance_array = create_distance_array(lost_players, unmatch_detections)

    if re_distance_array.size == 0:
        for detection in unmatch_detections:
            if (len(players)+len(new_players)) < cfg.MaxPlayerCount:
                detection.to_steady(detection.xywh, detection.camera_idx, detection.group, detection.pd)
                new_players.append(detection)
        return lost_players + steady_players + new_players

    rematch_pairs, unmatch_player_idx, unmatch_detections_idx = get_match(re_distance_array, threshold=200)  # 次匹配
    lost_players = matching(lost_players, unmatch_detections, rematch_pairs)
    _lost_players = []
    for player in lost_players:
        if player.state == PlayerState.Steady:
            steady_players.append(player)
        else:
            player.to_lost()
            _lost_players.append(player)
    lost_players = _lost_players
    unmatch_detections = [unmatch_detections[idx] for idx in unmatch_detections_idx]

    for detection in unmatch_detections:
        if (len(players + new_players)) < cfg.MaxPlayerCount:
            detection.to_steady(detection.xywh, detection.camera_idx, detection.group, detection.pd)
            new_players.append(detection)
    return lost_players + steady_players + new_players
