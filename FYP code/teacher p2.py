# track.py（已註解）
...


# track_v2.py
import math  # 用於距離計算
import numpy as np  # 用於數值運算
import lap  # LAPJV 匹配算法
from scipy.spatial.distance import cdist  # 選用的距離工具（未使用）

from player_v2 import PlayerState, Tracker  # 玩家狀態與追蹤器類別

def first_frame_people(cfg, detect_players_set):
    new_player_set = []  # 初始化玩家群組集合

    for players in detect_players_set:
        for player in players:
            SamePersonFlag = False
            for idx, compare_players in enumerate(new_player_set):
                if any(math.sqrt((player.mapxy[0] - cmp_player.mapxy[0])**2 + 
                                (player.mapxy[1] - cmp_player.mapxy[1])**2) < cfg.THRESHOLD.Same_Person_distance 
                       for cmp_player in compare_players):
                    SamePersonFlag = True
                    new_player_set[idx].append(player)
                    break
            if not SamePersonFlag:
                new_player_set.append([player])

    return new_player_set  # 回傳初始分群後的玩家列表

def create_distance_array(players:list, detections:list):
    distance_array = np.zeros((len(players), len(detections)))  # 距離矩陣初始化
    for player_count, player in enumerate(players):
        player_mapx, player_mapy = player.mapxy[0], player.mapxy[1]
        for detection_count, detection in enumerate(detections):
            detection_mapx, detection_mapy = detection.mapxy[0], detection.mapxy[1]
            dx, dy = player_mapx - detection_mapx, player_mapy - detection_mapy
            distance = math.hypot(dx, dy)  # 歐式距離
            distance_array[player_count, detection_count] = distance
    return distance_array

def create_team_cost_array(players:list, detections:list):
    team_cost_array = np.zeros((len(players), len(detections)))
    for player_count, player in enumerate(players):
        for detection_count, detection in enumerate(detections):
            if player.group != detection.group:
                distance = float('inf')  # 不同群組代價為無限
            else:
                distance = 0  # 相同群組無額外代價
            team_cost_array[player_count, detection_count] = distance
    return team_cost_array

def get_match(matrix, threshold):
    _, x, y = lap.lapjv(matrix, extend_cost=True, cost_limit=threshold)
    match_pairs = np.array([[ix, mx] for ix, mx in enumerate(x)])
    un_player_idx = np.where(x < 0)[0]
    un_detection_idx = np.where(y < 0)[0]
    return match_pairs, un_player_idx, un_detection_idx

def multiple_matching(players:list, multiple_match:list):
    for player_idx, dtc_players in multiple_match.items():
        mapxys = np.array([player.mapxy for player in dtc_players])
        avg_mapxy = np.mean(mapxys, axis=0)  # 計算平均位置
        players[player_idx].to_steady(avg_mapxy, dtc_players)  # 將多個偵測整合更新玩家

    return players

def multiple_pair(multiple_match:list, detections:list, match_pairs:list):
    for player_index, detection_index in match_pairs:
        if player_index < 0 or detection_index < 0:
            continue
        if player_index not in multiple_match:
            multiple_match[player_index] = []
        multiple_match[player_index].append(detections[detection_index])

    return multiple_match

def detect_players_merge(detect_players):
    mapxys = np.array([player.mapxy for player in detect_players])
    avg_mapxy = np.mean(mapxys, axis=0)  # 合併玩家位置為平均值

    return avg_mapxy

def tracking(cfg, players, detect_players_set):
    lost_players, steady_players, new_players, unmatch_detections = [], [], [], []

    if not players:  # 初始情況
        number = 1
        detect_players_set = first_frame_people(cfg, detect_players_set)
        for detect_players in detect_players_set:
            group = detect_players[0].group
            mapxys = np.array([player.mapxy for player in detect_players])
            avg_mapxy = np.mean(mapxys, axis=0)
            new_players.append(Tracker(cfg, avg_mapxy, group, number, detect_players))  # 建立 Tracker
            number += 1
        return new_players

    for player in players:
        if player.state == PlayerState.Lost:
            lost_players.append(player)
        else:
            steady_players.append(player)

    multiple_match = {}
    multiple_unmatch_player_idx = []
    multiple_unmatch_detections_idx = []

    for set_idx, detect_players in enumerate(detect_players_set):
        unmatch_detections_idx = []
        distance_array = create_distance_array(steady_players, detect_players)
        team_cost_array = create_team_cost_array(steady_players, detect_players)
        match_array = distance_array + team_cost_array  # 加總距離與群組成本

        if distance_array.size != 0:
            match_pairs, unmatch_player_idx, unmatch_detections_idx = get_match(match_array, threshold=60)
            multiple_match = multiple_pair(multiple_match, detect_players, match_pairs)
            multiple_unmatch_player_idx.append(unmatch_player_idx)

        multiple_unmatch_detections_idx.append(unmatch_detections_idx)

    steady_players = multiple_matching(steady_players, multiple_match)

    if len(multiple_unmatch_player_idx) > 1:
        unmatch_player_idx = np.array(multiple_unmatch_player_idx[0])
        for unmatch in multiple_unmatch_player_idx[1:]:
            unmatch_player_idx = np.intersect1d(unmatch_player_idx, unmatch)  # 交集保留穩定未配對
    elif len(multiple_unmatch_player_idx) == 1:
        unmatch_player_idx = multiple_unmatch_player_idx[0]
    else:
        unmatch_player_idx = []

    _steady_players = []
    for sid in range(len(steady_players)):
        if sid in unmatch_player_idx:
            steady_players[sid].to_lost()
            lost_players.append(steady_players[sid])
        else:
            _steady_players.append(steady_players[sid])

    steady_players = _steady_players
    unmatch_detections_set = []

    for set_idx, player_idx in enumerate(multiple_unmatch_detections_idx):
        unmatch_detections_set.append([detect_players_set[set_idx][idx] for idx in player_idx])

    print(f'1st: {[player.number for player in lost_players]}')

    # 第二輪匹配 Lost vs 偵測
    re_multiple_match = {}
    for unmatch_detections in unmatch_detections_set:
        re_distance_array = create_distance_array(lost_players, unmatch_detections)
        re_team_cost_array = create_team_cost_array(lost_players, unmatch_detections)
        re_match_array = re_distance_array + re_team_cost_array
        if re_distance_array.size != 0:
            rematch_pairs, unmatch_player_idx, unmatch_detections_idx = get_match(re_match_array, threshold=200)
            re_multiple_match = multiple_pair(re_multiple_match, unmatch_detections, rematch_pairs)

    lost_players = multiple_matching(lost_players, re_multiple_match)
    _lost_players = []
    for player in lost_players:
        if player.state == PlayerState.Steady:
            steady_players.append(player)
        else:
            player.to_lost()
            _lost_players.append(player)
    lost_players = _lost_players

    print(f'2nd: {[player.number for player in lost_players]}')
    return lost_players + steady_players + new_players
