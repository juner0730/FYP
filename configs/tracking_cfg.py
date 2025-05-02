class TrackingConfig:
    # 跟踪参数
    TRACK_THRESH = 0.5      # 检测阈值
    TRACK_BUFFER = 30       # 跟踪缓冲帧数
    MIN_BOX_AREA = 10       # 最小检测框面积
    
    # 可视化参数
    DRAW_TRACKS = True      # 是否绘制跟踪轨迹
    SHOW_CONF = True        # 是否显示置信度
    
    # 类别映射
    CLASS_NAMES = {
        0: 'player',
        1: 'ball'
    }