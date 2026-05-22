"""舰队路径前向模拟验证器。

用与游戏引擎完全相同的 swept-pair 碰撞检测，逐回合模拟舰队路径，
验证舰队确实会命中目标行星而非飞出边界或撞太阳。

游戏引擎参考: kaggle_environments/envs/orbit_wars/orbit_wars.py
"""

import math
from .constants import CENTER_X, CENTER_Y, SUN_RADIUS, BOARD_SIZE, SIM_HORIZON, LAUNCH_CLEARANCE
from .physics import fleet_speed, point_to_segment_distance
from .prediction import predict_target_position, predict_target_position_float


def swept_pair_hit(A, B, P0, P1, r):
    """True iff 舰队移动 A→B 与行星移动 P0→P1 在 t∈[0,1] 内距离 ≤ r。

    与游戏引擎 orbit_wars.py:swept_pair_hit 完全一致。
    """
    d0x, d0y = A[0] - P0[0], A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


def validate_fleet_arrival(src, target, ships: int, angle: float, state) -> tuple:
    """前向模拟舰队路径，验证舰队确实会命中目标。

    逐回合模拟：舰队沿 angle 方向以 fleet_speed 速度移动，
    同时目标行星按轨道/彗星路径移动。每回合用 swept_pair_hit
    检测舰队段与目标行星段是否相交。

    Args:
        src: 源行星 (Planet)
        target: 目标行星 (Planet)
        ships: 舰队舰船数
        angle: 发射角度 (rad)，来自 aim_at()
        state: GameState（需含 initial_by_id, angular_velocity, comets, comet_ids, remaining_steps）

    Returns:
        (is_valid, hit_turn, failure_reason)
        - is_valid: 舰队是否会命中目标
        - hit_turn: 命中回合（int），无效时为 None
        - failure_reason: 失败原因字符串
    """
    speed = fleet_speed(max(1, ships))
    clearance = src.radius + LAUNCH_CLEARANCE
    fx = src.x + math.cos(angle) * clearance
    fy = src.y + math.sin(angle) * clearance

    dir_x = math.cos(angle)
    dir_y = math.sin(angle)

    max_turns = min(state.remaining_steps, SIM_HORIZON)

    # 获取目标行星每回合的 old_pos → new_pos 用于 swept-pair
    comet_ids = state.comet_ids
    initial_by_id = state.initial_by_id
    ang_vel = state.angular_velocity

    for turn in range(1, max_turns + 1):
        # 舰队本回合起点和终点
        fleet_old_x = fx + dir_x * speed * (turn - 1)
        fleet_old_y = fy + dir_y * speed * (turn - 1)
        fleet_new_x = fx + dir_x * speed * turn
        fleet_new_y = fy + dir_y * speed * turn

        A = (fleet_old_x, fleet_old_y)
        B = (fleet_new_x, fleet_new_y)

        # ── 先检查目标行星是否在本回合被命中 ──
        tgt_old = predict_target_position(target, turn - 1, initial_by_id, ang_vel, state.comets, comet_ids)
        tgt_new = predict_target_position(target, turn, initial_by_id, ang_vel, state.comets, comet_ids)

        if tgt_old is None or tgt_new is None:
            return False, None, f"target position prediction failed at turn {turn}"

        P0 = tgt_old
        P1 = tgt_new

        if swept_pair_hit(A, B, P0, P1, target.radius):
            return True, turn, None

        # ── 检查出界 ──
        if not (0 <= fleet_new_x <= BOARD_SIZE and 0 <= fleet_new_y <= BOARD_SIZE):
            return False, None, f"fleet out of bounds at turn {turn}"

        # ── 检查撞太阳 ──
        if point_to_segment_distance(CENTER_X, CENTER_Y, fleet_old_x, fleet_old_y, fleet_new_x, fleet_new_y) < SUN_RADIUS:
            return False, None, f"fleet hits sun at turn {turn}"

        # ── 检查被其他行星拦截 ──
        for planet in state.planets:
            if planet.id == src.id or planet.id == target.id:
                continue
            p_old = predict_target_position(planet, turn - 1, initial_by_id, ang_vel, state.comets, comet_ids)
            p_new = predict_target_position(planet, turn, initial_by_id, ang_vel, state.comets, comet_ids)
            if p_old is None or p_new is None:
                continue
            if swept_pair_hit(A, B, p_old, p_new, planet.radius):
                return False, planet.id, f"fleet intercepted by planet {planet.id} at turn {turn}"

    return False, None, "fleet never reached target within SIM_HORIZON"


def hit_confidence(src, target, ships: int, state) -> float:
    """估算舰队命中目标的置信度 (0.0 ~ 1.0)。

    基于距离/半径比和目标是静态/轨道/彗星给出启发式置信度。
    此函数用于在无法运行完整前向模拟时做快速预判。
    """
    d = math.hypot(src.x - target.x, src.y - target.y)
    if d < 1e-6:
        return 1.0

    # 角度容差 ≈ target.radius / d
    angular_tolerance = target.radius / d

    # 静态行星：最高置信度
    is_comet = target.id in state.comet_ids
    is_static = not _target_can_move(target, state)

    if is_static:
        # 静态行星只需考虑初始瞄准精度
        if angular_tolerance > 0.05:
            return 1.0
        elif angular_tolerance > 0.02:
            return 0.95
        elif angular_tolerance > 0.01:
            return 0.85
        else:
            return 0.6
    elif is_comet:
        # 彗星：不确定性最高
        if angular_tolerance > 0.1:
            return 0.9
        elif angular_tolerance > 0.05:
            return 0.7
        elif angular_tolerance > 0.02:
            return 0.5
        else:
            return 0.3
    else:
        # 轨道行星：中等
        if angular_tolerance > 0.05:
            return 0.98
        elif angular_tolerance > 0.02:
            return 0.9
        elif angular_tolerance > 0.01:
            return 0.75
        else:
            return 0.5


def _target_can_move(target, state) -> bool:
    """判断目标行星是否可移动（轨道行星或彗星）。"""
    if target.id in state.comet_ids:
        return True
    init = state.initial_by_id.get(target.id)
    if init is None:
        return False
    from .physics import dist as _dist
    from .constants import ROTATION_LIMIT
    r = _dist(init.x, init.y, CENTER_X, CENTER_Y)
    return r + init.radius < ROTATION_LIMIT
