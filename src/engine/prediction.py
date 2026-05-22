"""Orbit Wars 位置预测模块 —— 行星 / 彗星未来位置预测。

来源：Structured Baseline v11.
"""

import math
from .constants import CENTER_X, CENTER_Y, ROTATION_LIMIT
from .physics import dist


def orbital_radius(planet) -> float:
    """行星到太阳中心的距离。"""
    return dist(planet.x, planet.y, CENTER_X, CENTER_Y)


def is_static_planet(planet) -> bool:
    """判断行星是否为静态（非轨道）行星。"""
    return orbital_radius(planet) + planet.radius >= ROTATION_LIMIT


def predict_planet_position(planet, initial_by_id: dict,
                             angular_velocity: float, turns: int):
    """预测行星在 turns 回合后的位置。

    对静态行星直接返回当前位置；对轨道行星根据角速度计算旋转后的位置。
    """
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    r = dist(init.x, init.y, CENTER_X, CENTER_Y)
    if r + init.radius >= ROTATION_LIMIT:
        return planet.x, planet.y
    cur_ang = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    new_ang = cur_ang + angular_velocity * turns
    return (
        CENTER_X + r * math.cos(new_ang),
        CENTER_Y + r * math.sin(new_ang),
    )


def predict_comet_position(planet_id: int, comets: list, turns: int):
    """预测彗星行星在 turns 回合后的位置。若彗星已过期返回 None。"""
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        path_index = group.get("path_index", 0)
        if idx >= len(paths):
            return None
        path = paths[idx]
        future_idx = path_index + int(turns)
        if 0 <= future_idx < len(path):
            return path[future_idx][0], path[future_idx][1]
        return None
    return None


def comet_remaining_life(planet_id: int, comets: list) -> int:
    """彗星行星的剩余存活回合数。非彗星返回 0。"""
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        path_index = group.get("path_index", 0)
        if idx < len(paths):
            return max(0, len(paths[idx]) - path_index)
    return 0


def predict_target_position(target, turns: int,
                             initial_by_id: dict, ang_vel: float,
                             comets: list, comet_ids: set):
    """统一位置预测 —— 按行星类型分发到对应的预测函数。"""
    if target.id in comet_ids:
        return predict_comet_position(target.id, comets, turns)
    return predict_planet_position(target, initial_by_id, ang_vel, turns)


def predict_target_position_float(target, turns: float,
                                   initial_by_id: dict, ang_vel: float,
                                   comets: list, comet_ids: set):
    """浮点数回合位置预测 —— 线性插值用于精确拦截瞄准。

    游戏引擎的碰撞检测是连续的（swept_pair_hit），舰队可在小数回合到达。
    直接使用 int(turns) 会在远距离目标上产生 ~1 单位的瞄准偏差。
    """
    lo = int(turns)
    hi = lo + 1
    frac = turns - lo
    pos_lo = predict_target_position(target, lo, initial_by_id, ang_vel, comets, comet_ids)
    if pos_lo is None:
        return None
    if frac < 1e-9:
        return pos_lo
    pos_hi = predict_target_position(target, hi, initial_by_id, ang_vel, comets, comet_ids)
    if pos_hi is None:
        return pos_lo
    return (
        pos_lo[0] + frac * (pos_hi[0] - pos_lo[0]),
        pos_lo[1] + frac * (pos_hi[1] - pos_lo[1]),
    )


def target_can_move(target, initial_by_id: dict, comet_ids: set) -> bool:
    """判断目标行星是否可移动（轨道行星或彗星）。"""
    if target.id in comet_ids:
        return True
    init = initial_by_id.get(target.id)
    if init is None:
        return False
    r = dist(init.x, init.y, CENTER_X, CENTER_Y)
    return r + init.radius < ROTATION_LIMIT
