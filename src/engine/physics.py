"""Orbit Wars 物理引擎 —— 与游戏引擎完全匹配的核心物理计算。

所有函数均为纯函数：输入确定 → 输出确定，无副作用。
来源：Structured Baseline v11.
"""

import math
from .constants import (
    CENTER_X, CENTER_Y, SUN_RADIUS, SUN_SAFETY,
    MAX_SPEED, LAUNCH_CLEARANCE,
)


def dist(ax: float, ay: float, bx: float, by: float) -> float:
    """两点之间的欧几里得距离。"""
    return math.hypot(ax - bx, ay - by)


def fleet_speed(ships: int) -> float:
    """对数速度曲线 —— 与 orbit_wars 引擎完全一致。

    1 艘舰船 → 1.0 units/turn
    较大舰队更快，趋近 MAX_SPEED。
    """
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


def point_to_segment_distance(px: float, py: float,
                               x1: float, y1: float,
                               x2: float, y2: float) -> float:
    """点 (px,py) 到线段 (x1,y1)→(x2,y2) 的最短距离。"""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return dist(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return dist(px, py, proj_x, proj_y)


def segment_hits_sun(x1: float, y1: float,
                     x2: float, y2: float,
                     safety: float = SUN_SAFETY) -> bool:
    """线段 (x1,y1)→(x2,y2) 是否穿越太阳（含安全余量）。"""
    return point_to_segment_distance(
        CENTER_X, CENTER_Y, x1, y1, x2, y2
    ) < SUN_RADIUS + safety


def launch_point(sx: float, sy: float, sr: float, angle: float):
    """从行星边界发射舰队后的起始坐标。"""
    clearance = sr + LAUNCH_CLEARANCE
    return sx + math.cos(angle) * clearance, sy + math.sin(angle) * clearance


def actual_path_geometry(sx: float, sy: float, sr: float,
                          tx: float, ty: float, tr: float):
    """计算从源行星边界到目标行星边界的实际路径几何。

    Returns:
        (angle, start_x, start_y, end_x, end_y, hit_distance)
    """
    angle = math.atan2(ty - sy, tx - sx)
    start_x, start_y = launch_point(sx, sy, sr, angle)
    hit_distance = max(0.0, dist(sx, sy, tx, ty) - (sr + LAUNCH_CLEARANCE) - tr)
    end_x = start_x + math.cos(angle) * hit_distance
    end_y = start_y + math.sin(angle) * hit_distance
    return angle, start_x, start_y, end_x, end_y, hit_distance


def safe_angle_and_distance(sx: float, sy: float, sr: float,
                             tx: float, ty: float, tr: float):
    """太阳安全的直接路径 → (angle, total_distance)。若被阻挡则返回 None。"""
    angle, start_x, start_y, end_x, end_y, hit_distance = actual_path_geometry(
        sx, sy, sr, tx, ty, tr
    )
    if segment_hits_sun(start_x, start_y, end_x, end_y):
        return None
    return angle, hit_distance


def estimate_arrival(sx: float, sy: float, sr: float,
                      tx: float, ty: float, tr: float,
                      ships: int):
    """估算舰队从源行星到目标行星的 (angle, turns)。

    使用边界感知 ETA 模型，服务于路由、排序、保留和发射决策。
    """
    safe = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    if safe is None:
        return None
    angle, total_d = safe
    turns = max(1, int(math.ceil(total_d / fleet_speed(max(1, ships)))))
    return angle, turns


def estimate_arrival_float(sx: float, sy: float, sr: float,
                            tx: float, ty: float, tr: float,
                            ships: int):
    """同 estimate_arrival，但返回浮点数 ETA（用于精确拦截瞄准）。

    游戏引擎的碰撞检测是连续的（swept_pair_hit），舰队可在小数回合到达。
    整数向上取整会在远距离目标上积累 ~1 单位的瞄准误差。
    """
    safe = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    if safe is None:
        return None
    angle, total_d = safe
    eta = total_d / fleet_speed(max(1, ships))
    return angle, eta


def travel_time(sx: float, sy: float, sr: float,
                 tx: float, ty: float, tr: float,
                 ships: int) -> int:
    """仅返回到达回合数（不安全路径返回大数）。"""
    est = estimate_arrival(sx, sy, sr, tx, ty, tr, ships)
    if est is None:
        return 10 ** 9
    return est[1]
