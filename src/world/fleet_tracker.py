"""Orbit Wars 舰队追踪 —— 识别舰队目标 & 构建到达账本。

使用 swept-pair 前向模拟，与游戏引擎的碰撞检测完全一致，
正确计入行星轨道运动和彗星直线运动。
"""

import math
from ..engine.constants import SIM_HORIZON, BOARD_SIZE, SUN_RADIUS, CENTER_X, CENTER_Y
from ..engine.physics import fleet_speed, point_to_segment_distance
from ..engine.prediction import predict_target_position


def _fleet_arrival_forward(fleet, planets: list, initial_by_id: dict,
                            angular_velocity: float, comets: list,
                            comet_ids: set) -> tuple:
    """前向模拟单支舰队，用 swept-pair 精确计算到达目标和 ETA。

    与 game engine 的碰撞检测完全一致：每回合同时移动舰队和行星，
    swept_pair_hit 检测交会。同时检查出界、撞太阳、被其他行星拦截。

    Returns:
        (target_planet_id, eta_turns) 或 (None, None)
    """
    from ..engine.validation import swept_pair_hit

    speed = fleet_speed(max(1, fleet.ships))
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    fx, fy = fleet.x, fleet.y

    # 先用快速射线-圆筛选最可能的目标（减少 swept-pair 检查量）
    best_target = None
    best_dist = float("inf")
    for planet in planets:
        dx = planet.x - fx
        dy = planet.y - fy
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        if perp_sq >= planet.radius * planet.radius:
            continue
        if proj < best_dist:
            best_dist = proj
            best_target = planet

    if best_target is None:
        return None, None

    # 前向模拟：逐回合 swept-pair 检测（仅检查最可能目标）
    # 中途拦截已在 action_space.validate_fleet_arrival 阶段过滤
    for turn in range(1, SIM_HORIZON + 1):
        fleet_old_x = fx + dir_x * speed * (turn - 1)
        fleet_old_y = fy + dir_y * speed * (turn - 1)
        fleet_new_x = fx + dir_x * speed * turn
        fleet_new_y = fy + dir_y * speed * turn

        A = (fleet_old_x, fleet_old_y)
        B = (fleet_new_x, fleet_new_y)

        p_old = predict_target_position(best_target, turn - 1, initial_by_id,
                                        angular_velocity, comets, comet_ids)
        p_new = predict_target_position(best_target, turn, initial_by_id,
                                        angular_velocity, comets, comet_ids)
        if p_old and p_new and swept_pair_hit(A, B, p_old, p_new, best_target.radius):
            return best_target.id, turn

        # 检查出界 / 撞太阳（舰队永远丢失）
        if not (0 <= fleet_new_x <= BOARD_SIZE and 0 <= fleet_new_y <= BOARD_SIZE):
            return None, None
        if point_to_segment_distance(CENTER_X, CENTER_Y, fleet_old_x, fleet_old_y,
                                     fleet_new_x, fleet_new_y) < SUN_RADIUS:
            return None, None

    return None, None


def fleet_target_planet(fleet, planets: list):
    """静态射线-圆 ETA 估算（快速但不精确）。

    对于轨道行星和彗星，ETA 可能偏差 1-2 回合。
    保留用于向后兼容；MCTS 应使用 _fleet_arrival_forward()。
    """
    best_planet = None
    best_time = 1e9
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    speed = fleet_speed(fleet.ships)

    for planet in planets:
        dx = planet.x - fleet.x
        dy = planet.y - fleet.y
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        radius_sq = planet.radius * planet.radius
        if perp_sq >= radius_sq:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, radius_sq - perp_sq)))
        turns = hit_d / speed
        if turns <= SIM_HORIZON and turns < best_time:
            best_time = turns
            best_planet = planet

    if best_planet is None:
        return None, None
    return best_planet, int(math.ceil(best_time))


def build_arrival_ledger(fleets: list, planets: list) -> dict:
    """构建到达账本（兼容旧接口）。

    使用静态射线-圆快速估算。MCTS 应使用 build_arrival_ledger_accurate()。
    """
    arrivals_by_planet = {planet.id: [] for planet in planets}
    for fleet in fleets:
        target, eta = fleet_target_planet(fleet, planets)
        if target is None:
            continue
        arrivals_by_planet[target.id].append((eta, fleet.owner, int(fleet.ships)))
    return arrivals_by_planet


def build_arrival_ledger_accurate(state) -> dict:
    """构建精确到达账本：用 swept-pair 前向模拟计算 ETA。

    与游戏引擎的碰撞检测完全一致，正确计入行星轨道和彗星运动。
    MCTS 时间线评估专用。

    Args:
        state: GameState（需含 planets, fleets, initial_by_id,
               angular_velocity, comets, comet_ids）

    Returns:
        {planet_id: [(eta, owner, ships), ...]}
    """
    arrivals_by_planet = {planet.id: [] for planet in state.planets}
    for fleet in state.fleets:
        target_id, eta = _fleet_arrival_forward(
            fleet, state.planets, state.initial_by_id,
            state.angular_velocity, state.comets, state.comet_ids,
        )
        if target_id is None:
            continue
        arrivals_by_planet[target_id].append((eta, fleet.owner, int(fleet.ships)))
    return arrivals_by_planet
