r"""Orbit Wars 移动目标拦截求解器。

核心算法：迭代收敛到自洽的拦截解。
来源：Structured Baseline v11.
"""

import math
from .constants import CENTER_X, CENTER_Y, SUN_RADIUS, INTERCEPT_TOLERANCE, ROUTE_SEARCH_HORIZON, SIM_HORIZON
from .physics import dist, estimate_arrival, estimate_arrival_float, segment_hits_sun, fleet_speed, LAUNCH_CLEARANCE
from .prediction import predict_target_position, predict_target_position_float, target_can_move, comet_remaining_life, is_static_planet


def search_safe_intercept(src, target, ships: int,
                           initial_by_id: dict, ang_vel: float,
                           comets: list, comet_ids: set):
    """扫描未来时间窗口，找到最早可行的太阳安全拦截。

    当直接路径被太阳阻挡时使用此函数。
    返回 (angle, turns, target_x, target_y) 或 None。
    """
    best = None
    best_score = None
    max_turns = min(SIM_HORIZON, ROUTE_SEARCH_HORIZON)
    if target.id in comet_ids:
        max_turns = min(max_turns, max(0, comet_remaining_life(target.id, comets) - 1))

    for candidate_turns in range(1, max_turns + 1):
        pos = predict_target_position(
            target, candidate_turns, initial_by_id, ang_vel, comets, comet_ids
        )
        if pos is None:
            continue
        est = estimate_arrival(
            src.x, src.y, src.radius,
            pos[0], pos[1], target.radius,
            ships,
        )
        if est is None:
            continue
        _, turns = est
        if abs(turns - candidate_turns) > INTERCEPT_TOLERANCE:
            continue

        actual_turns = max(turns, candidate_turns)
        actual_pos = predict_target_position(
            target, actual_turns, initial_by_id, ang_vel, comets, comet_ids
        )
        if actual_pos is None:
            continue

        confirm = estimate_arrival(
            src.x, src.y, src.radius,
            actual_pos[0], actual_pos[1], target.radius,
            ships,
        )
        if confirm is None:
            continue

        delta = abs(confirm[1] - actual_turns)
        if delta > INTERCEPT_TOLERANCE:
            continue

        score = (delta, confirm[1], candidate_turns)
        if best is None or score < best_score:
            best_score = score
            best = (confirm[0], confirm[1], actual_pos[0], actual_pos[1])

    return best


def aim_at(src, target, ships: int,
           initial_by_id: dict, ang_vel: float,
           comets: list, comet_ids: set):
    """迭代拦截求解器 —— 使用浮点 ETA + 插值位置预测。

    游戏引擎的碰撞检测是连续的（swept_pair_hit），舰队可在小数回合到达。
    旧版使用 int(ceil(距离/速度)) 进行位置预测，取整误差在远距离目标上
    放大为 ~1 单位的位置偏差，导致小型行星（r≈1）被完全打飞。

    改进：
    - 使用浮点 ETA（不取整）
    - 对轨道行星/彗星位置进行线性插值
    - 收敛条件：ETA 变化 < 1e-4 回合（固定点迭代）
    - 最多 50 次迭代
    - 收敛后将浮点 ETA 取整为 int 返回（保持 API 兼容）

    Returns:
        (angle, turns, predicted_target_x, predicted_target_y) 或 None
    """
    est = estimate_arrival_float(
        src.x, src.y, src.radius,
        target.x, target.y, target.radius,
        ships,
    )
    if est is None:
        if not target_can_move(target, initial_by_id, comet_ids):
            return None
        return search_safe_intercept(
            src, target, ships, initial_by_id, ang_vel, comets, comet_ids
        )

    _, eta = est
    for _ in range(50):
        pos = predict_target_position_float(
            target, eta, initial_by_id, ang_vel, comets, comet_ids
        )
        if pos is None:
            return None
        ntx, nty = pos
        next_est = estimate_arrival_float(
            src.x, src.y, src.radius,
            ntx, nty, target.radius,
            ships,
        )
        if next_est is None:
            if not target_can_move(target, initial_by_id, comet_ids):
                return None
            return search_safe_intercept(
                src, target, ships, initial_by_id, ang_vel, comets, comet_ids
            )
        next_angle, next_eta = next_est
        if abs(next_eta - eta) < 1e-4:
            turns = max(1, int(math.ceil(next_eta)))
            return next_angle, turns, ntx, nty
        eta = next_eta

    # 兜底：50 次后仍未收敛（极端罕见），用当前值
    turns = max(1, int(math.ceil(eta)))
    pos = predict_target_position_float(
        target, eta, initial_by_id, ang_vel, comets, comet_ids
    )
    if pos is not None:
        final_est = estimate_arrival_float(
            src.x, src.y, src.radius,
            pos[0], pos[1], target.radius,
            ships,
        )
        if final_est is not None:
            return final_est[0], max(1, int(math.ceil(final_est[1]))), pos[0], pos[1]
    final_est = estimate_arrival(
        src.x, src.y, src.radius,
        target.x, target.y, target.radius,
        ships,
    )
    if final_est is not None:
        return final_est[0], final_est[1], target.x, target.y
    return None


def check_path_blocked(src, target, ships: int, angle: float, eta: float,
                        state) -> tuple:
    """检查舰队从 src 到 target 的直线路径是否被中间行星拦截。

    游戏引擎每回合对全体行星做 swept_pair_hit 碰撞检测，舰队飞经任何
    行星附近都可能被截获。本函数预测此行为，防止智能体发送永远无法
    到达目标的舰队。

    算法：
    1. 射线-圆测试（当前行星位置）快速过滤可能的拦截者
    2. 对疑似拦截者逐回合检查（含轨道行星位置预测）

    Args:
        src: 源行星 (Planet)
        target: 目标行星 (Planet)
        ships: 舰队舰船数
        angle: 舰队发射角度 (rad)
        eta: 到达目标的预估回合数 (float)
        state: 全局 GameState（需含 initial_by_id, angular_velocity, comets, comet_ids）

    Returns:
        (is_blocked: bool, blocking_planet_id: int | None, block_turn: int | None)
    """
    speed = fleet_speed(max(1, ships))
    clearance = src.radius + LAUNCH_CLEARANCE
    fx = src.x + math.cos(angle) * clearance
    fy = src.y + math.sin(angle) * clearance

    dir_x = math.cos(angle)
    dir_y = math.sin(angle)

    # ── 第 1 层：射线-圆快速过滤 ──
    suspects = []
    for planet in state.planets:
        if planet.id == src.id or planet.id == target.id:
            continue
        dx = planet.x - fx
        dy = planet.y - fy
        proj = dx * dir_x + dy * dir_y
        if proj <= 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        r2 = planet.radius * planet.radius
        if perp_sq >= r2:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, r2 - perp_sq)))
        hit_t = hit_d / speed
        if hit_t < eta + 1.0:
            suspects.append((planet, hit_t))

    if not suspects:
        return False, None, None

    # ── 第 2 层：逐回合确认（含轨道行星运动）──
    for planet, approx_hit in suspects:
        static = is_static_planet(planet)
        t0 = max(0, int(approx_hit) - 1)
        t1 = min(int(eta) + 2, int(approx_hit) + 3)

        for turn in range(t0, t1 + 1):
            tf = float(turn)
            pos = predict_target_position_float(
                planet, tf, state.initial_by_id, state.angular_velocity,
                state.comets, state.comet_ids,
            )
            if pos is None:
                continue
            px, py = pos
            ftx = fx + dir_x * speed * tf
            fty = fy + dir_y * speed * tf
            # 保守容差：舰队半速 + 行星半移 ≈ 最多半回合运动
            margin = planet.radius + speed * 0.6
            if math.hypot(ftx - px, fty - py) < margin:
                return True, planet.id, turn

    return False, None, None
