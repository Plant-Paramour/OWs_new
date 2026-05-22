"""Orbit Wars 舰队追踪 —— 识别舰队目标 & 构建到达账本。

来源：Structured Baseline v11.
"""

import math
from ..engine.constants import SIM_HORIZON
from ..engine.physics import fleet_speed


def fleet_target_planet(fleet, planets: list):
    """识别舰队 f 正飞向哪个行星。

    使用射线-圆命中计时：投影到行星的几何关系判定是否命中。
    Returns:
        (target_planet, eta_turns) 或 (None, None)
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
    """构建到达账本：{planet_id: [(eta, owner, ships), ...]}。

    遍历所有舰队，识别每支舰队的目标行星，按目标分组。
    """
    arrivals_by_planet = {planet.id: [] for planet in planets}
    for fleet in fleets:
        target, eta = fleet_target_planet(fleet, planets)
        if target is None:
            continue
        arrivals_by_planet[target.id].append((eta, fleet.owner, int(fleet.ships)))
    return arrivals_by_planet
