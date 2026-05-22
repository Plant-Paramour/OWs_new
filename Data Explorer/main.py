"""
Orbit Wars — 最近行星狙击手智能体

一个简单的智能体，当拥有足够舰船能保证占领时，会夺取最近的非我方行星。

策略：
  对于每颗我方行星，找到最近的非我方行星。
  若我方舰船数超过目标驻军数，派遣刚好足够的数量去占领（驻军 + 1）。
  否则，等待并积累舰船。

演示的核心概念：
  - 解析观测数据（行星、玩家 ID）
  - 使用 atan2 计算舰队方向角度
  - 以 [from_planet_id, angle, num_ships] 格式发送行动
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


def agent(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets

    # 解析为命名元组以便字段访问：
    #   Planet(id, owner, x, y, radius, ships, production)
    #   owner == -1 表示中立，0-3 为玩家 ID
    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    for mine in my_planets:
        # 找到最近的非我方行星
        nearest = None
        min_dist = float("inf")
        for t in targets:
            dist = math.sqrt((mine.x - t.x) ** 2 + (mine.y - t.y) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest = t

        if nearest is None:
            continue

        # 需要派遣超过目标驻军数量的舰船才能占领。
        # 精确派遣 目标驻军 + 1 可保证占领成功。
        ships_needed = nearest.ships + 1

        # 只有在负担得起时才发射——否则继续积累
        if mine.ships >= ships_needed:
            # atan2(dy, dx) 计算从我方行星到目标的角度
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])

    return moves
