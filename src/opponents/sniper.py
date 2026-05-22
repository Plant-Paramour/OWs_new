"""Sniper 对手 —— 最近行星狙击手。

来源: getting-started notebook, cell 6。只发射足够数量 + 不关注飞行时间。
"""

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


def _parse_planets(obs):
    if isinstance(obs, dict):
        raw = obs.get("planets", [])
    else:
        raw = obs.planets
    return [Planet(*p) for p in raw]


def _get_player(obs):
    if isinstance(obs, dict):
        return obs.get("player", 0)
    return obs.player


class SniperOpponent:
    """最近行星狙击手对手。"""

    def __call__(self, observation, configuration):
        moves = []
        player = _get_player(observation)
        planets = _parse_planets(observation)

        my_planets = [p for p in planets if p.owner == player]
        targets = [p for p in planets if p.owner != player]

        if not targets:
            return moves

        for mine in my_planets:
            nearest = None
            min_dist = float("inf")
            for t in targets:
                dist = math.sqrt((mine.x - t.x) ** 2 + (mine.y - t.y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = t

            if nearest is None:
                continue

            ships_needed = max(nearest.ships + 1, 20)
            if mine.ships >= ships_needed:
                angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
                moves.append([mine.id, angle, ships_needed])

        return moves
