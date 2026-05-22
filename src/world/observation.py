"""Orbit Wars 观测解析 —— 将原始观测转换为标准化的 GameState。

支持 dict 和 attribute 两种观测格式。
来源：Structured Baseline v11.
"""

from collections import defaultdict
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from ..engine.prediction import is_static_planet
from .types import GameState
from .fleet_tracker import build_arrival_ledger
from .combat import simulate_planet_timeline


def obs_get(obs, key: str, default=None):
    """兼容 dict 和 attribute 两种观测访问方式。"""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def parse_observation(obs, player_override: int | None = None,
                     episode_steps: int = 500) -> GameState:
    """将原始观测解析为标准化的 GameState。

    Args:
        obs: 原始观测 (dict 或 attribute-based)
        player_override: 覆盖 player ID（用于对手视角）
        episode_steps: 游戏总回合数（1v1=500, 4p=200），来自 configuration.episodeSteps

    Returns:
        包含所有预计算分类和统计的 GameState
    """
    player = player_override if player_override is not None else obs_get(obs, "player", 0)
    step = obs_get(obs, "step", 0) or 0
    ang_vel = obs_get(obs, "angular_velocity", 0.0) or 0.0

    raw_planets = obs_get(obs, "planets", []) or []
    raw_fleets = obs_get(obs, "fleets", []) or []
    raw_init = obs_get(obs, "initial_planets", []) or []
    comets = obs_get(obs, "comets", []) or []
    comet_ids = set(obs_get(obs, "comet_planet_ids", []) or [])

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    initial_by_id = {Planet(*p).id: Planet(*p) for p in raw_init}

    my_planets = [p for p in planets if p.owner == player]
    enemy_planets = [p for p in planets if p.owner not in (-1, player)]
    neutral_planets = [p for p in planets if p.owner == -1]

    # 舰船 & 产量统计
    my_total_ships = 0
    enemy_total_ships = 0
    my_total_production = 0
    enemy_total_production = 0
    for p in planets:
        if p.owner == player:
            my_total_ships += int(p.ships)
            my_total_production += int(p.production)
        elif p.owner != -1:
            enemy_total_ships += int(p.ships)
            enemy_total_production += int(p.production)
    for f in fleets:
        if f.owner == player:
            my_total_ships += int(f.ships)
        else:
            enemy_total_ships += int(f.ships)

    # 活跃玩家数
    owners = {p.owner for p in planets if p.owner >= 0}
    owners |= {f.owner for f in fleets if f.owner >= 0}
    num_players = max(2, len(owners))

    remaining = max(1, episode_steps - step)

    return GameState(
        step=step,
        player=player,
        planets=planets,
        fleets=fleets,
        angular_velocity=ang_vel,
        initial_by_id=initial_by_id,
        comets=comets,
        comet_ids=comet_ids,
        my_planets=my_planets,
        enemy_planets=enemy_planets,
        neutral_planets=neutral_planets,
        remaining_steps=remaining,
        episode_steps=episode_steps,
        num_players=num_players,
        my_total_ships=my_total_ships,
        enemy_total_ships=enemy_total_ships,
        my_total_production=my_total_production,
        enemy_total_production=enemy_total_production,
    )
