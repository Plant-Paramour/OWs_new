"""MCTS 时间线投影价值函数。

核心理念：当前局面不是静态快照，而是未来全时间线投影。
逐回合追踪每个行星归属变化，计入所有在途舰队到达和战斗结算。
价值 = 终局舰船 + 各回合产值，直接对应游戏胜负。
"""

from ..world.fleet_tracker import build_arrival_ledger_accurate
from ..world.combat import simulate_planet_timeline
from ..world.types import GameState
from ..engine.prediction import comet_remaining_life


def build_baseline_timelines(state: GameState) -> dict:
    """对所有行星构建"如果没有任何新发射"的基线时间线。

    使用 swept-pair 前向模拟精确计算所有在途舰队的到达 ETA，
    与游戏引擎碰撞检测完全一致。
    彗星行星在寿命结束时正确标记过期（驻军损失）。

    Returns:
        {planet_id: timeline_dict}
        timeline_dict 包含: owner_at, ships_at, keep_needed, fall_turn, first_enemy, holds_full, horizon
    """
    ledger = build_arrival_ledger_accurate(state)
    horizon = min(state.remaining_steps, 110)
    timelines = {}
    for planet in state.planets:
        arrivals = ledger.get(planet.id, [])
        life = comet_remaining_life(planet.id, state.comets) if planet.id in state.comet_ids else None
        timelines[planet.id] = simulate_planet_timeline(planet, arrivals, state.player, horizon, planet_life=life)
    return timelines


def player_value_from_timelines(timelines: dict, planets: list, remaining_steps: int) -> dict:
    """从时间线投影计算逐玩家未来总价值。

    遍历未来每回合：owner 获得该回合的产值。
    终局：owner 获得最终驻军（全额，直接对应胜负）。
    """
    value = {}
    horizon = min(remaining_steps, 110)

    for planet in planets:
        tl = timelines.get(planet.id)
        if tl is None:
            continue

        for turn in range(1, horizon + 1):
            owner = tl["owner_at"].get(turn)
            if owner is not None and owner >= 0:
                value[owner] = value.get(owner, 0.0) + planet.production

        final_owner = tl["owner_at"].get(horizon)
        final_ships = tl["ships_at"].get(horizon, 0)
        if final_owner is not None and final_owner >= 0:
            value[final_owner] = value.get(final_owner, 0.0) + final_ships

    return value
