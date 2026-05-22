"""MCTS 逐玩家未来时间线价值函数。

核心理念：当前局面不是静态快照，而是未来全时间线投影。
价值公式套在时间线投影上，零和从底层自然涌现。
"""

from ..world.fleet_tracker import build_arrival_ledger
from ..world.combat import simulate_planet_timeline
from ..world.types import GameState
from ..engine.prediction import comet_remaining_life


def build_baseline_timelines(state: GameState) -> dict:
    """对所有行星构建"如果没有任何新发射"的基线时间线。

    Returns:
        {planet_id: timeline_dict}
        timeline_dict 包含: owner_at, ships_at, keep_needed, fall_turn, first_enemy, holds_full, horizon
    """
    ledger = build_arrival_ledger(state.fleets, state.planets)
    horizon = min(state.remaining_steps, 110)
    timelines = {}
    for planet in state.planets:
        arrivals = ledger.get(planet.id, [])
        timelines[planet.id] = simulate_planet_timeline(planet, arrivals, state.player, horizon)
    return timelines


def build_baseline_ledger(state: GameState) -> dict:
    """构建基线到达账本。"""
    return build_arrival_ledger(state.fleets, state.planets)


def player_value_from_timelines(timelines: dict, planets: list, remaining_steps: int) -> dict:
    """从时间线投影计算逐玩家未来总价值。

    遍历未来每回合：owner 获得该回合的产值。
    终局：owner 获得最终驻军。
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


def player_value_from_state(state: GameState) -> dict:
    """快速静态评估——用于 rollout 截断。

    不跑时间线模拟，直接用当前驻军 + 剩余回合产值近似。
    彗星产值使用实际剩余寿命，非全局 remaining。
    彗星接近过期时驻军价值归零 → 自然产生撤离激励。
    """
    value = {}
    remaining = state.remaining_steps

    for p in state.planets:
        if p.owner >= 0:
            life = _planet_life(p.id, state, remaining)
            ships_value = p.ships
            if p.id in state.comet_ids:
                comet_life = comet_remaining_life(p.id, state.comets)
                if comet_life <= 3:
                    ships_value = 0  # 彗星快过期，驻军即将永久消失
                elif comet_life <= 6:
                    ships_value = int(p.ships * 0.4)  # 渐进贬值
            value[p.owner] = value.get(p.owner, 0.0) + ships_value + p.production * life
            # 彗星跳板加成
            if p.id in state.comet_ids:
                springboard = comet_springboard_value(p, state, p.owner)
                value[p.owner] = value.get(p.owner, 0.0) + springboard

    for f in state.fleets:
        if f.owner >= 0:
            value[f.owner] = value.get(f.owner, 0.0) + f.ships * 0.8

    return value


def _planet_life(planet_id: int, state: GameState, remaining: int) -> int:
    """行星/彗星的实际剩余存活回合数。彗星取 min(remaining, comet_life)。"""
    if planet_id in state.comet_ids:
        life = comet_remaining_life(planet_id, state.comets)
        return max(0, min(remaining, life))
    return remaining


def comet_springboard_value(comet, state: GameState, player: int) -> float:
    """彗星跳板价值：彗星路径上靠近敌方行星的突袭潜力。

    沿彗星未来路径搜索，评估从彗星位置发起攻击的"缩短距离"优势。
    返回值为等效舰船数加成。
    """
    import math
    from ..engine.prediction import predict_comet_position

    if comet.id not in state.comet_ids:
        return 0.0

    life = comet_remaining_life(comet.id, state.comets)
    if life < 6:
        return 0.0

    enemy_planets = state.enemy_planets
    if not enemy_planets:
        return 0.0

    # 从彗星沿途位置找最近的敌方行星距离
    best_advantage = 0.0
    for turn_offset in range(0, min(life, 30), 3):
        pos = predict_comet_position(comet.id, state.comets, turn_offset)
        if pos is None:
            break
        cx, cy = pos
        # 找此位置最近的敌方行星
        min_dist = float("inf")
        for ep in enemy_planets:
            d = math.hypot(cx - ep.x, cy - ep.y)
            if d < min_dist:
                min_dist = d
        # 与我方最近行星到该敌星的距离比较
        my_closest_dist = float("inf")
        for mp in state.my_planets:
            for ep in enemy_planets:
                d = math.hypot(mp.x - ep.x, mp.y - ep.y)
                if d < my_closest_dist:
                    my_closest_dist = d
        if min_dist < my_closest_dist * 0.7 and min_dist < 40:
            advantage = my_closest_dist - min_dist
            best_advantage = max(best_advantage, advantage)

    # 等效舰船加成：距离优势 × 常数因子
    return best_advantage * 2.0


def action_impact(action, state: GameState, baseline_timelines: dict,
                  baseline_ledger: dict) -> dict:
    """增量计算单动作的各玩家价值变化（不跑完整 step_state）。

    仅重新模拟受影响的行星时间线（目标行星 + 源行星），
    其余行星保持基线时间线不变。

    Returns:
        {player_id: delta_value}
    """
    target_id = action.target_id
    source_id = action.source_id
    player = action.player
    ships = action.ships
    eta = action.eta

    # 找到目标行星和源行星
    target = None
    source = None
    for p in state.planets:
        if p.id == target_id:
            target = p
        if p.id == source_id:
            source = p

    if target is None:
        return {}

    horizon = min(state.remaining_steps, 110)

    # 构建旧时间线（只取源和目标两条）
    old_timelines = {
        target_id: baseline_timelines.get(target_id),
        source_id: baseline_timelines.get(source_id),
    }

    # 计算旧价值
    old_value = _planet_pair_value(old_timelines, target, source, horizon)

    # 增量更新目标行星时间线（注入新舰队）
    target_arrivals = list(baseline_ledger.get(target_id, []))
    target_arrivals.append((eta, player, ships))
    new_timeline_target = simulate_planet_timeline(target, target_arrivals, state.player, horizon)

    # 检查源行星安全性
    new_timeline_source = None
    if source is not None and baseline_timelines.get(source_id):
        src_tl = baseline_timelines[source_id]
        new_garrison = source.ships - ships
        if new_garrison < src_tl.get("keep_needed", 0):
            source_arrivals = list(baseline_ledger.get(source_id, []))
            new_timeline_source = _simulate_with_override_garrison(
                source, source_arrivals, state.player, horizon, float(new_garrison)
            )
        else:
            new_timeline_source = src_tl

    new_timelines = {
        target_id: new_timeline_target,
        source_id: new_timeline_source,
    }

    new_value = _planet_pair_value(new_timelines, target, source, horizon)

    # 差值
    delta = {}
    all_players = set(list(old_value.keys()) + list(new_value.keys()))
    for pid in all_players:
        dv = new_value.get(pid, 0.0) - old_value.get(pid, 0.0)
        if abs(dv) > 0.001:
            delta[pid] = dv

    return delta


def _planet_pair_value(timelines: dict, target, source, horizon: int) -> dict:
    """计算一对行星（目标+源）的未来价值贡献。"""
    value = {}
    for pid, tl in timelines.items():
        if tl is None:
            continue
        planet = target if pid == target.id else (source if source and pid == source.id else None)
        if planet is None:
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


def _simulate_with_override_garrison(planet, arrivals: list, player: int,
                                      horizon: int, override_garrison: float) -> dict:
    """以指定初始驻军覆盖，重新模拟行星时间线。"""
    from collections import defaultdict
    import math
    from ..world.combat import resolve_arrival_event, normalize_arrivals

    horizon = max(0, int(math.ceil(horizon)))
    events = normalize_arrivals(arrivals, horizon)
    by_turn = defaultdict(list)
    for item in events:
        by_turn[item[0]].append(item)

    owner = planet.owner
    garrison = override_garrison
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}
    first_enemy = None
    fall_turn = None
    min_owned = garrison if owner == player else 0.0

    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production
        group = by_turn.get(turn, [])
        prev_owner = owner
        if group:
            if prev_owner == player and first_enemy is None:
                if any(item[1] not in (-1, player) for item in group):
                    first_enemy = turn
            owner, garrison = resolve_arrival_event(owner, garrison, group)
            if prev_owner == player and owner != player and fall_turn is None:
                fall_turn = turn
        owner_at[turn] = owner
        ships_at[turn] = max(0.0, garrison)
        if owner == player:
            min_owned = min(min_owned, garrison)

    return {
        "owner_at": owner_at,
        "ships_at": ships_at,
        "keep_needed": 0,
        "min_owned": max(0, int(math.floor(min_owned))) if planet.owner == player else 0,
        "first_enemy": first_enemy,
        "fall_turn": fall_turn,
        "holds_full": True,
        "horizon": horizon,
    }


def truncated_value(state: GameState, truncation_turn: int) -> dict:
    """截断评估：假设之后所有权不变，剩余回合×产值。彗星使用实际寿命。"""
    remaining = state.episode_steps - truncation_turn
    value = {}
    for p in state.planets:
        if p.owner >= 0:
            life = _planet_life(p.id, state, max(0, remaining))
            value[p.owner] = value.get(p.owner, 0.0) + p.ships + p.production * max(0, life)
    for f in state.fleets:
        if f.owner >= 0:
            value[f.owner] = value.get(f.owner, 0.0) + f.ships * 0.8
    return value
