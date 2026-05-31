"""MCTS 动作空间与枚举。

支持渐进式候选池展开和剪枝规则。
同一枚举函数对任意玩家通用——我方和敌方动作空间共用。
"""

import math
from dataclasses import dataclass, field

from ..engine.interception import aim_at
from ..engine.physics import dist
from ..engine.validation import validate_fleet_arrival
from ..world.types import GameState
from ..world.fleet_tracker import build_arrival_ledger_accurate
from ..world.combat import simulate_planet_timeline, state_at_timeline
from ..engine.prediction import comet_remaining_life

# 舰队验证：validate_fleet_arrival() 使用与游戏引擎相同的 swept-pair 数学，
# 统一检测路径阻挡、飞出边界、撞太阳、是否能到达目标。
COMET_MIN_LIFE_FOR_CAPTURE = 4    # 彗星至少剩余回合数才考虑攻占


@dataclass
class Action:
    """单次舰队发射动作。"""
    source_id: int
    target_id: int
    ships: int
    angle: float
    eta: float
    player: int
    distance: float
    target_ships: int = 0
    needed: int = 0
    target_production: int = 0

    def is_pass(self) -> bool:
        return self.target_id == -1

    @property
    def arrival_turn(self) -> int:
        """舰队到达目标行星的回合数（ceil(eta)）。

        这是"卡牌"视角的核心标识：本回合出牌 → 第 N 回合到达。
        """
        return max(1, int(math.ceil(self.eta))) if self.eta > 0 else 0

    @property
    def sufficient(self) -> bool:
        """舰船数是否足以攻占目标。"""
        return self.ships >= self.needed if self.needed > 0 else True

    def key(self) -> tuple:
        return (self.source_id, self.target_id, self.ships)


def enumerate_actions(state: GameState, player: int,
                      max_candidates: int = 30,
                      max_distance: float | None = None) -> list:
    """枚举指定玩家的所有可行舰队发射动作。

    策略：
    1. 对玩家的每颗行星，计算可用舰船（ships - keep_needed）
    2. 对每对 (source=己方, target≠己方)，调用 aim_at() 获取角度和 ETA
    3. check_path_blocked() 过滤被阻挡的路径
    4. 尝试多种舰船规模（基于攻占所需兵力）
    5. 按距离升序排列，取 top max_candidates
    6. 始终包含"不行动"（空列表）

    Args:
        state: 当前游戏状态
        player: 执行动作的玩家 ID
        max_candidates: 返回的最大候选动作数
        max_distance: 限制搜索半径（渐进式展开用），None=无限制

    Returns:
        Action 列表（含 pass 动作），按距离排序
    """
    actions = []
    ledger = build_arrival_ledger_accurate(state)
    horizon = min(state.remaining_steps, 110)

    # 为所有行星构建基线时间线投影 —— 用于判断目标在舰队到达时是否已被友军占领
    target_timelines = {}
    for p in state.planets:
        arrivals = ledger.get(p.id, [])
        life = comet_remaining_life(p.id, state.comets) if p.id in state.comet_ids else None
        target_timelines[p.id] = simulate_planet_timeline(p, arrivals, player, horizon, planet_life=life)

    owned = state.my_planets if player == state.player else state.enemy_planets

    for src in owned:
        timeline = target_timelines.get(src.id, {})
        keep_needed = timeline.get("keep_needed", 0)

        # 彗星源特殊处理：过期后无防守需求，接近过期时放开所有舰船用于撤离
        if src.id in state.comet_ids:
            comet_life_src = comet_remaining_life(src.id, state.comets)
            if comet_life_src <= 3:
                keep_needed = 0  # 即将消失，尽快撤离全部驻军
            elif comet_life_src <= 6:
                keep_needed = min(keep_needed, max(0, src.ships // 4))

        available = max(0, src.ships - keep_needed)

        if available < 1:
            continue

        for tgt in state.planets:
            if tgt.id == src.id:
                continue
            if tgt.owner == player:
                continue

            d = dist(src.x, src.y, tgt.x, tgt.y)
            if max_distance is not None and d > max_distance:
                continue

            # aim_at 获取角度和 ETA（粗略，用最大可用舰船）
            result = aim_at(
                src, tgt, max(1, available),
                state.initial_by_id, state.angular_velocity,
                state.comets, state.comet_ids,
            )
            if result is None:
                continue
            angle, eta, _, _ = result
            eta = float(eta)

            if eta > state.remaining_steps:
                continue

            # 时间线投影：舰队到达时目标是否已被友军在途舰队占领？
            tl = target_timelines.get(tgt.id)
            if tl:
                proj_owner, proj_ships = state_at_timeline(tl, eta)
                if proj_owner == player:
                    continue  # 友军舰队会在我们到达前占领此目标
                target_ships = int(proj_ships)
                # 用投影 garrison 计算 needed（已计入生产增长和其他舰队影响）
                if proj_owner == -1:
                    needed = max(1, target_ships + 1)
                else:
                    needed = max(1, int(target_ships * 1.1) + 1)
            else:
                target_ships = tgt.ships
                needed = _compute_needed(tgt, eta, player, target_ships, state.comets, state.comet_ids)

            # 剪枝：available 甚至不到 needed 的一半 → 没戏
            if available < needed * 0.5:
                continue

            # 尝试多种舰船规模
            ship_scales = _compute_ship_scales(available, tgt, eta, player, state.comets, state.comet_ids, needed=needed)

            for s in ship_scales:
                if s < 1 or s > available:
                    continue
                r = aim_at(
                    src, tgt, max(1, s),
                    state.initial_by_id, state.angular_velocity,
                    state.comets, state.comet_ids,
                )
                if r is None:
                    continue
                act_angle, act_eta, _, _ = r

                if act_eta > state.remaining_steps:
                    continue

                # 前向模拟验证（替代 check_path_blocked，使用与游戏引擎相同的 swept-pair 数学）
                # 同时检测：路径阻挡、飞出边界、撞太阳、是否能到达目标
                valid, hit_turn, reason = validate_fleet_arrival(src, tgt, max(1, s), act_angle, state)
                if not valid:
                    continue

                # 彗星寿命过滤：只攻占寿命足够的彗星（跳过寿命太短的中性彗星）
                is_comet_target = tgt.id in state.comet_ids
                if is_comet_target and tgt.owner != player:
                    comet_life = comet_remaining_life(tgt.id, state.comets)
                    if comet_life < COMET_MIN_LIFE_FOR_CAPTURE and tgt.owner == -1:
                        continue  # 中性彗星寿命太短，攻占不划算

                actions.append(Action(
                    source_id=src.id,
                    target_id=tgt.id,
                    ships=s,
                    angle=act_angle,
                    eta=float(act_eta),
                    player=player,
                    distance=d,
                    target_ships=target_ships,
                    needed=needed,
                    target_production=tgt.production,
                ))

    # ── 彗星撤离: 彗星 life == 1 时显式生成撤离动作 ──
    # 回合顺序: 彗星消失(Step 1) → 舰队发射(Step 3)，life==1 是最后发射窗口
    for src in owned:
        if src.id not in state.comet_ids:
            continue
        if src.ships < 1:
            continue
        comet_life = comet_remaining_life(src.id, state.comets)
        if comet_life > 1:
            continue

        # 找最近的安全友方行星
        best_target = None
        best_dist = float("inf")
        for mp in state.planets:
            if mp.id == src.id:
                continue
            if mp.owner != player:
                continue
            if mp.id in state.comet_ids:
                tgt_life = comet_remaining_life(mp.id, state.comets)
                if tgt_life <= 3:
                    continue
            d = dist(src.x, src.y, mp.x, mp.y)
            if d < best_dist:
                best_dist = d
                best_target = mp

        if best_target is None:
            continue

        # 瞄准
        r = aim_at(
            src, best_target, max(1, src.ships),
            state.initial_by_id, state.angular_velocity,
            state.comets, state.comet_ids,
        )
        if r is None:
            continue
        evac_angle, evac_eta, _, _ = r

        if evac_eta > state.remaining_steps:
            continue

        valid, _, _ = validate_fleet_arrival(src, best_target, src.ships, evac_angle, state)
        if not valid:
            continue

        actions.append(Action(
            source_id=src.id,
            target_id=best_target.id,
            ships=src.ships,
            angle=evac_angle,
            eta=float(evac_eta),
            player=player,
            distance=best_dist,
            target_ships=best_target.ships,
            needed=0,
            target_production=best_target.production,
        ))

    # 按距离排序
    actions.sort(key=lambda a: a.distance)

    # 去重
    unique_actions = _deduplicate_actions(actions)

    # 截断
    result = unique_actions[:max_candidates]

    # 始终包含 pass 动作
    result.append(Action(source_id=-1, target_id=-1, ships=0, angle=0.0, eta=0.0, player=player, distance=0.0))

    return result


def _compute_needed(target, eta: float, player: int, raw_garrison: int, comets=None, comet_ids=None) -> int:
    """计算攻占该目标至少需要的舰船数。

    战斗结算要求 attacker > garrison 才能占领（garrison < 0 触发易主）。
    中立星无生产，只需 garrison + 1；敌方星计入途中生产 +10% ETA 余量。
    彗星 garrison 增长上限为 min(eta, comet_remaining_life)。
    """
    garrison = raw_garrison
    if target.owner != -1 and target.owner != player:
        growth_cap = min(eta, 50)
        if comet_ids and target.id in comet_ids and comets:
            comet_life = comet_remaining_life(target.id, comets)
            growth_cap = min(eta, comet_life)
        garrison += target.production * growth_cap
        return max(1, int(garrison * 1.1) + 1)
    # 中立星：无生产，严格大于驻军即可
    return max(1, garrison + 1)


def _compute_ship_scales(available: int, target, eta: float, player: int, comets=None, comet_ids=None, needed: int = None) -> list:
    """舰船规模选项——聚焦关键数量，过滤无意义小规模。

    舰队越大 → 速度越快 → ETA 越短。选项聚焦：
      needed       — 刚好够单独攻占（最经济）
      needed×1.2   — 小安全边际
      needed×1.5   — 舒适边际
      needed×2     — 压倒性兵力
      available    — 全部可用（最快到达）
      available//3, available//2 — 合击贡献（当单源不够时）
    """
    if needed is None:
        needed = _compute_needed(target, eta, player, target.ships, comets, comet_ids)
    scales = []

    if available >= needed:
        scales.append(needed)
        scales.append(min(available, int(needed * 1.2)))
        scales.append(min(available, int(needed * 1.5)))
        if needed * 2 <= available:
            scales.append(needed * 2)
    else:
        scales.append(max(1, available // 3))
        scales.append(max(1, available // 2))

    scales.append(available)

    return sorted(set(s for s in scales if 1 <= s <= available))


def _deduplicate_actions(actions: list) -> list:
    """去重：相同 (source_id, target_id, ships) 精确去重。"""
    if not actions:
        return []
    seen = set()
    result = []
    for a in actions:
        key = a.key()
        if key in seen:
            continue
        seen.add(key)
        result.append(a)
    return result


def progressive_expand_actions(state: GameState, player: int,
                                iteration_count: int,
                                max_candidates: int = 30) -> list:
    """根据迭代次数渐进式展开动作池。

    阶段 0 (iter 0-100): 近距离目标 (distance ≤ 30)
    阶段 1 (iter 100-300): 中等距离 (distance ≤ 60)
    阶段 2 (iter 300+): 全图
    """
    if iteration_count < 100:
        max_dist = 30.0
    elif iteration_count < 300:
        max_dist = 60.0
    else:
        max_dist = None

    return enumerate_actions(state, player, max_candidates=max_candidates, max_distance=max_dist)
