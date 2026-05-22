"""动作搜索算法 —— 枚举并评估所有 (source, target, ships) 组合，最大化占领价值。

流程:
1. 对每颗己方行星 (source)，枚举候选目标
2. 对每个 (source, target) 对，二分搜索最小攻占舰船数
3. 尝试多种舰船规模，运行 what-if 模拟
4. 计算每种组合的净价值，排序返回

支持单步搜索 (search_best_actions) 和浅层束搜索 (beam_search)。
"""

from dataclasses import dataclass
import math

from ..engine.physics import travel_time
from ..engine.interception import aim_at, check_path_blocked
from .simulator import simulate_fleet_launch, find_min_ships_to_capture
from .simulator import simulate_multi_fleet_launch
from .valuation import compute_action_value, value_of_capture, lookahead_adjustment
from .valuation import value_of_reinforcement, value_of_evacuation

SEARCH_MIN_SHIPS = 5


@dataclass
class ScoredAction:
    """评分后的候选动作。"""
    source_id: int
    target_id: int
    ships: int
    value: float
    capture_turn: int | None
    hold_until: int | None
    eta: int
    source_available: int
    source_at_risk: bool
    is_enemy: bool


def _try_ship_amounts(src, tgt, ship_amounts, state, ledger, timelines,
                      delays=None):
    """对给定的舰船数量列表逐一模拟，返回最佳结果。

    Args:
        ship_amounts: 要尝试的舰船数列表 (去重排序)
        delays: {ships: wait_turns} 映射，表示该舰船数需要攒几回合
        state, ledger, timelines: 世界模型状态

    Returns:
        (best_value, best_ships, best_outcome) 或 (None, None, None)
    """
    best_value = float("-inf")
    best_ships = None
    best_outcome = None

    for ships in ship_amounts:
        if ships < SEARCH_MIN_SHIPS:
            continue
        outcome = simulate_fleet_launch(src, tgt, ships, state, ledger, timelines)
        if outcome.blocked:
            continue
        if outcome.aim is None:
            continue

        delay = (delays or {}).get(ships, 0)

        if delay > 0 and outcome.capture_turn is not None:
            adj_capture = outcome.capture_turn + delay
            adj_hold = outcome.hold_until + delay
            is_enemy = tgt.owner not in (-1, state.player)
            value = value_of_capture(
                tgt, adj_capture, adj_hold,
                state.remaining_steps, ships, is_enemy,
                source_at_risk=outcome.source_at_risk,
            )
        else:
            value = compute_action_value(
                outcome, tgt, state.player, state.remaining_steps,
                ships, state.comet_ids, timelines=timelines,
            )

        if value > best_value:
            best_value = value
            best_ships = ships
            best_outcome = outcome

    return best_value, best_ships, best_outcome


def _evaluate_candidate(src, tgt, available, state, ledger, timelines):
    """评估单个 (source, target) 候选对的最佳舰船数。

    策略:
    1. 二分搜索找到最小攻占舰船数 min_ships
    2. 尝试 [min_ships, min_ships×1.3, min(min_ships×2, available), available]
    3. 返回最佳结果

    Args:
        src: 源行星
        tgt: 目标行星
        available: src 可用的舰船数上限
        state, ledger, timelines: 世界模型状态

    Returns:
        ScoredAction 或 None
    """
    if available < SEARCH_MIN_SHIPS:
        return None

    cap_available = available

    # Step 1: 找到最小攻占舰船数
    min_ships, _min_outcome = find_min_ships_to_capture(
        src, tgt, cap_available, state, ledger, timelines,
    )

    # Step 2: 构建尝试列表（仅评估当前能派出的舰船数）
    if min_ships is not None:
        candidates = [
            min_ships,
            min(int(min_ships * 1.3), cap_available),
            min(min_ships * 2, cap_available),
            cap_available,
        ]
    else:
        candidates = [cap_available]

    candidates = sorted(set(c for c in candidates if c >= SEARCH_MIN_SHIPS))

    if not candidates:
        return None

    # Step 3: 逐一模拟
    best_value, best_ships, best_outcome = _try_ship_amounts(
        src, tgt, candidates, state, ledger, timelines, delays=None,
    )

    if best_ships is None:
        return None

    aim = best_outcome.aim
    eta = int(aim[1]) if aim else 999

    is_enemy = tgt.owner not in (-1, state.player)

    return ScoredAction(
        source_id=src.id,
        target_id=tgt.id,
        ships=best_ships,
        value=best_value,
        capture_turn=best_outcome.capture_turn,
        hold_until=best_outcome.hold_until,
        eta=eta,
        source_available=available,
        source_at_risk=best_outcome.source_at_risk,
        is_enemy=is_enemy,
    )


def search_best_actions(state, ledger, timelines, top_k=20,
                        include_comets=False, eta_progressive=True):
    """搜索最佳单步动作 —— 枚举所有候选，返回按价值排序的结果。

    Args:
        state: GameState
        ledger: 到达账本
        timelines: 行星时间线
        top_k: 返回前 K 个最佳动作
        include_comets: 是否包含彗星目标（默认跳过，由 warmup 处理）
        eta_progressive: 是否使用渐进式 ETA 过滤（前期限制远距离目标）

    Returns:
        list[ScoredAction]: 按 value 降序排列
    """
    results = []
    game_progress = state.step / max(1, state.episode_steps)

    for src in state.my_planets:
        src_timeline = timelines.get(src.id, {})
        keep_needed = src_timeline.get("keep_needed", 0)
        reserve = min(int(src.ships), int(keep_needed))
        available = max(0, int(src.ships) - reserve)

        if available < SEARCH_MIN_SHIPS:
            continue

        all_targets = state.enemy_planets + state.neutral_planets

        # 渐进式 ETA 过滤
        max_eta = 40.0 + game_progress * 85.0 if eta_progressive else 999.0

        for tgt in all_targets:
            if tgt.id == src.id:
                continue
            if not include_comets and tgt.id in state.comet_ids:
                continue

            # 快速 ETA 预筛选
            eta_quick = travel_time(
                src.x, src.y, src.radius,
                tgt.x, tgt.y, tgt.radius,
                max(SEARCH_MIN_SHIPS, available),
            )
            if eta_quick > max_eta:
                continue

            scored = _evaluate_candidate(
                src, tgt, available, state, ledger, timelines,
            )
            if scored is not None:
                results.append(scored)

    # ── 防御动作: 增援受威胁的己方行星 ──
    for tgt in state.my_planets:
        tgt_timeline = timelines.get(tgt.id, {})
        keep_needed = tgt_timeline.get("keep_needed", 0)
        if keep_needed <= 0:
            continue

        incoming_friendly = sum(
            s for _, owner, s in ledger.get(tgt.id, []) if owner == state.player
        )
        current_defense = int(tgt.ships) + incoming_friendly
        deficit = keep_needed - current_defense
        if deficit <= 0:
            continue

        fall_turn = tgt_timeline.get("fall_turn")

        for src in state.my_planets:
            if src.id == tgt.id:
                continue
            src_timeline = timelines.get(src.id, {})
            src_keep = src_timeline.get("keep_needed", 0)
            src_available = max(0, int(src.ships) - int(src_keep))
            if src_available < SEARCH_MIN_SHIPS:
                continue

            # 尝试覆盖缺口
            for ratio in (0.5, 1.0):
                ships = max(SEARCH_MIN_SHIPS, int(src_available * ratio))
                ships = min(ships, deficit + 5)

                value, eta = value_of_reinforcement(
                    src, tgt, ships, state, timelines, ledger,
                )
                if value > 0:
                    results.append(ScoredAction(
                        source_id=src.id, target_id=tgt.id,
                        ships=ships, value=value,
                        capture_turn=None, hold_until=None,
                        eta=eta or 99, source_available=src_available,
                        source_at_risk=False, is_enemy=False,
                    ))

        # ── 撤离: 行星注定失守，保存舰船 ──
        if fall_turn is not None and current_defense < keep_needed:
            remaining_ships = int(tgt.ships)
            if remaining_ships < SEARCH_MIN_SHIPS:
                continue

            # 找最近的不会失守的己方行星
            best_target = None
            best_dist = float("inf")
            for mp in state.my_planets:
                if mp.id == tgt.id:
                    continue
                mp_timeline = timelines.get(mp.id, {})
                mp_fall = mp_timeline.get("fall_turn")
                if mp_fall is not None:
                    continue  # 目标行星也不安全
                d = math.hypot(mp.x - tgt.x, mp.y - tgt.y)
                if d < best_dist:
                    best_dist = d
                    best_target = mp

            if best_target is not None:
                value, eta = value_of_evacuation(
                    tgt, remaining_ships, best_target, state, timelines,
                )
                if value > 0:
                    results.append(ScoredAction(
                        source_id=tgt.id, target_id=best_target.id,
                        ships=remaining_ships, value=value,
                        capture_turn=None, hold_until=None,
                        eta=eta or 99, source_available=remaining_ships,
                        source_at_risk=True, is_enemy=False,
                    ))

    results.sort(key=lambda a: a.value, reverse=True)
    return results[:top_k]


def beam_search(state, ledger, timelines, beam_width=5, max_depth=2):
    """多步前瞻束搜索 —— 枚举候选动作并施加前瞻调整。

    两层评估：
    1. 一步搜索 (search_best_actions) — 基于 what-if 模拟的直接占领价值
    2. 前瞻调整 (lookahead_adjustment) — 考虑敌方反制 + 扩张链期权

    前瞻调整的核心逻辑：
    - 敌方反制威胁：敌方能否在我方增援前夺回目标？能 → 降分
    - 扩张链期权：占领此位置是否开启了新的战略机会？是 → 加分

    「敌不动我不动」—— 如果一个动作看两步之后净收益为负，就不该推荐。

    Args:
        state: GameState
        ledger: 到达账本
        timelines: 行星时间线
        beam_width: 返回的最佳动作数
        max_depth: 保留参数（当前前瞻调整已覆盖 2 步语义）

    Returns:
        list[ScoredAction]: 按调整后价值降序排列
    """
    # Step 1: 一步搜索，获取足够多的候选
    pool_size = max(beam_width * 6, 30)
    candidates = search_best_actions(
        state, ledger, timelines, top_k=pool_size,
    )

    if not candidates:
        return []

    # Step 2: 对每个候选施加前瞻调整
    for action in candidates:
        adj = lookahead_adjustment(action, state, timelines)
        action.value += adj

    # Step 3: 按调整后价值重新排序
    candidates.sort(key=lambda a: a.value, reverse=True)

    return candidates[:beam_width]


# ═══════════════════════════════════════════════════════════════════
# 多源合击搜索 & 纯搜索智能体
# ═══════════════════════════════════════════════════════════════════


@dataclass
class MultiSourceAction:
    """多源合击动作 —— 多个源行星向同一目标发射舰队。"""
    target_id: int
    sources: list  # [(source_id, ships, eta), ...]
    total_ships: int
    value: float
    capture_turn: int | None
    hold_until: int | None
    is_enemy: bool


def _find_multi_source_actions(state, ledger, timelines):
    """寻找多源合击机会 —— 单一源行星无法攻占、但联合可以的目标。

    策略：
    1. 对每个敌方/中立目标，计算至少需要多少舰船
    2. 检查是否有单一源行星能独立攻占
    3. 若不能，尝试合并最近的 2-3 个源行星的舰船
    4. 模拟合击效果，计算联合价值
    """
    multi_actions = []
    game_progress = state.step / max(1, state.episode_steps)
    max_eta = 40.0 + game_progress * 85.0

    # 收集每个源行星的可用舰船
    source_info = {}
    for src in state.my_planets:
        src_timeline = timelines.get(src.id, {})
        keep_needed = src_timeline.get("keep_needed", 0)
        reserve = min(int(src.ships), int(keep_needed))
        available = max(0, int(src.ships) - reserve)
        if available >= SEARCH_MIN_SHIPS:
            source_info[src.id] = (src, available)

    if len(source_info) < 2:
        return multi_actions  # 需要至少 2 个源行星才能合击

    all_targets = state.enemy_planets + state.neutral_planets

    for tgt in all_targets:
        if tgt.id in state.comet_ids:
            continue

        # 收集能到达此目标的源行星（按距离排序）
        reachable = []
        for sid, (src, available) in source_info.items():
            if src.id == tgt.id:
                continue
            eta = travel_time(
                src.x, src.y, src.radius,
                tgt.x, tgt.y, tgt.radius,
                max(SEARCH_MIN_SHIPS, available),
            )
            if eta <= max_eta:
                reachable.append((eta, src, available))

        if len(reachable) < 2:
            continue

        # 按距离排序
        reachable.sort(key=lambda x: x[0])

        # 估算最小攻占舰船数（用最近的源行星估算）
        closest_src = reachable[0][1]
        closest_avail = reachable[0][2]
        min_ships, _ = find_min_ships_to_capture(
            closest_src, tgt, closest_avail, state, ledger, timelines,
        )

        if min_ships is None:
            # 最近的源行星用全部舰船也打不下 → 估算一个合理的最小值
            garrison = int(tgt.ships)
            if tgt.owner != -1:
                garrison += tgt.production * 15  # 粗略估算
            min_ships = garrison + 10

        # 检查是否有单一源行星能独立完成
        any_single_can = any(avail >= min_ships for _, _, avail in reachable)
        if any_single_can:
            continue  # 已有单源能独立占领，不需要合击

        # 尝试合并最近 2-3 个源行星
        for num_sources in (2, 3):
            if num_sources > len(reachable):
                break

            selected = reachable[:num_sources]
            total_available = sum(avail for _, _, avail in selected)

            if total_available < min_ships:
                continue

            # 按比例分配舰船（优先用最近的、留最少的）
            # 简单策略：每个源行星出 min(available, min_ships * available/total_available + 5)
            sources_to_send = []
            remaining_needed = min_ships + 5  # 稍微多派一点确保拿下
            for eta, src, avail in selected:
                contribution = min(avail, max(SEARCH_MIN_SHIPS,
                                   int(remaining_needed * avail / total_available)))
                sources_to_send.append((src, contribution))
                remaining_needed -= contribution
                if remaining_needed <= 0:
                    break

            if sum(s for _, s in sources_to_send) < min_ships:
                continue

            # 模拟合击
            outcome = simulate_multi_fleet_launch(
                sources_to_send, tgt, state, ledger, timelines,
            )

            if outcome["capture_turn"] is None:
                continue

            # 计算联合价值
            is_enemy = tgt.owner not in (-1, state.player)
            neutral_contested = False
            if not is_enemy and tgt.owner == -1:
                tl = timelines.get(tgt.id, {})
                owner_at = tl.get("owner_at", {})
                for t in range(1, state.remaining_steps + 1):
                    if owner_at.get(t, -1) not in (-1, state.player):
                        neutral_contested = True
                        break
            joint_value = value_of_capture(
                tgt, outcome["capture_turn"], outcome["hold_until"],
                state.remaining_steps, outcome["total_ships"],
                is_enemy, outcome["source_at_risk"],
                neutral_contested=neutral_contested,
            )

            if joint_value <= 0:
                continue

            multi_actions.append(MultiSourceAction(
                target_id=tgt.id,
                sources=[(sid, ships, eta) for sid, ships, eta in outcome["fleets"]],
                total_ships=outcome["total_ships"],
                value=joint_value,
                capture_turn=outcome["capture_turn"],
                hold_until=outcome["hold_until"],
                is_enemy=is_enemy,
            ))

    return multi_actions


def search_agent_act(state, ledger=None, timelines=None):
    """纯搜索智能体 —— 不依赖神经网络，直接用搜索决定本回合动作。

    全流程：
    1. 构建 ledger + timelines
    2. 单源搜索 + 防御搜索
    3. 多源合击搜索
    4. 前瞻调整
    5. 贪心分配舰船（解决同一源行星被多个动作争用的问题）
    6. 返回可执行的动作列表 [(source_id, angle, ships), ...]

    Args:
        state: GameState（必需）
        ledger: 可选，若未提供则自动构建
        timelines: 可选，若未提供则自动构建

    Returns:
        list[list]: [[source_id, angle, ships], ...] 可直接提交给环境
    """
    from ..world.fleet_tracker import build_arrival_ledger
    from ..world.combat import simulate_planet_timeline
    from ..engine.interception import aim_at

    # 构建世界模型
    if ledger is None:
        ledger = build_arrival_ledger(state.fleets, state.planets)
    if timelines is None:
        timelines = {}
        for p in state.planets:
            timelines[p.id] = simulate_planet_timeline(
                p, ledger.get(p.id, []), state.player, state.remaining_steps,
            )

    # ── 第 1 层: 单源攻击 + 防御搜索 ──
    single_actions = search_best_actions(
        state, ledger, timelines, top_k=60,
    )

    # ── 第 2 层: 多源合击 ──
    multi_actions = _find_multi_source_actions(state, ledger, timelines)

    # ── 合并所有候选动作 ──
    all_candidates = []

    # 单源动作：短局跳过前瞻（扩张速度 > 战术谨慎）
    for sa in single_actions:
        if state.remaining_steps > 100:
            adj = lookahead_adjustment(sa, state, timelines)
            sa.value += adj
        all_candidates.append(("single", sa))

    # 多源动作：也加入候选池
    for ma in multi_actions:
        all_candidates.append(("multi", ma))

    # 按价值排序
    all_candidates.sort(key=lambda c: c[1].value, reverse=True)

    # ── 贪心分配舰船 ──
    planet_used = {}  # {planet_id: ships_already_committed}
    actions_out = []

    for cand_type, cand in all_candidates:
        # 跳过负价值动作（无法占领或成本大于收益）
        if cand.value <= 0:
            continue

        if cand_type == "single":
            sa = cand
            sid = sa.source_id

            # 重新计算该行星当前可用舰船
            src = next((p for p in state.my_planets if p.id == sid), None)
            if src is None:
                continue
            src_timeline = timelines.get(sid, {})
            keep_needed = src_timeline.get("keep_needed", 0)
            already_used = planet_used.get(sid, 0)
            current_available = max(0, int(src.ships) - int(keep_needed) - already_used)

            ships_to_send = min(sa.ships, current_available)
            if ships_to_send < SEARCH_MIN_SHIPS:
                continue

            # 瞄准
            tgt = next((p for p in state.planets if p.id == sa.target_id), None)
            if tgt is None:
                continue

            aim = aim_at(
                src, tgt, ships_to_send,
                state.initial_by_id, state.angular_velocity,
                state.comets, state.comet_ids,
            )
            if aim is None:
                continue

            angle, _, _, _ = aim

            # 检查太阳阻挡
            blocked, _, _ = check_path_blocked(
                src, tgt, ships_to_send, angle, float(aim[1]), state,
            )
            if blocked:
                continue

            actions_out.append([int(sid), float(angle), int(ships_to_send)])
            planet_used[sid] = already_used + ships_to_send

        elif cand_type == "multi":
            ma = cand

            # 检查合击的所有源行星是否仍有足够舰船
            valid = True
            total_committed = 0
            fleet_angles = []

            for sid, ships, _eta in ma.sources:
                src = next((p for p in state.my_planets if p.id == sid), None)
                if src is None:
                    valid = False
                    break
                src_timeline = timelines.get(sid, {})
                keep_needed = src_timeline.get("keep_needed", 0)
                already_used = planet_used.get(sid, 0)
                current_available = max(0, int(src.ships) - int(keep_needed) - already_used)

                actual_ships = min(ships, current_available)
                if actual_ships < SEARCH_MIN_SHIPS:
                    valid = False
                    break

                tgt = next((p for p in state.planets if p.id == ma.target_id), None)
                if tgt is None:
                    valid = False
                    break

                aim = aim_at(
                    src, tgt, actual_ships,
                    state.initial_by_id, state.angular_velocity,
                    state.comets, state.comet_ids,
                )
                if aim is None:
                    valid = False
                    break

                angle, _, _, _ = aim
                blocked, _, _ = check_path_blocked(
                    src, tgt, actual_ships, angle, float(aim[1]), state,
                )
                if blocked:
                    valid = False
                    break

                fleet_angles.append((sid, angle, actual_ships))
                total_committed += actual_ships

            if not valid or total_committed < SEARCH_MIN_SHIPS * len(ma.sources) * 0.5:
                continue

            for sid, angle, ships in fleet_angles:
                actions_out.append([int(sid), float(angle), int(ships)])
                planet_used[sid] = planet_used.get(sid, 0) + ships

    return actions_out
