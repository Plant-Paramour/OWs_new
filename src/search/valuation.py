"""动作价值计算 —— 评估占领一颗行星的净价值。

公式: value = productive_turns × production × swing_factor − ships_sent

- 中立行星: swing=1.0（仅我方获得产能）
- 敌方行星: swing=2.5（我方获得 + 敌方失去 + 剥夺复利 = 净 swing）
  - 2.0 = 直接零和 swing（我增 + 敌减）
  - 0.5 = 剥夺复利溢价（敌方永久失去的产能会滚雪球式影响其扩张能力）
- 彗星: 不使用此公式（自然价值太小，由 warmup 引导探索）
"""

import math
from ..engine.physics import travel_time


def value_of_capture(target, capture_turn, hold_until, remaining_steps,
                     ships_sent, is_enemy, source_at_risk=False,
                     neutral_contested=False):
    """计算占领一颗行星的净价值。

    零和博弈核心：中立星若对手也能到达 (contested)，swing=2.0
    （我方获得 + 剥夺对手未来产能 = 双倍收益）。
    """
    if capture_turn is None:
        penalty = ships_sent * 1.5 if source_at_risk else ships_sent
        return -float(penalty)

    productive_turns = hold_until - capture_turn + 1
    max_possible = remaining_steps - capture_turn
    productive_turns = min(productive_turns, max_possible)
    productive_turns = max(0, productive_turns)

    if is_enemy:
        swing = 2.5  # 敌方: 直接零和 2.0 + 剥夺复利 0.5
    elif neutral_contested:
        swing = 2.0  # contested 中立: 我不拿对手就拿 → 净 swing 2.0
    else:
        swing = 1.0  # safe 中立: 只有我方能拿到

    gross_value = productive_turns * target.production * swing

    cost = ships_sent
    if source_at_risk:
        cost *= 1.5

    return float(gross_value - cost)


def is_comet_target(target, comet_ids):
    """判断目标是否为彗星（彗星不使用占领价值公式）。"""
    return target.id in comet_ids


def compute_action_value(outcome, target, player_id, remaining_steps,
                         ships_sent, comet_ids, timelines=None):
    """一站式动作价值计算 —— 从 LaunchOutcome 直接得到价值。

    对彗星返回 0.0（由 reward shaping warmup 引导探索）。
    对中立星自动检测是否 contested（对手也能到达），若是则 swing=2.0。
    """
    if target.id in comet_ids:
        return 0.0

    if outcome.blocked or outcome.aim is None:
        return -float(ships_sent)

    if outcome.capture_turn is None:
        return -float(ships_sent)

    is_enemy = target.owner not in (-1, player_id)

    # 中立星 contested 检测：检查原时间线中敌方是否会占领此星
    neutral_contested = False
    if not is_enemy and target.owner == -1 and timelines is not None:
        tl = timelines.get(target.id, {})
        owner_at = tl.get("owner_at", {})
        for t in range(1, remaining_steps + 1):
            if owner_at.get(t, -1) not in (-1, player_id):
                neutral_contested = True
                break

    return value_of_capture(
        target, outcome.capture_turn, outcome.hold_until,
        remaining_steps, ships_sent, is_enemy,
        source_at_risk=outcome.source_at_risk,
        neutral_contested=neutral_contested,
    )


# ═══════════════════════════════════════════════════════════════════
# 多步前瞻调整 (lookahead adjustment)
# ═══════════════════════════════════════════════════════════════════


def lookahead_adjustment(action, state, timelines):
    """对候选动作施加多步前瞻调整，返回价值修正量。

    考虑两个维度：
    1. 敌方反制威胁：敌方能否在我方增援到达前夺回目标？→ 惩罚
    2. 扩张链期权：占领此位置后新开启了哪些目标？→ 奖励

    核心原则——「敌不动我不动」：
    如果一个动作会让敌方轻松反制、我方无利可图，就大幅降分。
    如果一个动作能打开战略通道、创造后续机会，就适度加分。

    Args:
        action: ScoredAction (含 capture_turn, hold_until, ships, value 等)
        state: GameState
        timelines: 当前时间线 {planet_id: dict}

    Returns:
        float: 价值修正量（正 = 比表面看起来更好, 负 = 比表面更差）
    """
    if action.capture_turn is None:
        return 0.0

    target = _find_planet_in_state(state, action.target_id)
    if target is None:
        return 0.0

    adjustment = 0.0

    # 1. 敌方反制威胁
    enemy_threat = _enemy_counter_threat(target, action, state, timelines)
    adjustment -= enemy_threat

    # 2. 扩张链期权（只对成功占领且非彗星目标）
    if action.hold_until is not None and target.id not in state.comet_ids:
        chain_bonus = _expansion_chain_bonus(target, action, state)
        adjustment += chain_bonus

    return adjustment


def _find_planet_in_state(state, planet_id):
    """在 GameState 中按 ID 查找行星。"""
    for p in state.planets:
        if p.id == planet_id:
            return p
    return None


def _enemy_counter_threat(target, action, state, timelines):
    """估算敌方反制威胁 —— 敌方能否在我方增援前夺回目标。

    只考虑能在我方增援到达前抵达目标的敌方行星（而非所有敌方行星之和），
    估算其可投入的舰船数，评估反制风险。

    返回惩罚值（越大 = 威胁越大）。
    """
    remaining = state.remaining_steps
    capture_turn = action.capture_turn

    # ── 我方最快增援时间（先算，用于截断敌方考虑范围）──
    min_reinforce_eta = float("inf")
    for mp in state.my_planets:
        if mp.id == target.id or mp.id == action.source_id:
            continue
        eta = travel_time(
            mp.x, mp.y, mp.radius,
            target.x, target.y, target.radius,
            min(int(mp.ships), 200),
        )
        if eta < min_reinforce_eta:
            min_reinforce_eta = eta

    reinforce_arrival = capture_turn + min_reinforce_eta

    # ── 收集能在我方增援前到达的敌方行星 ──
    threats = []  # [(eta, threat_ships), ...]
    for ep in state.enemy_planets:
        if ep.ships < 10:
            continue
        eta = travel_time(
            ep.x, ep.y, ep.radius,
            target.x, target.y, target.radius,
            min(int(ep.ships), 200),
        )
        enemy_arrival = capture_turn + eta
        # 只考虑能在我方增援前到达的行星
        if enemy_arrival >= reinforce_arrival and min_reinforce_eta != float("inf"):
            continue
        if eta >= remaining - capture_turn:
            continue
        # 敌方单颗行星可投入兵力（保留 30% 自守）
        threat_ships = ep.ships * 0.7
        threats.append((eta, threat_ships))

    if not threats:
        return 0.0  # 无敌方能在我方增援前到达 → 安全

    # 按到达时间排序，取最快到达的威胁
    threats.sort(key=lambda x: x[0])
    min_enemy_eta = threats[0][0]
    # 主力威胁 = 最快到达者 + 后续能在暴露窗口内到达者（折扣）
    primary_threat = threats[0][1]
    secondary_threat = 0.0
    for eta, ships in threats[1:]:
        if eta <= min_enemy_eta + 5:  # 5 回合内跟进的才算
            secondary_threat += ships * 0.3
    enemy_threat_ships = primary_threat + secondary_threat

    enemy_arrival = capture_turn + min_enemy_eta

    # ── 判定：敌方先到还是我方先到？ ──
    if enemy_arrival >= reinforce_arrival:
        return 0.0  # 我方增援先到 → 能守住

    # 敌方先到 → 评估我方防御能力
    defense_turns = max(0, enemy_arrival - capture_turn)
    our_defense = action.ships + target.production * defense_turns

    if enemy_threat_ships < our_defense * 0.9:
        return 0.0  # 我方兵力足够防守（需敌方明显优势才判定威胁）

    # 敌方可能反制成功 → 计算暴露价值
    exposure_turns = max(0, min(reinforce_arrival, action.hold_until or remaining) - enemy_arrival)
    if exposure_turns <= 0:
        return 0.0

    # 被反制期间的价值损失 (敌方获得 + 我方失去)
    denied_value = target.production * exposure_turns * 2.0
    # 惩罚上限 25%（保守估计，敌方不一定反制）
    return min(denied_value, abs(action.value) * 0.25)


def _expansion_chain_bonus(target, action, state):
    """估算扩张链期权 —— 占领 target 后新开启了哪些战略机会。

    从新位置出发，评估附近未占领行星的「跳板价值」：
    - 越近的行星期权价值越高
    - 敌方行星的期权价值 > 中立行星（剥夺效应）
    """
    if action.capture_turn is None:
        return 0.0

    remaining_after = max(0, state.remaining_steps - action.capture_turn)
    if remaining_after <= 0:
        return 0.0

    chain_value = 0.0

    for p in state.neutral_planets + state.enemy_planets:
        if p.id == target.id:
            continue
        # 使用快速距离估算
        dist = math.hypot(p.x - target.x, p.y - target.y)
        if dist > 100:
            continue

        # 检查是否比从最近的我方行星出发更近
        min_existing_dist = float("inf")
        for mp in state.my_planets:
            if mp.id == target.id:
                continue
            d = math.hypot(p.x - mp.x, p.y - mp.y)
            if d < min_existing_dist:
                min_existing_dist = d

        # 只有新位置确实更近时才有跳板价值
        if dist >= min_existing_dist * 0.85:
            continue

        proximity = 1.0 - dist / 100.0  # 越近权重越高
        is_enemy_target = p.owner not in (-1, state.player)
        swing = 2.5 if is_enemy_target else 1.0

        # 期权价值 = 产能 × 接近度 × 剩余时间 × 折扣因子
        chain_value += p.production * proximity * remaining_after * 0.06 * swing

    return chain_value


# ═══════════════════════════════════════════════════════════════════
# 防御动作价值 (增援 & 撤离)
# ═══════════════════════════════════════════════════════════════════


def value_of_reinforcement(src, tgt, ships_to_send, state, timelines, ledger):
    """评估向己方行星 tgt 增援 ships_to_send 艘舰船的价值。

    核心逻辑：行星的 keep_needed 超过当前防御力 → 面临失守风险。
    增援的价值 = 行星未来产出 × 风险覆盖比例 − 调动成本。

    Returns:
        (value, eta): 净价值和预计到达回合
    """
    tgt_timeline = timelines.get(tgt.id, {})
    keep_needed = tgt_timeline.get("keep_needed", 0)
    if keep_needed <= 0:
        return 0.0, 0

    # 当前防御力 = 驻军 + 已在路上的友军
    incoming_friendly = sum(
        s for _, owner, s in ledger.get(tgt.id, []) if owner == state.player
    )
    current_defense = int(tgt.ships) + incoming_friendly
    deficit = keep_needed - current_defense
    if deficit <= 0:
        return 0.0, 0  # 已经安全

    # 计算到达时间
    from ..engine.physics import travel_time
    eta = travel_time(
        src.x, src.y, src.radius,
        tgt.x, tgt.y, tgt.radius,
        max(1, int(ships_to_send)),
    )
    if eta >= state.remaining_steps:
        return 0.0, eta  # 来不及

    # 到达时行星还在吗？
    tgt_fall_turn = tgt_timeline.get("fall_turn")
    if tgt_fall_turn is not None and eta >= tgt_fall_turn:
        return 0.0, eta  # 到达前已失守

    # 覆盖的防御缺口
    covered = min(ships_to_send, deficit)
    coverage_ratio = covered / max(1, deficit)

    # 行星的现值（剩余生命 × 产能）
    from ..engine.prediction import comet_remaining_life
    if tgt.id in state.comet_ids:
        life = comet_remaining_life(tgt.id, state.comets)
    else:
        life = state.remaining_steps
    planet_pv = tgt.production * life

    # 增援价值 = 行星现值 × 风险覆盖 − 调动成本
    gross_value = planet_pv * coverage_ratio * 0.6
    cost = ships_to_send + eta * tgt.production * 0.3
    return float(gross_value - cost), int(eta)


def value_of_evacuation(src, ships_to_send, nearest_safe_planet, state, timelines):
    """评估从注定失守的行星撤离舰船的价值。

    如果 keep_needed > 当前舰船 + 产能积累 → 守不住 → 撤离保存实力。
    """
    src_timeline = timelines.get(src.id, {})
    fall_turn = src_timeline.get("fall_turn")
    if fall_turn is None:
        return 0.0, 0

    tgt = nearest_safe_planet
    if tgt is None:
        return 0.0, 0

    from ..engine.physics import travel_time
    eta = travel_time(
        src.x, src.y, src.radius,
        tgt.x, tgt.y, tgt.radius,
        max(1, int(ships_to_send)),
    )

    # 必须在失守前撤离
    if eta >= fall_turn:
        return 0.0, eta

    # 撤离价值 = 保存的舰船 − 运输时间成本
    saved_value = ships_to_send * 1.0
    time_cost = eta * 0.3
    return float(saved_value - time_cost), int(eta)
