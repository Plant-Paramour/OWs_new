"""What-if 模拟器 —— 评估"如果发射这支舰队会怎样"。

向到达账本注入候选舰队，重新模拟目标行星时间线，分析占领时机。
"""

import math
from dataclasses import dataclass

from ..engine.interception import aim_at, check_path_blocked
from ..world.combat import simulate_planet_timeline


@dataclass
class LaunchOutcome:
    """单次舰队发射的完整 what-if 模拟结果。"""

    aim: tuple | None
    blocked: bool
    blocker_id: int | None
    new_timeline: dict | None
    capture_turn: int | None
    hold_until: int | None
    source_at_risk: bool


def simulate_fleet_launch(src, tgt, ships_to_send, state, ledger, timelines):
    """模拟从 src 向 tgt 发射 ships_to_send 艘舰船后的结果。

    将我方舰队注入到达账本副本，重新运行 simulate_planet_timeline，
    分析占领回合和持续时间。

    Args:
        src: 源行星 (Planet)
        tgt: 目标行星 (Planet)
        ships_to_send: 发射舰船数
        state: GameState
        ledger: 到达账本 {planet_id: [(eta, owner, ships)]}
        timelines: 现有时间线 {planet_id: dict}

    Returns:
        LaunchOutcome
    """
    outcome = LaunchOutcome(
        aim=None, blocked=False, blocker_id=None,
        new_timeline=None, capture_turn=None, hold_until=None,
        source_at_risk=False,
    )

    ships = max(1, int(ships_to_send))

    # Step 1: 瞄准
    aim = aim_at(
        src, tgt, ships,
        state.initial_by_id, state.angular_velocity,
        state.comets, state.comet_ids,
    )
    if aim is None:
        return outcome
    outcome.aim = aim

    angle, turns, _pred_x, _pred_y = aim
    eta = float(turns)

    # Step 2: 路径阻挡检查
    blocked, blocker_id, _block_turn = check_path_blocked(
        src, tgt, ships, angle, eta, state,
    )
    if blocked:
        outcome.blocked = True
        outcome.blocker_id = blocker_id
        return outcome

    # Step 3: 将我方舰队注入到达账本 → 重新模拟目标时间线
    original_arrivals = list(ledger.get(tgt.id, []))
    new_arrivals = original_arrivals + [(eta, state.player, ships)]
    sim_horizon = max(1, state.remaining_steps)
    new_timeline = simulate_planet_timeline(
        tgt, new_arrivals, state.player, sim_horizon,
    )
    outcome.new_timeline = new_timeline

    # Step 4: 分析占领时机
    capture_turn = None
    hold_until = None

    for turn in range(1, sim_horizon + 1):
        owner = new_timeline["owner_at"].get(turn, -1)
        if capture_turn is None and owner == state.player:
            capture_turn = turn
            hold_until = turn
        elif capture_turn is not None and owner == state.player:
            hold_until = turn
        elif capture_turn is not None and owner != state.player:
            break

    outcome.capture_turn = capture_turn
    outcome.hold_until = hold_until

    # Step 5: 源行星安全检测
    src_timeline = timelines.get(src.id, {})
    src_keep = src_timeline.get("keep_needed", 0)
    remaining = max(0, int(src.ships) - ships)
    outcome.source_at_risk = remaining < src_keep

    return outcome


def simulate_multi_fleet_launch(sources, tgt, state, ledger, timelines):
    """模拟多个源行星同时向 tgt 发射舰队的合击结果。

    将所有我方舰队注入到达账本副本，重新运行行星时间线，
    分析联合占领的时机和持续时间。

    Args:
        sources: list of (src_planet, ships_to_send) tuples
        tgt: 目标行星
        state: GameState
        ledger: 到达账本
        timelines: 现有时间线

    Returns:
        dict: {
            'capture_turn': int | None,
            'hold_until': int | None,
            'total_ships': int,
            'source_at_risk': bool,
            'fleets': [(src_id, ships, eta), ...],  # 成功发射的舰队
            'blocked_sources': [src_id, ...],
        }
    """
    import math
    new_arrivals = list(ledger.get(tgt.id, []))
    total_ships = 0
    source_at_risk = False
    launched = []
    blocked_sources = []

    for src, ships in sources:
        ships = max(1, int(ships))
        aim = aim_at(
            src, tgt, ships,
            state.initial_by_id, state.angular_velocity,
            state.comets, state.comet_ids,
        )
        if aim is None:
            continue

        angle, turns, _, _ = aim
        eta = float(turns)

        blocked, _, _ = check_path_blocked(src, tgt, ships, angle, eta, state)
        if blocked:
            blocked_sources.append(src.id)
            continue

        new_arrivals.append((eta, state.player, ships))
        total_ships += ships
        launched.append((src.id, ships, eta))

        src_timeline = timelines.get(src.id, {})
        keep_needed = src_timeline.get("keep_needed", 0)
        remaining = max(0, int(src.ships) - ships)
        if remaining < keep_needed:
            source_at_risk = True

    if total_ships == 0:
        return {
            "capture_turn": None, "hold_until": None,
            "total_ships": 0, "source_at_risk": False,
            "fleets": [], "blocked_sources": blocked_sources,
        }

    sim_horizon = max(1, state.remaining_steps)
    new_timeline = simulate_planet_timeline(
        tgt, new_arrivals, state.player, sim_horizon,
    )

    capture_turn = None
    hold_until = None
    for turn in range(1, sim_horizon + 1):
        owner = new_timeline["owner_at"].get(turn, -1)
        if capture_turn is None and owner == state.player:
            capture_turn = turn
            hold_until = turn
        elif capture_turn is not None and owner == state.player:
            hold_until = turn
        elif capture_turn is not None and owner != state.player:
            break

    return {
        "capture_turn": capture_turn,
        "hold_until": hold_until,
        "total_ships": total_ships,
        "source_at_risk": source_at_risk,
        "fleets": launched,
        "blocked_sources": blocked_sources,
        "new_timeline": new_timeline,
    }


def find_min_ships_to_capture(src, tgt, max_ships, state, ledger, timelines):
    """二分搜索找到从 src 攻占 tgt 所需的最小舰船数。

    Args:
        src: 源行星
        tgt: 目标行星
        max_ships: 可用舰船上限 (available)
        state, ledger, timelines: 世界模型状态

    Returns:
        (min_ships, outcome) 或 (None, None) 若无法攻占
    """
    if max_ships < 8:
        return None, None

    # 先检查上限能否攻占
    hi_outcome = simulate_fleet_launch(src, tgt, max_ships, state, ledger, timelines)
    if hi_outcome.capture_turn is None or hi_outcome.blocked:
        return None, None

    lo = 8
    hi = max_ships
    best_outcome = hi_outcome

    while lo < hi:
        mid = (lo + hi) // 2
        outcome = simulate_fleet_launch(src, tgt, mid, state, ledger, timelines)
        if outcome.capture_turn is not None:
            hi = mid
            best_outcome = outcome
        else:
            lo = mid + 1

    return lo, best_outcome
