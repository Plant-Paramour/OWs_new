"""MCTS 时间线 —— 未来 N 回合的全局面投影。

核心理念：在途舰队 = 未来某回合到达某位置的确定性事件。
时间线将所有在途舰队"烘焙"进逐回合归属/驻军投影，
智能体只需关心"本回合出哪些牌 → 时间线如何变化 → 价值差"。

不再需要 step_state() 完整物理仿真。
"""

import math
from collections import defaultdict

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from ..world.fleet_tracker import build_arrival_ledger_accurate
from ..world.combat import simulate_planet_timeline
from ..world.types import GameState
from ..engine.prediction import comet_remaining_life


class Timeline:
    """全局面时间线投影 —— 纯策略视角，不包含物理仿真。

    每颗行星独立拥有未来归属/驻军时间线，
    插入新舰队到达事件后局部重算受影响行星。
    """

    def __init__(self):
        self.planet_data: dict[int, Planet] = {}
        self._baseline_arrivals: dict[int, list] = {}  # {pid: [(eta, owner, ships)]}
        self.timelines: dict[int, dict] = {}  # {pid: simulate_planet_timeline result}
        self._planet_life: dict[int, int | None] = {}
        self.remaining_steps: int = 0
        self.horizon: int = 0
        self.player: int = 0

    @classmethod
    def from_state(cls, state: GameState) -> "Timeline":
        """从 GameState 构建基线时间线。

        使用 swept-pair 精确 ETA 将所有在途舰队投影到未来回合，
        然后逐行星构建归属/驻军时间线。
        """
        tl = cls()
        tl.remaining_steps = state.remaining_steps
        tl.horizon = min(state.remaining_steps, 110)
        tl.player = state.player

        ledger = build_arrival_ledger_accurate(state)

        for planet in state.planets:
            arrivals = ledger.get(planet.id, [])
            tl._baseline_arrivals[planet.id] = list(arrivals)
            life = None
            if planet.id in state.comet_ids:
                life = comet_remaining_life(planet.id, state.comets)
            tl._planet_life[planet.id] = life
            tl.timelines[planet.id] = simulate_planet_timeline(
                planet, arrivals, state.player, tl.horizon, planet_life=life,
            )
            tl.planet_data[planet.id] = planet

        return tl

    def with_actions(self, actions_by_player: dict[int, list]) -> "Timeline":
        """返回应用新舰队发射后的新时间线（不修改原对象）。

        同时处理两个效果：
        1. 源行星 garrison 减少（影响防守）
        2. 目标行星新增到达事件（影响攻占）

        Args:
            actions_by_player: {player_id: [Action, ...]}
        """
        new_tl = Timeline()
        new_tl.remaining_steps = self.remaining_steps
        new_tl.horizon = self.horizon
        new_tl.player = self.player
        new_tl._planet_life = dict(self._planet_life)

        # 源行星 garrison 变化
        garrison_deltas: dict[int, int] = defaultdict(int)
        # 目标行星新增到达事件
        new_by_target: dict[int, list] = defaultdict(list)

        for player, actions in actions_by_player.items():
            for a in actions:
                if a.is_pass():
                    continue
                turn = a.arrival_turn
                if turn > new_tl.horizon:
                    continue
                garrison_deltas[a.source_id] -= a.ships
                new_by_target[a.target_id].append((turn, player, a.ships))

        # 构建修正后的行星数据（源行星 garrison 减少）
        new_tl.planet_data = {}
        for pid, planet in self.planet_data.items():
            delta = garrison_deltas.get(pid, 0)
            if delta != 0:
                new_ships = max(0, planet.ships + delta)
                new_tl.planet_data[pid] = Planet(
                    planet.id, planet.owner, planet.x, planet.y,
                    planet.radius, new_ships, planet.production,
                )
            else:
                new_tl.planet_data[pid] = planet

        # 基线到达事件（深拷贝）
        new_tl._baseline_arrivals = {
            k: list(v) for k, v in self._baseline_arrivals.items()
        }

        # 重算受影响行星的时间线；未受影响的行星共享引用
        affected = set(garrison_deltas.keys()) | set(new_by_target.keys())
        new_tl.timelines = dict(self.timelines)  # 浅拷贝，未受影响的共享
        for pid in affected:
            planet = new_tl.planet_data[pid]
            baseline = self._baseline_arrivals.get(pid, [])
            extra = new_by_target.get(pid, [])
            all_arrivals = sorted(baseline + extra) if extra else list(baseline)
            life = self._planet_life.get(pid)
            new_tl.timelines[pid] = simulate_planet_timeline(
                planet, all_arrivals, self.player, new_tl.horizon,
                planet_life=life,
            )

        return new_tl

    def evaluate(self) -> dict[int, float]:
        """计算逐玩家未来总价值。

        未来每回合：owner 获得该行星产值（易手行星 ×1.5 加权）。
        终局：owner 获得最终驻军（全额，直接对应胜负）。
        """
        value: dict[int, float] = {}
        for planet in self.planet_data.values():
            tl = self.timelines.get(planet.id)
            if tl is None:
                continue
            initial_owner = tl["owner_at"].get(0, -1)
            for turn in range(1, self.horizon + 1):
                owner = tl["owner_at"].get(turn)
                if owner is not None and owner >= 0:
                    w = 1.5 if owner != initial_owner else 1.0
                    value[owner] = value.get(owner, 0.0) + planet.production * w
            final_owner = tl["owner_at"].get(self.horizon)
            final_ships = tl["ships_at"].get(self.horizon, 0)
            if final_owner is not None and final_owner >= 0:
                value[final_owner] = value.get(final_owner, 0.0) + final_ships
        return value

    def evaluate_diff(self, player_a: int, player_b: int) -> float:
        """两玩家价值差 = player_a 未来总价值 - player_b 未来总价值。"""
        val = self.evaluate()
        return val.get(player_a, 0.0) - val.get(player_b, 0.0)
