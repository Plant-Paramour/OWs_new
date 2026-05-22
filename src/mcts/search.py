"""MCTS 核心搜索引擎 —— 完整动作集合 UCB 搜索。

每个 MCTS 子节点 = 一个完整的本回合动作集合（多行星同时行动）。
每轮迭代: Select(UCB1) → Simulate(full turn + deep rollout) → Backprop
"""

import math
import time
import random
from dataclasses import dataclass, field

from ..world.types import GameState
from ..world.fleet_tracker import build_arrival_ledger
from ..world.combat import simulate_planet_timeline
from .state_transition import clone_state, step_state
from .action_space import Action, enumerate_actions
from .value import (
    build_baseline_timelines, build_baseline_ledger,
    player_value_from_state, action_impact,
)


@dataclass
class MCTSNode:
    """MCTS 树节点——代表一个完整的动作集合。"""
    actions: list = field(default_factory=list)
    parent: "MCTSNode | None" = None

    visits: int = 0
    total_value: float = 0.0

    children: dict = field(default_factory=dict)

    @property
    def avg_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits


class MCTSSearch:
    """完整动作集合 MCTS 搜索引擎。

    每个根子节点是一组完整的本回合动作（所有行星的发射决定）。
    UCB1 选择最有前景的动作集合，深度 rollout 到终局附近进行评估。
    """

    def __init__(self, C: float = 1.414, time_budget_ms: int = 900):
        self.C = C
        self.time_budget_ms = time_budget_ms
        self.root = None
        self.iteration = 0
        self.root_state = None
        self.player = 0

    def search(self, root_state: GameState) -> list:
        """MCTS 搜索，返回本回合最优动作列表。"""
        start_time = time.monotonic()
        self.root_state = root_state
        self.player = root_state.player

        action_sets = self._generate_action_sets(root_state)

        if not action_sets:
            return []

        self.root = MCTSNode()
        for i, acts in enumerate(action_sets):
            child = MCTSNode(actions=acts, parent=self.root)
            self.root.children[i] = child

        self.iteration = 0

        while not self._time_up(start_time):
            self.iteration += 1

            child = self._select()
            value = self._simulate_action_set(child.actions)
            self._backpropagate(child, value)

        return self._best_actions()

    def _generate_action_sets(self, state: GameState) -> list:
        """生成候选动作集合池。

        策略：
        1. 为每个行星选最优 (source, target) 对（优先 sufficient+近距离）
        2. 生成多个变体（不同舰船规模、单行星 pass）
        3. 始终包含"全 pass"候选
        """
        all_actions = enumerate_actions(state, self.player, max_candidates=80)
        real_actions = [a for a in all_actions if not a.is_pass()]

        if not real_actions:
            return [[]]

        # 按行星分组
        by_source = {}
        for a in real_actions:
            by_source.setdefault(a.source_id, []).append(a)

        # 基线：每个行星选最优动作（优先 sufficient + 近距离）
        baseline = []
        for src_id, acts in by_source.items():
            sufficient = [a for a in acts if a.sufficient]
            if sufficient:
                sufficient.sort(key=lambda a: a.distance)
                baseline.append(sufficient[0])
            else:
                acts.sort(key=lambda a: (-a.ships, a.distance))
                baseline.append(acts[0])

        # 总是包含"全 pass"作为候选策略
        action_sets = [baseline, []]

        # 生成变体：扰动舰船规模
        for i, base_act in enumerate(baseline):
            src_id = base_act.source_id
            variants = by_source.get(src_id, [])
            for v in variants[:6]:
                if v.ships != base_act.ships:
                    variant = list(baseline)
                    variant[i] = v
                    action_sets.append(variant)

        # 试试"单行星 pass"（不发兵）
        if len(baseline) > 1:
            for i in range(len(baseline)):
                variant = [a for j, a in enumerate(baseline) if j != i]
                if variant:
                    action_sets.append(variant)

        # 去重
        seen = set()
        unique = []
        for acts in action_sets:
            key = tuple(sorted((a.source_id, a.target_id, a.ships) for a in acts))
            if key not in seen:
                seen.add(key)
                unique.append(acts)

        return unique[:60]

    def _select(self) -> MCTSNode:
        """UCB1 选择。"""
        best_child = None
        best_ucb = -float("inf")

        for key, child in self.root.children.items():
            if child.visits == 0:
                return child
            exploit = child.avg_value
            explore = self.C * math.sqrt(math.log(self.root.visits + 1) / child.visits)
            ucb = exploit + explore
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child if best_child else next(iter(self.root.children.values()))

    def _simulate_action_set(self, my_actions: list) -> float:
        """模拟完整回合：我方动作集合 + 敌方启发式 + 深度 rollout。"""
        sim_state = clone_state(self.root_state)

        my_acts = list(my_actions)

        # 敌方动作（启发式）
        enemy_acts = self._heuristic_actions(sim_state, 1, [])

        # step
        actions = {
            self.player: [[a.source_id, a.angle, a.ships] for a in my_acts],
            1: [[a.source_id, a.angle, a.ships] for a in enemy_acts],
        }
        sim_state = step_state(sim_state, actions)

        # 深度 rollout
        remaining = sim_state.remaining_steps
        if remaining <= 20:
            rd = remaining - 1
        elif remaining <= 100:
            rd = 40
        elif remaining <= 300:
            rd = 60
        else:
            rd = 50

        for _ in range(rd):
            if sim_state.remaining_steps <= 1:
                break
            my_a = self._heuristic_actions(sim_state, self.player, [])
            enemy_a = self._heuristic_actions(sim_state, 1, [])
            acts = {
                self.player: [[a.source_id, a.angle, a.ships] for a in my_a],
                1: [[a.source_id, a.angle, a.ships] for a in enemy_a],
            }
            sim_state = step_state(sim_state, acts)

        # 评估
        val = player_value_from_state(sim_state)
        my_val = val.get(self.player, 0)
        enemy_val = val.get(1 - self.player, 0)
        total = my_val + enemy_val
        if total > 0:
            advantage = (my_val - enemy_val) / total
            return (advantage + 1.0) / 2.0
        return 0.5

    def _backpropagate(self, child: MCTSNode, value: float):
        current = child
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent

    def _best_actions(self) -> list:
        """选择访问次数最多的动作集合。"""
        if not self.root or not self.root.children:
            return []

        best_child = None
        best_visits = -1
        for child in self.root.children.values():
            if child.visits > best_visits:
                best_visits = child.visits
                best_child = child

        if best_child is None:
            return []

        return [[a.source_id, a.angle, a.ships] for a in best_child.actions]

    def _heuristic_actions(self, sim_state: GameState, player: int,
                            already_committed: list) -> list:
        """快速启发式动作选择（用于 rollout）。

        只派遣足以攻占目标的兵力，不足则不派。
        """
        used = {}
        for a in already_committed:
            sid = a.source_id if hasattr(a, 'source_id') else a[0]
            s = a.ships if hasattr(a, 'ships') else (a[2] if len(a) > 2 else 0)
            used[sid] = used.get(sid, 0) + s

        actions = []
        owned = sim_state.my_planets if player == sim_state.player else sim_state.enemy_planets

        for src in owned:
            already_used = used.get(src.id, 0)
            avail = src.ships - already_used
            if avail < 10:
                continue

            best_tgt = None
            best_dist = float("inf")
            for tgt in sim_state.planets:
                if tgt.id == src.id or tgt.owner == player:
                    continue
                d = math.hypot(src.x - tgt.x, src.y - tgt.y)
                if d < best_dist:
                    best_dist = d
                    best_tgt = tgt

            if best_tgt is None:
                continue

            # 计算攻占所需兵力
            eta_est = best_dist / 3.0
            garrison = best_tgt.ships
            if best_tgt.owner != -1 and best_tgt.owner != player:
                garrison += best_tgt.production * min(eta_est, 50)
            needed = max(1, int(garrison * 1.1))

            # 兵力不足则不浪费舰船
            if avail < needed:
                continue

            ships_to_send = min(avail, needed)
            angle = math.atan2(best_tgt.y - src.y, best_tgt.x - src.x)

            from ..engine.physics import safe_angle_and_distance
            safe = safe_angle_and_distance(src.x, src.y, src.radius, best_tgt.x, best_tgt.y, best_tgt.radius)
            if safe is not None:
                angle = safe[0]

            actions.append(Action(src.id, best_tgt.id, ships_to_send, angle, 0.0, player, best_dist))
            used[src.id] = used.get(src.id, 0) + ships_to_send

        return actions

    def _time_up(self, start_time: float) -> bool:
        return (time.monotonic() - start_time) * 1000 > self.time_budget_ms - 30
