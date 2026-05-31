"""MCTS 博弈树搜索引擎 —— minimax 深度搜索。

二层树结构：
  Level 0 (root, 我方回合): 我方动作集 → Level 1 (敌方回合): 敌方回应 → 评估
每轮迭代: Select(UCB1, 我方max/敌方min) → Expand → Simulate → Backprop
"""

import math
import time
import random
from dataclasses import dataclass, field

from ..world.types import GameState
from .state_transition import clone_state, step_state
from .action_space import Action, enumerate_actions
from .value import build_baseline_timelines, player_value_from_timelines

# 敌方回应采样上限
MAX_ENEMY_CHILDREN = 12


@dataclass
class MCTSNode:
    """MCTS 树节点——代表一个完整的动作集合。"""
    actions: list = field(default_factory=list)
    parent: "MCTSNode | None" = None
    is_enemy: bool = False

    visits: int = 0
    total_value: float = 0.0

    children: dict = field(default_factory=dict)

    @property
    def avg_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    @property
    def minimax_value(self) -> float:
        """当前 minimax 估值。
        敌方节点 = avg_value（直接评估）。
        我方节点 = min over 敌方子节点（最坏情况）。
        """
        if self.is_enemy or not self.children:
            return self.avg_value
        worst = float("inf")
        for child in self.children.values():
            if child.visits > 0:
                worst = min(worst, child.avg_value)
        return worst if worst != float("inf") else self.avg_value


class MCTSSearch:
    """Minimax MCTS 搜索引擎。

    我方节点选 max UCB，敌方节点选 min UCB。
    每次模拟：步进我方+敌方动作 → 短 rollout → 时间线评估。
    """

    def __init__(self, C: float = 100, time_budget_ms: int = 950):
        self.C = C
        self.time_budget_ms = time_budget_ms
        self.root = None
        self.iteration = 0
        self.root_state = None
        self.player = 0
        self.enemy = 1
        self._fixed_enemy_sets = []  # 所有我方节点共享的固定敌方回应

    def search(self, root_state: GameState) -> list:
        start_time = time.monotonic()
        self.root_state = root_state
        self.player = root_state.player
        self.enemy = 1 - self.player

        # 生成我方动作集
        my_action_sets = self._generate_action_sets(root_state, self.player)
        if not my_action_sets:
            return []

        # 预生成敌方全部合法动作，固定顺序供所有我方节点共享比较
        enemy_all = enumerate_actions(root_state, self.enemy, max_candidates=80)
        enemy_real = [a for a in enemy_all if not a.is_pass()]
        all_enemy_sets = self._build_action_sets(enemy_real)
        if not all_enemy_sets:
            all_enemy_sets = [[]]
        # 固定顺序：打乱一次后固定，所有我方节点共享同一套敌方回应
        rng = random.Random(42)
        rng.shuffle(all_enemy_sets)
        self._fixed_enemy_sets = all_enemy_sets[:MAX_ENEMY_CHILDREN]

        # 构建二层树
        self.root = MCTSNode()
        for i, acts in enumerate(my_action_sets):
            child = MCTSNode(actions=acts, parent=self.root, is_enemy=False)
            self.root.children[i] = child

        self.iteration = 0

        while not self._time_up(start_time):
            self.iteration += 1

            # 1. Select: 遍历二层树到叶子
            my_node, enemy_node = self._select()
            if my_node is None:
                continue

            # 2. Expand: 为敌方节点层添加新回应
            if enemy_node is None:
                enemy_node = self._expand_enemy(my_node)

            if enemy_node is None:
                continue

            # 3. Simulate
            value = self._evaluate(my_node.actions, enemy_node.actions)

            # 4. Backprop
            self._backpropagate(enemy_node, my_node, value)

        return self._best_actions()

    # ── 动作集生成 ───────────────────────────────────────────

    def _greedy_allocate(self, actions, source_available):
        """全局贪心分配：同一源舰船不超支，同一目标足够覆盖后不再追加。

        只分配 sufficient 的动作——舰船不足以攻占目标的动作不纳入动作集。
        非 sufficient 的动作由 _joint_strike_variants 专门探索合击。
        """
        used = {}
        covered_targets = set()
        result = []
        for a in actions:
            if not a.sufficient:
                continue
            sid, tid = a.source_id, a.target_id
            avail = source_available.get(sid, 0)
            if a.ships > avail - used.get(sid, 0):
                continue
            if tid in covered_targets:
                continue
            used[sid] = used.get(sid, 0) + a.ships
            result.append(a)
            covered_targets.add(tid)
        return result

    def _quick_action_score(self, action, comet_ids):
        """快速启发式评分，用于全局贪心排序。高分优先分配。"""
        score = 20.0
        if action.sufficient:
            score += 30.0
        if action.target_id not in comet_ids:
            score += 10.0
        score -= action.distance * 0.2
        if action.needed > 0 and action.ships > 0:
            score += (action.needed / action.ships) * 15.0
        return score

    def _generate_action_sets(self, state: GameState, player: int) -> list:
        """生成候选动作集合池 —— 全局贪心分配 + 随机扰动。

        所有候选动作全局排序后贪心分配，同一目标被足够兵力覆盖后
        自然跳过冗余动作。随机打乱顺序产生多样化出牌组合，
        MCTS 评估每种组合对应的场面，选出 maximin 最优。
        """
        all_actions = enumerate_actions(state, player, max_candidates=80)
        real_actions = [a for a in all_actions if not a.is_pass()]

        if not real_actions:
            return [[]]

        # 可用舰船 = actions 中每源的最大 ship count
        # （enumerate_actions 已按 ships - keep_needed 限制，max ship = available）
        source_available = {}
        for a in real_actions:
            sid = a.source_id
            source_available[sid] = max(source_available.get(sid, 0), a.ships)

        comet_ids = state.comet_ids
        for a in real_actions:
            a._heuristic_score = self._quick_action_score(a, comet_ids)

        action_sets = [[]]  # 始终包含"全部 pass"

        # 1. 启发式最优优先 → 基线动作集
        sorted_actions = sorted(real_actions, key=lambda a: a._heuristic_score, reverse=True)
        baseline = self._greedy_allocate(sorted_actions, dict(source_available))
        if baseline:
            action_sets.append(baseline)

        # 2. 随机顺序扰动 → 多样化目标分配（C→D 而非 C→B）
        for seed in range(10):
            rng = random.Random(seed * 137 + 42)
            shuffled = list(real_actions)
            rng.shuffle(shuffled)
            variant = self._greedy_allocate(shuffled, dict(source_available))
            if variant and variant != baseline:
                action_sets.append(variant)

        # 3. 单源最优（每个源独立出最好的牌，用于探索）
        by_source = {}
        for a in real_actions:
            by_source.setdefault(a.source_id, []).append(a)
        for src_id, acts in by_source.items():
            acts.sort(key=lambda a: a._heuristic_score, reverse=True)
            single = self._greedy_allocate(acts, dict(source_available))
            if single and single != baseline:
                action_sets.append(single)

        # 4. 合击变体：无单源能独立攻占时需要多源联合
        action_sets.extend(self._joint_strike_variants(real_actions, source_available))

        # 去重
        seen = set()
        unique = []
        for acts in action_sets:
            key = tuple(sorted((a.source_id, a.target_id, a.ships) for a in acts))
            if key not in seen:
                seen.add(key)
                unique.append(acts)

        return unique[:50]

    def _joint_strike_variants(self, real_actions: list, source_available: dict) -> list:
        """多星合击变体：对无单源能独立攻占的目标，组合 2 源联合出兵。"""
        variants = []

        by_target = {}
        for a in real_actions:
            by_target.setdefault(a.target_id, []).append(a)

        for tid, actions in by_target.items():
            if any(a.sufficient for a in actions):
                continue

            src_best = {}
            for a in actions:
                if a.source_id not in src_best or a.ships > src_best[a.source_id].ships:
                    src_best[a.source_id] = a

            if len(src_best) < 2:
                continue

            needed = max(a.needed for a in src_best.values())
            src_list = list(src_best.items())[:4]

            for si in range(len(src_list)):
                for sj in range(si + 1, len(src_list)):
                    src_a_id = src_list[si][0]
                    src_b_id = src_list[sj][0]

                    acts_a = [a for a in actions if a.source_id == src_a_id][:4]
                    acts_b = [a for a in actions if a.source_id == src_b_id][:4]

                    for aa in acts_a:
                        for ab in acts_b:
                            if aa.ships + ab.ships < needed:
                                continue
                            if aa.ships > source_available.get(src_a_id, 0):
                                continue
                            if ab.ships > source_available.get(src_b_id, 0):
                                continue
                            variants.append([aa, ab])

        return variants[:15]

    def _build_action_sets(self, actions: list) -> list:
        """从动作列表构建动作集合（敌方用，同样使用全局贪心分配）。"""
        if not actions:
            return [[]]

        # 可用舰船 = actions 中每源的最大 ship count
        source_available = {}
        for a in actions:
            source_available[a.source_id] = max(source_available.get(a.source_id, 0), a.ships)

        for a in actions:
            a._heuristic_score = 30.0 if a.sufficient else 10.0
            a._heuristic_score -= a.distance * 0.2

        sets = [[]]

        sorted_actions = sorted(actions, key=lambda a: a._heuristic_score, reverse=True)
        baseline = self._greedy_allocate(sorted_actions, dict(source_available))
        if baseline:
            sets.append(baseline)

        for seed in range(6):
            rng = random.Random(seed * 251 + 17)
            shuffled = list(actions)
            rng.shuffle(shuffled)
            variant = self._greedy_allocate(shuffled, dict(source_available))
            if variant and variant != baseline:
                sets.append(variant)

        seen = set()
        unique = []
        for acts in sets:
            key = tuple(sorted((a.source_id, a.target_id, a.ships) for a in acts))
            if key not in seen:
                seen.add(key)
                unique.append(acts)

        return unique[:30]

    # ── 树遍历 ────────────────────────────────────────────────

    def _select(self) -> tuple:
        """二层树选择：我方层 max UCB → 敌方层 min UCB。

        Returns:
            (my_node, enemy_node_or_None)
        """
        # Level 1: 我方节点（max UCB）
        my_node = self._select_child(self.root, maximize=True)
        if my_node is None:
            return None, None
        if my_node.visits == 0:
            return my_node, None  # 未探索过，需要扩展敌方回应

        # Level 2: 敌方节点（min UCB）
        if not my_node.children:
            return my_node, None  # 需要扩展

        enemy_node = self._select_child(my_node, maximize=False)
        return my_node, enemy_node

    def _select_child(self, parent: MCTSNode, maximize: bool) -> MCTSNode | None:
        """选择 UCB 最优子节点。maximize=True 选最大，False 选最小（敌方视角）。

        使用 minimax_value 作为 exploit，确保博弈树估值正确传导。
        """
        if not parent.children:
            return None

        best = None
        best_ucb = -float("inf") if maximize else float("inf")

        for child in parent.children.values():
            if child.visits == 0:
                return child  # 优先探索未访问节点
            exploit = child.minimax_value
            explore = self.C * math.sqrt(math.log(parent.visits + 1) / child.visits)
            ucb = exploit + explore if maximize else exploit - explore
            if (maximize and ucb > best_ucb) or (not maximize and ucb < best_ucb):
                best_ucb = ucb
                best = child

        return best

    def _expand_enemy(self, my_node: MCTSNode) -> MCTSNode | None:
        """为我方节点扩展固定的敌方回应子节点。

        所有我方节点按相同顺序使用 self._fixed_enemy_sets，
        确保 minimax 比较建立在一致的敌方回应基础上。
        """
        max_children = min(len(self._fixed_enemy_sets), MAX_ENEMY_CHILDREN)
        next_idx = len(my_node.children)
        if next_idx >= max_children:
            return None

        enemy_acts = self._fixed_enemy_sets[next_idx]
        child = MCTSNode(actions=enemy_acts, parent=my_node, is_enemy=True)
        my_node.children[next_idx] = child
        return child

    @staticmethod
    def _action_key(actions: list) -> tuple:
        return tuple(sorted((a.source_id, a.target_id, a.ships) for a in actions))

    # ── 模拟与评估 ────────────────────────────────────────────

    def _evaluate(self, my_actions: list, enemy_actions: list) -> float:
        """步进状态 → 时间线评估 → 原始未来价值差 (my_val - enemy_val)。

        不做归一化——价值差直接决定输赢，任何数学变换都会破坏决策信号。
        额外尝试 0/3/6 回合双方 pass 后的局面，取最优值。
        """
        sim_state = clone_state(self.root_state)

        actions = {
            self.player: [[a.source_id, a.angle, a.ships] for a in my_actions],
            self.enemy: [[a.source_id, a.angle, a.ships] for a in enemy_actions],
        }
        sim_state = step_state(sim_state, actions)

        best_val = self._eval_state(sim_state)

        for wait_turns in (3, 6):
            if wait_turns >= sim_state.remaining_steps:
                continue
            wait_state = clone_state(sim_state)
            for _ in range(wait_turns):
                wait_state = step_state(wait_state, {0: [], 1: []})
            wait_val = self._eval_state(wait_state)
            if wait_val > best_val:
                best_val = wait_val

        return best_val

    def _eval_state(self, state: GameState) -> float:
        timelines = build_baseline_timelines(state)
        val = player_value_from_timelines(timelines, state.planets, state.remaining_steps)
        my_val = val.get(self.player, 0)
        enemy_val = val.get(self.enemy, 0)
        return my_val - enemy_val

    # ── 回溯 ──────────────────────────────────────────────────

    def _backpropagate(self, enemy_node: MCTSNode, my_node: MCTSNode, value: float):
        """二层回溯。敌方节点累加评估值；我方节点只计访问次数。

        minimax_value 由属性实时从敌方子节点计算，不依赖 total_value。
        """
        enemy_node.visits += 1
        enemy_node.total_value += value
        my_node.visits += 1
        self.root.visits += 1

    # ── 最优动作选择 ──────────────────────────────────────────

    def _best_actions(self) -> list:
        """选 minimax 最优动作集：我方节点中 minimax_value 最高者。

        对每个我方节点，其 minimax_value = min over 敌方回应（最坏情况估值）。
        选最坏情况下最好的动作（maximin）。
        """
        if not self.root or not self.root.children:
            return []

        best_child = None
        best_value = -float("inf")
        for child in self.root.children.values():
            if child.visits > 0 and child.minimax_value > best_value:
                best_value = child.minimax_value
                best_child = child

        if best_child is None:
            for child in self.root.children.values():
                if child.actions:
                    return [[a.source_id, a.angle, a.ships] for a in child.actions]
            return []

        return [[a.source_id, a.angle, a.ships] for a in best_child.actions]

    # ── 启发式动作（rollout 用）─────────────────────────────────

    def _heuristic_actions(self, sim_state: GameState, player: int,
                            already_committed: list) -> list:
        """快速启发式动作选择。只派遣足以攻占目标的兵力。"""
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

            eta_est = best_dist / 3.0
            garrison = best_tgt.ships
            if best_tgt.owner != -1 and best_tgt.owner != player:
                garrison += best_tgt.production * min(eta_est, 50)
            needed = max(1, int(garrison * 1.1))

            if avail < needed:
                continue

            ships_to_send = min(avail, needed)
            angle = math.atan2(best_tgt.y - src.y, best_tgt.x - src.x)

            from ..engine.physics import safe_angle_and_distance
            safe = safe_angle_and_distance(src.x, src.y, src.radius,
                                           best_tgt.x, best_tgt.y, best_tgt.radius)
            if safe is not None:
                angle = safe[0]

            actions.append(Action(src.id, best_tgt.id, ships_to_send, angle, 0.0, player, best_dist))
            used[src.id] = used.get(src.id, 0) + ships_to_send

        return actions

    def _time_up(self, start_time: float) -> bool:
        return (time.monotonic() - start_time) * 1000 > self.time_budget_ms - 50
