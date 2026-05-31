"""MCTS 博弈树搜索引擎 —— minimax 回合策略搜索。

二层树结构：
  Level 0 (root, 我方回合): 我方动作集 → Level 1 (敌方回合): 敌方回应 → 评估

核心理念：不重复物理仿真。时间线（Timeline）将所有在途舰队 baked into
未来归属投影，搜索只需"插入卡牌 → 重算时间线 → 评估价值差"。
"""

import math
import time
import random
from dataclasses import dataclass, field

from ..world.types import GameState
from .timeline import Timeline
from .action_space import Action, enumerate_actions

MAX_ENEMY_CHILDREN = 12


@dataclass
class MCTSNode:
    """MCTS 树节点——代表一个完整的动作集合。"""
    actions: list = field(default_factory=list)
    parent: "MCTSNode | None" = None
    is_enemy: bool = False

    visits: int = 0
    total_value: float = 0.0
    _pending: int = 0

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
    """Minimax MCTS 搜索引擎 —— 基于时间线的纯粹回合策略。

    我方节点选 max UCB，敌方节点选 min UCB。
    每次模拟：时间线插入我方+敌方动作 → 重算归属 → 评估价值差。
    """

    def __init__(self, C: float = 100, time_budget_ms: int = 950):
        self.C = C
        self.time_budget_ms = time_budget_ms
        self.root = None
        self.iteration = 0
        self.root_state = None
        self.root_timeline: Timeline | None = None
        self.player = 0
        self.enemy = 1
        self._fixed_enemy_sets = []
        self._enemy_action_pool = []
        self._enemy_source_available: dict[int, int] = {}

    def search(self, root_state: GameState) -> list:
        start_time = time.monotonic()
        self.root_state = root_state
        self.player = root_state.player
        self.enemy = 1 - self.player

        self.root_timeline = Timeline.from_state(root_state)
        baseline_diff = self.root_timeline.evaluate_diff(self.player, self.enemy)

        # ── 提取 swing 目标：即将被敌方占领的行星（反抢机会）──
        swing_targets = set()
        for pid, tl in self.root_timeline.timelines.items():
            planet = self.root_timeline.planet_data[pid]
            if planet.owner == -1:  # 中立星 → 检查是否将被敌方占领
                for turn in range(1, min(30, self.root_timeline.horizon)):
                    owner = tl["owner_at"].get(turn)
                    if owner == self.enemy:
                        swing_targets.add(pid)
                        break
            elif planet.owner == self.player:  # 我方星 → 检查是否会失守
                fall_turn = tl.get("fall_turn")
                if fall_turn is not None and fall_turn < 30:
                    swing_targets.add(pid)

        # ── 物理计算只跑一次：生成全部动作，距离过滤在内存中做 ──
        my_all = enumerate_actions(root_state, self.player, max_candidates=120)
        my_real = [a for a in my_all if not a.is_pass()]

        def filter_by_dist(actions, max_dist):
            return [a for a in actions if a.distance <= max_dist]

        my_sets_close = self._generate_action_sets_from_pool(
            filter_by_dist(my_real, 30.0), root_state, swing_targets)
        my_sets_medium = self._generate_action_sets_from_pool(
            filter_by_dist(my_real, 60.0), root_state, swing_targets)
        my_sets_far = self._generate_action_sets_from_pool(my_real, root_state, swing_targets)

        # 合并去重，计算评分和最大距离（渐进式加宽用）
        seen = set()
        self._all_set_entries = []  # [(score, max_distance, actions), ...]
        for pool in (my_sets_close, my_sets_medium, my_sets_far):
            for acts in pool:
                key = tuple(sorted((a.source_id, a.target_id, a.ships) for a in acts))
                if key in seen:
                    continue
                seen.add(key)
                score = sum(getattr(a, '_heuristic_score', 0) for a in acts)
                max_d = max((a.distance for a in acts), default=0.0)
                self._all_set_entries.append((score, max_d, acts))
        # 按评分降序（空集排最前，高评分集优先被渐进式加宽选中）
        empty_entries = [e for e in self._all_set_entries if not e[2]]
        non_empty = [e for e in self._all_set_entries if e[2]]
        non_empty.sort(key=lambda x: x[0], reverse=True)
        self._all_set_entries = empty_entries + non_empty

        # ── 敌方动作池（物理预计算，自适应回应只用 re-score + greedy）──
        enemy_all = enumerate_actions(root_state, self.enemy, max_candidates=80)
        self._enemy_action_pool = [a for a in enemy_all if not a.is_pass()]
        self._enemy_source_available = {}
        for a in self._enemy_action_pool:
            self._enemy_source_available[a.source_id] = max(
                self._enemy_source_available.get(a.source_id, 0), a.ships)

        all_enemy_sets = self._build_action_sets(self._enemy_action_pool)
        self._fixed_enemy_sets = all_enemy_sets[:4] if all_enemy_sets else [[]]

        # ── 初始构建（少量近距离高评分集）──
        self.root = MCTSNode()
        self._active_my_sets = self._get_progressive_active_sets()
        self._rebuild_root_children()
        self._last_widen_size = len(self._active_my_sets)

        self.iteration = 0

        while not self._time_up(start_time):
            self.iteration += 1

            # 渐进式加宽：每 6 次迭代检查是否需要扩充动作集
            if self.iteration % 6 == 0:
                new_active = self._get_progressive_active_sets()
                if len(new_active) > self._last_widen_size:
                    self._active_my_sets = new_active
                    self._rebuild_root_children()
                    self._last_widen_size = len(new_active)

            my_node, enemy_node = self._select()
            if my_node is None:
                continue

            if enemy_node is None:
                enemy_node = self._expand_enemy(my_node)

            if enemy_node is None:
                continue

            value = self._evaluate(my_node.actions, enemy_node.actions)
            self._backpropagate(enemy_node, my_node, value)

        return self._best_actions(baseline_diff)

    def _rebuild_root_children(self):
        """根据当前活跃动作集重建 root 的子节点，保留已有统计。"""
        existing = {}
        for child in self.root.children.values():
            key = tuple(sorted((a.source_id, a.target_id, a.ships) for a in child.actions))
            existing[key] = child

        self.root.children.clear()
        for i, acts in enumerate(self._active_my_sets):
            key = tuple(sorted((a.source_id, a.target_id, a.ships) for a in acts))
            if key in existing:
                self.root.children[i] = existing[key]
            else:
                self.root.children[i] = MCTSNode(actions=acts, parent=self.root, is_enemy=False)

    def _get_progressive_active_sets(self) -> list:
        """基于 root 访问次数渐进式返回动作集子集。

        两个维度同时渐进：
        1. 距离上限随访问量递增——早期聚焦近处战术
        2. 动作集数量随访问量递增——后期展开全局战略

        动作集已按启发式评分降序排列，渐进式加宽自然优先探索高质量集。
        """
        v = self.root.visits

        if v < 20:
            max_dist, max_sets = 30.0, 8
        elif v < 50:
            max_dist, max_sets = 38.0, 14
        elif v < 100:
            max_dist, max_sets = 50.0, 22
        elif v < 180:
            max_dist, max_sets = 65.0, 32
        elif v < 300:
            max_dist, max_sets = 85.0, 45
        elif v < 500:
            max_dist, max_sets = float("inf"), 60
        else:
            max_dist, max_sets = float("inf"), len(self._all_set_entries)

        result = []
        for _score, max_d, acts in self._all_set_entries:
            if max_d > max_dist:
                continue
            result.append(acts)
            if len(result) >= max_sets:
                break

        return result

    # ── 动作集生成 ───────────────────────────────────────────

    def _greedy_allocate(self, actions, source_available):
        """全局贪心分配：同一源舰船不超支，同一目标足够覆盖后不再追加。"""
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

    def _quick_action_score(self, action, comet_ids, swing_targets=None):
        """快速启发式评分，用于全局贪心排序。

        swing_targets: 即将被敌方占领的行星集合——反抢机会价值极高。
        """
        score = 20.0
        if action.sufficient:
            score += 30.0
        if action.target_id not in comet_ids:
            score += 10.0
        score -= action.distance * 0.2
        # 产能加权：高产能行星长期价值远超低产能星（prod=10 × 100回合 = 1000+ 资源）
        score += action.target_production * 5.0
        if action.needed > 0 and action.ships > 0:
            score += (action.needed / action.ships) * 15.0
        # 反抢 bonus：敌方投入舰船攻占 → 我方轻取 → 净 swing 极大
        if swing_targets and action.target_id in swing_targets:
            score += 50.0
        return score

    def _generate_action_sets_from_pool(self, real_actions: list, state: GameState, swing_targets: set = None) -> list:
        """从预计算的动作池生成候选动作集合 —— 纯内存操作，无物理计算。"""
        if not real_actions:
            return [[]]

        source_available = {}
        for a in real_actions:
            sid = a.source_id
            source_available[sid] = max(source_available.get(sid, 0), a.ships)

        comet_ids = state.comet_ids
        for a in real_actions:
            a._heuristic_score = self._quick_action_score(a, comet_ids, swing_targets)
            a._efficiency = a._heuristic_score / max(1, a.ships)
            a._priority = 2.0 if a.target_id not in comet_ids else 0.5

        action_sets = [[]]

        sorted_by_score = sorted(real_actions, key=lambda a: a._heuristic_score, reverse=True)
        baseline = self._greedy_allocate(sorted_by_score, dict(source_available))
        if baseline:
            action_sets.append(baseline)

        sorted_by_efficiency = sorted(real_actions, key=lambda a: a._efficiency, reverse=True)
        eff_set = self._greedy_allocate(sorted_by_efficiency, dict(source_available))
        if eff_set and eff_set != baseline:
            action_sets.append(eff_set)

        sorted_by_distance = sorted(real_actions, key=lambda a: a.distance)
        dist_set = self._greedy_allocate(sorted_by_distance, dict(source_available))
        if dist_set and dist_set not in (baseline, eff_set):
            action_sets.append(dist_set)

        sorted_by_priority = sorted(real_actions, key=lambda a: (a._priority, a._heuristic_score), reverse=True)
        pri_set = self._greedy_allocate(sorted_by_priority, dict(source_available))
        if pri_set and pri_set not in (baseline, eff_set, dist_set):
            action_sets.append(pri_set)

        # 产能优先：高产能星排最前 → 贪心分配自然集中兵力占大星
        sorted_by_prod = sorted(real_actions, key=lambda a: (
            -a.target_production, -a._heuristic_score
        ))
        prod_set = self._greedy_allocate(sorted_by_prod, dict(source_available))
        if prod_set and prod_set not in action_sets:
            action_sets.append(prod_set)

        # ── 反抢优先排序：swing 目标排最前 ──
        if swing_targets:
            sorted_by_swing = sorted(real_actions, key=lambda a: (
                0 if a.target_id in swing_targets else 1,
                -a._heuristic_score
            ))
            swing_set = self._greedy_allocate(sorted_by_swing, dict(source_available))
            if swing_set and swing_set not in action_sets:
                action_sets.append(swing_set)

        for seed in range(8):
            rng = random.Random(seed * 137 + 42)
            shuffled = list(real_actions)
            rng.shuffle(shuffled)
            variant = self._greedy_allocate(shuffled, dict(source_available))
            if variant and variant != baseline:
                action_sets.append(variant)

        by_source = {}
        for a in real_actions:
            by_source.setdefault(a.source_id, []).append(a)
        for src_id, acts in by_source.items():
            acts.sort(key=lambda a: a._heuristic_score, reverse=True)
            single = self._greedy_allocate(acts, dict(source_available))
            if single and single != baseline:
                action_sets.append(single)

        # ── 专用反抢动作集：每个 swing 目标找最佳单源/合击 ──
        if swing_targets:
            for tid in swing_targets:
                swing_actions = [a for a in real_actions if a.target_id == tid]
                if not swing_actions:
                    continue
                swing_actions.sort(key=lambda a: a._heuristic_score, reverse=True)
                recapture = self._greedy_allocate(swing_actions, dict(source_available))
                if recapture:
                    action_sets.append(recapture)

        action_sets.extend(self._joint_strike_variants(real_actions, source_available))

        seen = set()
        unique = []
        for acts in action_sets:
            key = tuple(sorted((a.source_id, a.target_id, a.ships) for a in acts))
            if key not in seen:
                seen.add(key)
                unique.append(acts)

        return unique[:50]

    def _generate_action_sets(self, state: GameState, player: int, max_distance: float | None = None) -> list:
        """生成候选动作集合池（含物理计算，用于非渐进式场景）。"""
        all_actions = enumerate_actions(state, player, max_candidates=80, max_distance=max_distance)
        real_actions = [a for a in all_actions if not a.is_pass()]
        return self._generate_action_sets_from_pool(real_actions, state)

    def _joint_strike_variants(self, real_actions: list, source_available: dict) -> list:
        """多星合击变体。"""
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
        """从动作列表构建动作集合（敌方用）。"""
        if not actions:
            return [[]]

        source_available = {}
        for a in actions:
            source_available[a.source_id] = max(source_available.get(a.source_id, 0), a.ships)

        for a in actions:
            a._heuristic_score = 30.0 if a.sufficient else 10.0
            a._heuristic_score -= a.distance * 0.2
            a._heuristic_score += a.target_production * 5.0

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
        """二层树选择：我方层 max UCB → 敌方层 min UCB。"""
        my_node = self._select_child(self.root, maximize=True)
        if my_node is None:
            return None, None
        if my_node.visits == 0:
            return my_node, None

        if not my_node.children:
            return my_node, None

        enemy_node = self._select_child(my_node, maximize=False)
        return my_node, enemy_node

    def _select_child(self, parent: MCTSNode, maximize: bool) -> MCTSNode | None:
        """选择 UCB 最优子节点（含 virtual loss 避免重复选同一条路径）。"""
        if not parent.children:
            return None

        best = None
        best_ucb = -float("inf") if maximize else float("inf")

        for child in parent.children.values():
            if child.visits == 0:
                child._pending += 1
                return child
            effective_n = child.visits + child._pending
            exploit = child.minimax_value
            explore = self.C * math.sqrt(math.log(parent.visits + 1) / effective_n)
            ucb = exploit + explore if maximize else exploit - explore
            if (maximize and ucb > best_ucb) or (not maximize and ucb < best_ucb):
                best_ucb = ucb
                best = child

        if best is not None:
            best._pending += 1
        return best

    def _expand_enemy(self, my_node: MCTSNode) -> MCTSNode | None:
        """为我方节点扩展敌方回应子节点。

        前几个用固定基线池（通用回应），后续用自适应回应——
        用和我方相同的算法但输入反映我方发射后的 garrison 变化，
        敌人自然优先攻击我方削弱的行星。
        """
        max_children = self._max_enemy_children(my_node)
        next_idx = len(my_node.children)
        if next_idx >= max_children:
            return None

        # 前 N 个用固定基线
        if next_idx < len(self._fixed_enemy_sets):
            enemy_acts = self._fixed_enemy_sets[next_idx]
        else:
            # 自适应回应：同一算法，看到我方削弱后的状态
            enemy_acts = self._generate_enemy_response(my_node.actions)

        if not enemy_acts:
            enemy_acts = []

        child = MCTSNode(actions=enemy_acts, parent=my_node, is_enemy=True)
        my_node.children[next_idx] = child
        return child

    def _max_enemy_children(self, my_node: MCTSNode) -> int:
        """动态敌方回应数量：高访问量节点扩展更多回应（渐进式）。"""
        base = min(len(self._fixed_enemy_sets) + 4, MAX_ENEMY_CHILDREN)
        if my_node.visits > 120:
            return min(base + 4, MAX_ENEMY_CHILDREN)
        elif my_node.visits > 60:
            return base
        elif my_node.visits > 20:
            return max(5, base - 3)
        else:
            return 4

    def _generate_enemy_response(self, my_actions: list) -> list:
        """用和我方完全相同的贪心分配算法生成敌方回应。

        关键在于：根据我方发射的舰队覆写敌方视角下的目标 needed，
        我方削弱的行星 needed 减少 → sufficient 属性自动变为 True →
        贪心分配器自然优先选取这些目标。不需要任何启发式加权——对称博弈。
        """
        if not self._enemy_action_pool:
            return []

        # 计算我方发射导致的 garrison 变化
        garrison_delta = {}
        for a in my_actions:
            if a.is_pass():
                continue
            garrison_delta[a.source_id] = garrison_delta.get(a.source_id, 0) - a.ships

        if not garrison_delta:
            sorted_actions = sorted(self._enemy_action_pool,
                                    key=lambda a: a._heuristic_score
                                    if hasattr(a, '_heuristic_score') else 0,
                                    reverse=True)
            return self._greedy_allocate(sorted_actions, dict(self._enemy_source_available))

        # 临时覆写 needed（sufficient 属性依赖 needed，贪心分配器检查 sufficient）
        saved_needed = {}
        for a in self._enemy_action_pool:
            delta = garrison_delta.get(a.target_id, 0)
            if delta != 0:
                saved_needed[a] = a.needed
                a.needed = max(1, a.needed + delta)  # delta 为负 → 更容易打
                a._heuristic_score = self._quick_action_score(a, set())

        sorted_actions = sorted(self._enemy_action_pool,
                                key=lambda a: a._heuristic_score, reverse=True)
        result = self._greedy_allocate(sorted_actions, dict(self._enemy_source_available))

        # 恢复原始 needed
        for a, orig in saved_needed.items():
            a.needed = orig

        return result

    # ── 时间线评估（核心：替代 step_state 物理仿真）────────────

    def _evaluate(self, my_actions: list, enemy_actions: list) -> float:
        """时间线评估：插入双方动作 → 重算归属 → 价值差。

        不再克隆 GameState、不再跑行星轨道/彗星移动/swept-pair 碰撞。
        只在基线时间线上插入本回合的舰队到达事件，局部重算受影响行星。
        """
        actions_by_player = {
            self.player: my_actions,
            self.enemy: enemy_actions,
        }
        tl = self.root_timeline.with_actions(actions_by_player)
        return tl.evaluate_diff(self.player, self.enemy)

    # ── 回溯 ──────────────────────────────────────────────────

    def _backpropagate(self, enemy_node: MCTSNode, my_node: MCTSNode, value: float):
        """二层回溯（清除 virtual loss）。"""
        enemy_node._pending = max(0, enemy_node._pending - 1)
        enemy_node.visits += 1
        enemy_node.total_value += value
        my_node._pending = max(0, my_node._pending - 1)
        my_node.visits += 1
        self.root.visits += 1

    # ── 最优动作选择 ──────────────────────────────────────────

    def _best_actions(self, baseline_diff: float = -float("inf")) -> list:
        """选 maximin 最优动作集。若不出牌更好则 pass。"""
        if not self.root or not self.root.children:
            return []

        best_child = None
        best_value = -float("inf")
        for child in self.root.children.values():
            if child.visits > 0 and child.minimax_value > best_value:
                best_value = child.minimax_value
                best_child = child

        # 不出牌比所有出牌方案都好 → pass
        if baseline_diff > best_value:
            return []

        if best_child is None:
            for child in self.root.children.values():
                if child.actions:
                    return [[a.source_id, a.angle, a.ships] for a in child.actions]
            return []

        return [[a.source_id, a.angle, a.ships] for a in best_child.actions]

    # ── 工具 ──────────────────────────────────────────────────

    def _time_up(self, start_time: float) -> bool:
        return (time.monotonic() - start_time) * 1000 > self.time_budget_ms - 50
