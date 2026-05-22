"""搜索智能体 —— 纯搜索算法决策，不依赖神经网络。

对每个观测运行动作搜索 (search_best_actions)，选择价值最高的动作执行。
可作为独立对手使用，也可用于验证搜索算法本身的强度。
"""

from ..world.observation import parse_observation
from ..world.fleet_tracker import build_arrival_ledger
from ..world.combat import simulate_planet_timeline
from ..engine.interception import aim_at
from ..search.search import search_best_actions


class SearchOpponent:
    """纯搜索算法智能体。

    每步运行 search_best_actions 枚举所有 (source, target) 组合，
    选择价值 > 0 的最佳动作执行（每个源行星最多一个动作）。
    """

    def __init__(self, top_k: int = 20, min_value: float = 0.0,
                 episode_steps: int = 500):
        self.top_k = top_k
        self.min_value = min_value
        self._episode_steps = episode_steps

    def __call__(self, observation, configuration):
        episode_steps = self._episode_steps
        if isinstance(configuration, dict):
            episode_steps = configuration.get("episodeSteps", episode_steps)

        state = parse_observation(observation, episode_steps=episode_steps)

        ledger = build_arrival_ledger(state.fleets, state.planets)
        timelines = {}
        for p in state.planets:
            timelines[p.id] = simulate_planet_timeline(
                p, ledger.get(p.id, []), state.player, state.remaining_steps,
            )

        results = search_best_actions(state, ledger, timelines, top_k=self.top_k)

        actions = []
        used_sources = set()

        for r in results:
            if r.source_id in used_sources:
                continue
            if r.value <= self.min_value:
                continue

            src = next((p for p in state.planets if p.id == r.source_id), None)
            tgt = next((p for p in state.planets if p.id == r.target_id), None)
            if src is None or tgt is None:
                continue

            aim = aim_at(src, tgt, r.ships,
                         state.initial_by_id, state.angular_velocity,
                         state.comets, state.comet_ids)
            if aim is None:
                continue

            angle = aim[0]
            actions.append([r.source_id, angle, r.ships])
            used_sources.add(r.source_id)

        return actions
