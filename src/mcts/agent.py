"""MCTS 智能体主入口。

符合标准 agent 协议: agent(observation, configuration) -> [[source_id, angle, ships], ...]
"""

import traceback
from ..world.observation import parse_observation
from .search import MCTSSearch


def mcts_agent(observation, configuration) -> list:
    """MCTS 智能体主入口。"""
    try:
        episode_steps = 500
        act_timeout = 1.0

        if hasattr(configuration, "episodeSteps"):
            episode_steps = int(configuration.episodeSteps)
        elif isinstance(configuration, dict):
            episode_steps = int(configuration.get("episodeSteps", 500))

        if hasattr(configuration, "actTimeout"):
            act_timeout = float(configuration.actTimeout)
        elif isinstance(configuration, dict):
            act_timeout = float(configuration.get("actTimeout", 1.0))

        time_budget_ms = max(100, int(act_timeout * 900))

        state = parse_observation(observation, episode_steps=episode_steps)
        searcher = MCTSSearch(time_budget_ms=time_budget_ms)
        actions = searcher.search(state)

        return actions
    except Exception:
        traceback.print_exc()
        return []


class MCTSOpponent:
    """MCTS 智能体包装器，实现 __call__ 协议。

    用于 gen_replays.py 和对手注册：
        opponent = MCTSOpponent()
        actions = opponent(observation, configuration)
    """

    def __init__(self, time_budget_ms: int = 900, episode_steps: int = 500):
        self.time_budget_ms = time_budget_ms
        self.episode_steps = episode_steps

    def __call__(self, observation, configuration) -> list:
        return mcts_agent(observation, configuration)
