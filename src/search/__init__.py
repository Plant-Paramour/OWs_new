"""动作搜索模块 —— 评估并最大化行星占领价值。

核心组件:
- LaunchOutcome: what-if 模拟结果
- simulate_fleet_launch(): 注入候选舰队 → 重新模拟时间线
- find_min_ships_to_capture(): 二分搜索最小攻占舰船数
- value_of_capture(): 占领价值公式
- compute_action_value(): 一站式价值计算
- ScoredAction: 评分后的候选动作
- search_best_actions(): 枚举搜索最佳动作
- beam_search(): 浅层束搜索（多步前瞻）
"""

from .simulator import LaunchOutcome, simulate_fleet_launch, find_min_ships_to_capture
from .valuation import value_of_capture, compute_action_value, is_comet_target
from .search import ScoredAction, search_best_actions, beam_search

__all__ = [
    "LaunchOutcome",
    "simulate_fleet_launch",
    "find_min_ships_to_capture",
    "value_of_capture",
    "compute_action_value",
    "is_comet_target",
    "ScoredAction",
    "search_best_actions",
    "beam_search",
]
