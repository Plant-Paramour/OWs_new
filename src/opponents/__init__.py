"""对手模块 —— 课程学习对手池。"""

from .base import Opponent, OpponentLike
from .sniper import SniperOpponent
from .heuristic import HeuristicOpponent
from .pool import OpponentPool
from .lb1200 import LB1200Opponent
from .v4_hybrid import V4HybridOpponent
from .search_opponent import SearchOpponent
from ..mcts.agent import MCTSOpponent


def _get_self_play_opponent():
    from .self_play import SelfPlayOpponent
    return SelfPlayOpponent


def get_self_play():
    return _get_self_play_opponent()


__all__ = [
    "Opponent",
    "OpponentLike",
    "SniperOpponent",
    "HeuristicOpponent",
    "OpponentPool",
    "LB1200Opponent",
    "V4HybridOpponent",
    "SearchOpponent",
    "MCTSOpponent",
    "get_self_play",
]
