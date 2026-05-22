"""对手池 —— 加权随机采样 + 课程权重调度。"""

import random
from typing import Optional

from .base import OpponentLike


class OpponentPool:
    """对手池，支持加权随机采样。"""

    def __init__(self):
        self._entries = []  # list of (weight, opponent, name)

    def add(self, opponent: OpponentLike, weight: float, name: str = ""):
        """添加一个对手。

        Args:
            opponent: 对手（字符串名或可调用对象）
            weight: 采样权重（≥0，0 表示暂不采样）
            name: 可读名称（用于日志）
        """
        self._entries.append((weight, opponent, name or str(opponent)))

    def sample(self, rng: Optional[random.Random] = None) -> OpponentLike:
        """加权随机采样一个对手。

        Args:
            rng: 可选的 random.Random 实例（用于可复现测试）

        Returns:
            采样到的对手
        """
        if not self._entries:
            raise RuntimeError("对手池为空")
        _rng = rng if rng is not None else random
        weights = [e[0] for e in self._entries]
        total = sum(weights)
        if total <= 0:
            raise RuntimeError("所有对手权重均为 0")
        r = _rng.random() * total
        cumulative = 0.0
        for w, opp, _ in self._entries:
            cumulative += w
            if r <= cumulative:
                return opp
        return self._entries[-1][1]

    def set_weight(self, name: str, weight: float):
        """按名称更新对手权重（用于课程调度）。"""
        for i, (_, opp, n) in enumerate(self._entries):
            if n == name:
                self._entries[i] = (weight, opp, n)
                return
        raise KeyError(f"未找到对手: {name}")

    def get_weight(self, name: str) -> float:
        """查询对手权重。"""
        for w, _, n in self._entries:
            if n == name:
                return w
        raise KeyError(f"未找到对手: {name}")

    def names(self) -> list[str]:
        """返回所有对手名称列表。"""
        return [n for _, _, n in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        entries = ", ".join(f"{n}:{w:.1f}" for w, _, n in self._entries)
        return f"OpponentPool({entries})"
