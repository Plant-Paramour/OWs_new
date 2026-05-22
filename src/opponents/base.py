"""对手基类 —— 定义对手协议。

所有对手都实现为可调用对象，接收 (observation, configuration)，返回 actions 列表。
兼容 Kaggle env.train() 的 agent 协议。
"""

from typing import Protocol, Union


class Opponent(Protocol):
    """可调用对手协议。

    实现 __call__(self, observation, configuration) -> list[list]。
    """
    def __call__(self, observation, configuration) -> list[list]:
        ...


# 对手类型：可以是内置名称字符串或可调用对象
OpponentLike = Union[str, Opponent]
