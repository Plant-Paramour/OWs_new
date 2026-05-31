"""h3b1 启发式对手 —— 手写规则管线智能体。

来源: other's_work/h3b1.py
接口: agent(obs, config=None) -> list[list]，兼容 Kaggle agent 协议。
"""

import importlib.util
import os
import sys


_MODULE = None


def _load_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    filepath = os.path.join(project, "other's_work", "h3b1.py")

    spec = importlib.util.spec_from_file_location("h3b1_agent", filepath)
    _MODULE = importlib.util.module_from_spec(spec)
    _MODULE.__builtins__ = __builtins__
    sys.modules["h3b1_agent"] = _MODULE
    spec.loader.exec_module(_MODULE)
    return _MODULE


class H3b1Opponent:
    """h3b1 启发式管线对手。

    包装 other's_work/h3b1.py 中的 agent 函数，提供标准 (observation, configuration) -> actions 接口。
    """

    def __init__(self):
        self._module = None
        self._name = "h3b1"

    @property
    def module(self):
        if self._module is None:
            self._module = _load_module()
        return self._module

    def __call__(self, observation, configuration):
        return self.module.agent(observation, configuration)

    def __repr__(self):
        return "H3b1Opponent()"
