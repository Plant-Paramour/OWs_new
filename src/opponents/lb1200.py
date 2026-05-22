"""LB1200 PPO 策略对手 —— 榜单 ~700 分的最强开源对手。

来源: other's_work/lb-1200-orbit-wars-ppo-strategy.ipynb → .py
接口: agent(obs, config=None) → list[list]，兼容 Kaggle agent 协议。
"""

import importlib.util
import os
import sys


_MODULE = None


def _load_module():
    """惰性加载 lb1200 模块（仅首次调用时导入）。"""
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    filepath = os.path.join(project, "other's_work", "lb-1200-orbit-wars-ppo-strategy.py")

    spec = importlib.util.spec_from_file_location("lb1200_agent", filepath)
    _MODULE = importlib.util.module_from_spec(spec)

    # 注入模块所需的全局变量（避免 NameError）
    _MODULE.__builtins__ = __builtins__

    sys.modules["lb1200_agent"] = _MODULE
    spec.loader.exec_module(_MODULE)
    return _MODULE


class LB1200Opponent:
    """LB1200 PPO 策略对手。

    包装 other's_work/ 中的 agent 函数，提供标准 (observation, configuration) → actions 接口。
    """

    def __init__(self):
        self._module = None
        self._name = "lb1200"

    @property
    def module(self):
        if self._module is None:
            self._module = _load_module()
        return self._module

    def __call__(self, observation, configuration):
        return self.module.agent(observation, configuration)

    def __repr__(self):
        return "LB1200Opponent()"
