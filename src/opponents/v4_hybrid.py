"""v4 Hybrid 对手 —— v4 规则引擎 + ML 验证器 + topk1，LB ~970。

来源: other's_work/train-submit-v4-ml-validator-topk2-tutorial.py
接口: agent(obs, config=None) → list[list]，兼容 Kaggle agent 协议。
依赖: other's_work/weights.npz（ML 验证器权重，缺失时自动降级为纯 v4+topk1）。
"""

import importlib.util
import os
import sys


_MODULE = None


def _load_module():
    """惰性加载 v4 hybrid 模块（仅首次调用时导入）。"""
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    filepath = os.path.join(project, "other's_work", "train-submit-v4-ml-validator-topk2-tutorial.py")

    spec = importlib.util.spec_from_file_location("v4_hybrid_agent", filepath)
    _MODULE = importlib.util.module_from_spec(spec)

    _MODULE.__builtins__ = __builtins__

    sys.modules["v4_hybrid_agent"] = _MODULE
    spec.loader.exec_module(_MODULE)

    has_ml = _MODULE._W is not None
    if has_ml:
        print(f"[v4_hybrid] 加载成功 (含 ML 验证器)")
    else:
        print(f"[v4_hybrid] 加载成功 (无 ML 验证器，仅 topk1)")
    return _MODULE


class V4HybridOpponent:
    """v4 + ML 验证器 + topk1 混合对手。

    LB ~970 (含 ML) / ~892 (纯 v4+topk1)。
    每回合输出 ≤1 个动作（topk1 节流）。
    """

    def __init__(self):
        self._module = None
        self._name = "v4_hybrid"

    @property
    def module(self):
        if self._module is None:
            self._module = _load_module()
        return self._module

    def __call__(self, observation, configuration):
        return self.module.agent(observation, configuration)

    def __repr__(self):
        return "V4HybridOpponent()"
