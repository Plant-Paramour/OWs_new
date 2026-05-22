"""SelfPlay 对手 —— 包装我方策略网络供自对弈训练使用。

SelfPlay 对手遵循 Kaggle agent 协议 (observation, configuration) → actions。
内部使用项目特征工程 + 策略网络进行决策。
"""

import numpy as np
import torch

from ..world.observation import parse_observation
from ..features.builder import build_decision_matrix
from ..engine.interception import aim_at
from ..policy.action_head import compute_ships_to_send
from ..world.types import GameState


def _find_planet(state, planet_id: int):
    for p in state.planets:
        if p.id == planet_id:
            return p
    return None


class SelfPlayOpponent:
    """自对弈对手 —— 使用历史 checkpoint 的策略网络。

    定期从 Trainer 同步权重以模拟过往版本的自己。
    """

    def __init__(self, policy, candidate_count: int = 20, episode_steps: int = 500,
                 player_id: int = 0, deterministic: bool = False):
        """
        Args:
            policy: PolicyNetwork 实例
            candidate_count: 候选行星数
            episode_steps: 最大步数
            player_id: 对局中的玩家 ID（由 Kaggle 环境在运行时覆盖）
            deterministic: 是否使用确定性动作（True=argmax/mean, False=随机采样）
        """
        self.policy = policy
        self.candidate_count = candidate_count
        self.episode_steps = episode_steps
        self.player_id = player_id
        self.deterministic = deterministic

    def sync_weights(self, policy):
        """从当前训练中的策略同步权重。"""
        self.policy.load_state_dict(policy.state_dict())

    def __call__(self, observation, configuration) -> list[list]:
        """Kaggle agent 协议。

        Args:
            observation: Kaggle 原始观测 (dict 或 object)
            configuration: Kaggle 配置 dict

        Returns:
            actions: list of [source_id, angle, ships]
        """
        state = parse_observation(observation, episode_steps=self.episode_steps)
        rows = build_decision_matrix(state, candidate_count=self.candidate_count)
        if not rows:
            return []

        source_groups = {}
        for row in rows:
            source_groups.setdefault(row.source_id, []).append(row)

        actions = []
        for source_id, srows in source_groups.items():
            self_t = torch.tensor(srows[0].self_feat, dtype=torch.float32)
            cand_t = torch.tensor(
                np.stack([r.cand_feat for r in srows]), dtype=torch.float32
            )
            global_t = torch.tensor(srows[0].global_feat, dtype=torch.float32)
            mask_t = torch.tensor([r.mask for r in srows])

            available = srows[0].action_info.get("available", 0)

            with torch.no_grad():
                out = self.policy.act(self_t, cand_t, global_t, mask=mask_t,
                                      deterministic=self.deterministic)

            target_idx = out["target_idx"].item()
            if target_idx >= len(srows):
                continue
            target_row = srows[target_idx]

            if target_row.candidate_id == -1:
                continue

            ships_to_send = compute_ships_to_send(available, out["ship_ratio"].item())
            if ships_to_send == 0:
                continue

            src_planet = _find_planet(state, source_id)
            tgt_planet = _find_planet(state, target_row.candidate_id)
            if src_planet is None or tgt_planet is None:
                continue

            aim = aim_at(
                src_planet, tgt_planet, ships_to_send,
                state.initial_by_id, state.angular_velocity,
                state.comets, state.comet_ids,
            )
            if aim is None:
                continue

            angle = aim[0]
            actions.append([source_id, angle, ships_to_send])

        return actions
