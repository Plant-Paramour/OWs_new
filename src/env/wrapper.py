"""OrbitWarsEnv —— Kaggle 环境包装器。

封装 env.train() 接口，支持对手集成、策略决策收集、shaped reward 计算。
可选集成动作搜索评分为策略提供 Logit Bias。
"""

import numpy as np
import torch
from kaggle_environments import make

from ..world.observation import parse_observation
from ..features.builder import build_decision_matrix
from ..engine.interception import aim_at
from ..policy.action_head import compute_ships_to_send, sample_action
from ..opponents.base import OpponentLike
from .reward import compute_step_reward, soft_clip, state_potential


class OrbitWarsEnv:
    """Orbit Wars 训练环境包装器。

    封装 Kaggle env.train() 接口，提供：
      - reset() → 返回初始 state + raw obs
      - step()  → 收集策略决策 → 执行 → 计算 shaped reward

    对手参数支持:
      - str: 内置名 ("random")
      - callable: 实现 (observation, configuration) → actions 的任意可调用对象
      - OpponentPool: 每局开始前 sample() 一次
    """

    def __init__(self, opponent: OpponentLike = "random", candidate_count: int = 20,
                 player_id: int = 0, episode_steps: int = 500):
        self.opponent = opponent
        self.candidate_count = candidate_count
        self.player_id = player_id
        self.episode_steps = episode_steps

        self.env = None
        self.trainer = None
        self._current_opponent = None

    def reset(self):
        """重置环境并返回初始 (GameState, raw_obs)。

        每局重置时从对手池采样（如果 opponent 是 OpponentPool），
        确保训练在单局内对手一致、局间轮换。
        """
        if self.env is None:
            self.env = make("orbit_wars", debug=True,
                           configuration={"episodeSteps": self.episode_steps})

        opp = self.opponent
        if hasattr(opp, "sample"):
            opp = opp.sample()
        self._current_opponent = opp

        agents = [None] * 2
        agents[self.player_id] = None
        agents[1 - self.player_id] = opp
        self.trainer = self.env.train(agents)

        raw_obs = self.trainer.reset()
        state = parse_observation(raw_obs, episode_steps=self.episode_steps)
        return state, raw_obs

    def collect_decisions(self, policy, state, deterministic=False,
                          search_alpha=0.0):
        """对所有己方行星采样动作，返回 (game_actions, transitions)。

        可选 search_alpha > 0 时，运行动作搜索为策略提供 Logit Bias：
          augmented_logits = target_logits + search_alpha * search_value

        Args:
            policy: PolicyNetwork 实例
            state: 当前 GameState
            deterministic: True=argmax/mean, False=随机采样
            search_alpha: 搜索偏置强度 (0=不使用搜索, 建议 0.01~0.1)

        Returns:
            (decisions, transitions) where:
              decisions:   list of (source_id, target_id, ships, angle)
              transitions: list of dicts with (self_feat, cand_feat, global_feat,
                           mask, target_idx, ship_ratio, target_log_prob,
                           ship_log_prob, value)
        """
        rows = build_decision_matrix(state, candidate_count=self.candidate_count)
        if not rows:
            return [], []

        # ── 可选: 运行搜索，构建 (source, target) → search_value 映射 ──
        search_value_map = {}
        if search_alpha > 0:
            search_value_map = _build_search_value_map(state)

        # 按源行星分组
        source_groups = {}
        for row in rows:
            source_groups.setdefault(row.source_id, []).append(row)

        decisions = []
        transitions = []

        for source_id, srows in source_groups.items():
            self_t = torch.tensor(srows[0].self_feat, dtype=torch.float32)
            cand_t = torch.tensor(
                np.stack([r.cand_feat for r in srows]), dtype=torch.float32
            )
            global_t = torch.tensor(srows[0].global_feat, dtype=torch.float32)
            mask_t = torch.tensor([r.mask for r in srows])

            available = srows[0].action_info.get("available", 0)

            with torch.no_grad():
                out = policy.forward(self_t, cand_t, global_t, mask=mask_t)

                raw_target_logits = out["target_logits"].clone()

                # ── 搜索 Logit Bias: 只影响动作采样，不影响 log_prob ──
                target_logits = raw_target_logits.clone()
                if search_alpha > 0:
                    bias = torch.zeros_like(target_logits)
                    for i, row in enumerate(srows):
                        key = (source_id, row.candidate_id)
                        if key in search_value_map:
                            bias[i] = search_value_map[key]
                    target_logits = target_logits + search_alpha * bias

                # 从 augmented logits 采样动作
                idx, ratio, _, s_lp = sample_action(
                    target_logits, out["ship_alpha"], out["ship_beta"],
                    mask=mask_t, deterministic=deterministic,
                )

                # 从 raw logits 计算 log_prob，与 PPO 更新分布一致
                raw_masked = raw_target_logits.clone()
                if mask_t is not None:
                    raw_masked[~mask_t] = float('-inf')
                from torch.distributions import Categorical
                t_lp = Categorical(logits=raw_masked).log_prob(idx)

            target_idx = idx.item()
            if target_idx >= len(srows):
                continue
            target_row = srows[target_idx]

            transition = {
                "self_feat": self_t, "cand_feat": cand_t,
                "global_feat": global_t, "mask": mask_t,
                "target_idx": idx,
                "ship_ratio": ratio,
                "target_log_prob": t_lp,
                "ship_log_prob": s_lp,
                "value": out["value"],
            }

            # no-op：不派兵
            if target_row.candidate_id == -1:
                transitions.append(transition)
                continue

            ships_to_send = compute_ships_to_send(available, ratio.item())
            if ships_to_send == 0:
                transitions.append(transition)
                continue

            # 用实际舰船数重新计算角度
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
            decisions.append((source_id, target_row.candidate_id, ships_to_send, angle))
            transitions.append(transition)

        return decisions, transitions

    def step(self, decisions, prev_state, comet_warmup=0.0,
             early_terminal_scale=0.0):
        """执行一步游戏。

        Args:
            decisions: list of (source_id, target_id, ships, angle)
            prev_state: 执行前的 GameState
            comet_warmup: 彗星探索引导加分
            early_terminal_scale: Phase 0 终局 Φ 倍率 (0=禁用)

        Returns:
            (next_state, raw_obs, reward, done, info)
        """
        actions = [[s, angle, sh] for s, t, sh, angle in decisions]
        raw_obs, game_reward, done, info = self.trainer.step(actions)
        curr_state = parse_observation(raw_obs, episode_steps=self.episode_steps)
        reward = compute_step_reward(prev_state, curr_state, self.player_id,
                                     comet_warmup=comet_warmup)
        reward = float(soft_clip(reward, linear_range=5.0, soft_coef=0.3, hard_cap=8.0))

        # Phase 0 终局奖励：谁的前期发育更好？Φ 大者获胜
        if done and early_terminal_scale > 0:
            terminal_bonus = early_terminal_scale * state_potential(
                curr_state, self.player_id)
            terminal_bonus = float(np.clip(terminal_bonus, -15.0, 15.0))
            reward += terminal_bonus

        return curr_state, raw_obs, reward, done, info


def _build_search_value_map(state):
    """运行动作搜索，返回 {(source_id, target_id): search_value} 映射。"""
    from ..world.fleet_tracker import build_arrival_ledger
    from ..world.combat import simulate_planet_timeline
    from ..engine.prediction import comet_remaining_life
    from ..search import search_best_actions
    from ..search.valuation import lookahead_adjustment

    ledger = build_arrival_ledger(state.fleets, state.planets)
    timelines = {}
    for p in state.planets:
        life = comet_remaining_life(p.id, state.comets) if p.id in state.comet_ids else None
        timelines[p.id] = simulate_planet_timeline(
            p, ledger.get(p.id, []), state.player, state.remaining_steps,
            planet_life=life,
        )
    results = search_best_actions(state, ledger, timelines, top_k=100)

    # 对每个候选施加多步前瞻调整（敌方反制 + 扩张链）
    for r in results:
        r.value += lookahead_adjustment(r, state, timelines)

    # 调整后重新排序
    results.sort(key=lambda r: r.value, reverse=True)

    # 归一化到 logit 兼容量级: max ≈ swing×max_prod = 10
    norm = max(1.0, float(state.remaining_steps))
    return {(r.source_id, r.target_id): r.value / norm for r in results}


def _find_planet(state, planet_id: int):
    """在 GameState 中按 ID 查找行星。"""
    for p in state.planets:
        if p.id == planet_id:
            return p
    return None
