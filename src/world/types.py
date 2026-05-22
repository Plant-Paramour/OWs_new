"""Orbit Wars 世界模型 —— 数据类型定义。"""

from dataclasses import dataclass, field
from collections import namedtuple

# 复用 kaggle_environments 的命名元组
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet


@dataclass
class GameState:
    """解析后的游戏状态，包含所有观测信息。"""
    step: int
    player: int
    planets: list          # list[Planet]
    fleets: list           # list[Fleet]
    angular_velocity: float
    initial_by_id: dict    # {planet_id: Planet at step 0}
    comets: list           # raw comet group data
    comet_ids: set         # planet IDs that are comets

    # 预计算的分类列表
    my_planets: list = field(default_factory=list)
    enemy_planets: list = field(default_factory=list)
    neutral_planets: list = field(default_factory=list)

    # 派生统计数据
    remaining_steps: int = 0
    episode_steps: int = 500      # 1v1 默认 500，4p 为 200（来自 configuration.episodeSteps）
    num_players: int = 2
    my_total_ships: int = 0
    enemy_total_ships: int = 0
    my_total_production: int = 0
    enemy_total_production: int = 0
