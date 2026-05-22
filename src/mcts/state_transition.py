"""MCTS 状态转移引擎 —— 严格复制游戏引擎的回合逻辑。

使用纯函数操作不可变的 Planet/Fleet 命名元组，
模拟结果必须与 Kaggle orbit_wars 引擎逐回合一致。
"""

import math
import copy
from collections import defaultdict

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from ..engine.constants import (
    CENTER_X, CENTER_Y, SUN_RADIUS, ROTATION_LIMIT,
    LAUNCH_CLEARANCE, SIM_HORIZON,
)
from ..engine.physics import fleet_speed, segment_hits_sun
from ..engine.prediction import predict_target_position
from ..world.types import GameState
from ..world.combat import resolve_arrival_event


def clone_state(state: GameState) -> GameState:
    """克隆 GameState，深拷贝可变字段。

    Planet/Fleet 是命名元组（不可变），浅拷列表即可。
    comets 列表中的 dict 含可变的 path_index，需深拷。
    """
    planets = list(state.planets)
    fleets = list(state.fleets)
    comets = copy.deepcopy(state.comets)

    my_planets = [p for p in planets if p.owner == state.player]
    enemy_planets = [p for p in planets if p.owner not in (-1, state.player)]
    neutral_planets = [p for p in planets if p.owner == -1]

    player = state.player
    my_total_ships = sum(p.ships for p in my_planets) + sum(f.ships for f in fleets if f.owner == player)
    enemy_total_ships = sum(p.ships for p in enemy_planets) + sum(f.ships for f in fleets if f.owner not in (-1, player))
    my_total_production = sum(p.production for p in my_planets)
    enemy_total_production = sum(p.production for p in enemy_planets)

    return GameState(
        step=state.step,
        player=state.player,
        planets=planets,
        fleets=fleets,
        angular_velocity=state.angular_velocity,
        initial_by_id=state.initial_by_id,
        comets=comets,
        comet_ids=state.comet_ids,
        my_planets=my_planets,
        enemy_planets=enemy_planets,
        neutral_planets=neutral_planets,
        remaining_steps=state.remaining_steps,
        episode_steps=state.episode_steps,
        num_players=state.num_players,
        my_total_ships=my_total_ships,
        enemy_total_ships=enemy_total_ships,
        my_total_production=my_total_production,
        enemy_total_production=enemy_total_production,
    )


def _planet_by_id(planets: list, pid: int):
    for i, p in enumerate(planets):
        if p.id == pid:
            return i, p
    return None, None


def _update_comet_positions(planets: list, comets: list, comet_ids: set):
    """更新彗星位置：path_index += 1，从 paths 读取新坐标。"""
    for group in comets:
        pids = group.get("planet_ids", [])
        paths = group.get("paths", [])
        path_index = group.get("path_index", 0)
        group["path_index"] = path_index + 1
        new_idx = path_index + 1

        for pi, pid in enumerate(pids):
            if pi >= len(paths):
                continue
            path = paths[pi]
            if new_idx < len(path):
                new_x, new_y = path[new_idx]
                for j, p in enumerate(planets):
                    if p.id == pid:
                        planets[j] = Planet(p.id, p.owner, new_x, new_y, p.radius, int(p.ships), p.production)
                        break


def step_state(state: GameState, actions: dict) -> GameState:
    """模拟一个完整游戏回合（所有玩家同时行动）。

    Args:
        state: 当前游戏状态
        actions: {player_id: [[source_id, angle, ships], ...]}

    Returns:
        全新的 GameState（不修改输入状态）

    严格按照游戏引擎 orbit_wars.py 的 interpreter 顺序：
    1. 舰队发射（所有玩家同时）
    2. 生产
    3. 行星/彗星位置更新
    4. 舰队移动 + swept-pair 碰撞检测
    5. 战斗结算
    6. 更新元数据
    """
    new_planets = list(state.planets)
    new_fleets = list(state.fleets)
    comets = copy.deepcopy(state.comets)
    player = state.player

    # 生成新舰队 ID
    max_fid = max((f.id for f in new_fleets), default=0)

    # ── 1. 舰队发射（所有玩家同时）──
    for pid, action_list in actions.items():
        if not action_list:
            continue
        for act in action_list:
            if len(act) < 3:
                continue
            src_id, angle, ships = act[0], float(act[1]), int(act[2])
            idx, src = _planet_by_id(new_planets, src_id)
            if src is None or ships <= 0 or ships > src.ships:
                continue
            if src.owner != pid:
                continue

            # 扣减源行星舰船
            new_planets[idx] = Planet(src.id, src.owner, src.x, src.y, src.radius, src.ships - ships, src.production)

            # 创建舰队（从行星边界发射）
            clearance = src.radius + LAUNCH_CLEARANCE
            lx = src.x + math.cos(angle) * clearance
            ly = src.y + math.sin(angle) * clearance
            max_fid += 1
            new_fleets.append(Fleet(max_fid, pid, lx, ly, angle, src.id, ships))

    # ── 2. 生产 ──
    for i, p in enumerate(new_planets):
        if p.owner != -1:
            new_planets[i] = Planet(p.id, p.owner, p.x, p.y, p.radius, p.ships + p.production, p.production)

    # ── 3. 行星/彗星位置更新 ──
    # 轨道行星旋转
    for i, p in enumerate(new_planets):
        init = state.initial_by_id.get(p.id)
        if init is None or p.id in state.comet_ids:
            continue
        r = math.hypot(init.x - CENTER_X, init.y - CENTER_Y)
        if r + init.radius >= ROTATION_LIMIT:
            continue
        cur_ang = math.atan2(p.y - CENTER_Y, p.x - CENTER_X)
        new_ang = cur_ang + state.angular_velocity
        new_planets[i] = Planet(p.id, p.owner, CENTER_X + r * math.cos(new_ang), CENTER_Y + r * math.sin(new_ang), p.radius, p.ships, p.production)

    # 彗星位置更新
    _update_comet_positions(new_planets, comets, state.comet_ids)

    # ── 4. 舰队移动 + swept-pair 碰撞检测 ──
    surviving_fleets = []
    arrivals_by_planet = defaultdict(list)  # {planet_id: [(owner, ships), ...]}

    for fleet in new_fleets:
        speed = fleet_speed(max(1, fleet.ships))
        fx = fleet.x
        fy = fleet.y
        nx = fx + math.cos(fleet.angle) * speed
        ny = fy + math.sin(fleet.angle) * speed

        # 检测撞太阳
        if segment_hits_sun(fx, fy, nx, ny):
            continue

        # 检测舰队到达行星（swept-pair / ray-circle 最近命中）
        hit_planet_idx = None
        hit_planet = None
        hit_dist = float("inf")

        dir_x = math.cos(fleet.angle)
        dir_y = math.sin(fleet.angle)

        for j, planet in enumerate(new_planets):
            dx = planet.x - fx
            dy = planet.y - fy
            proj = dx * dir_x + dy * dir_y
            if proj < 0:
                continue
            perp_sq = dx * dx + dy * dy - proj * proj
            r2 = planet.radius * planet.radius
            if perp_sq >= r2:
                continue
            hd = proj - math.sqrt(max(0.0, r2 - perp_sq))
            if hd < hit_dist and hd <= speed + 1.0:
                hit_dist = hd
                hit_planet = planet
                hit_planet_idx = j

        if hit_planet is not None:
            arrivals_by_planet[hit_planet.id].append((fleet.owner, fleet.ships))
        else:
            surviving_fleets.append(Fleet(fleet.id, fleet.owner, nx, ny, fleet.angle, fleet.from_planet_id, fleet.ships))

    # ── 5. 战斗结算 ──
    for planet_id, arrivals in arrivals_by_planet.items():
        idx, planet = _planet_by_id(new_planets, planet_id)
        if planet is None:
            continue
        # 转换为 resolve_arrival_event 期望的格式: [(eta, owner, ships), ...]
        formatted = [(1, owner, s) for owner, s in arrivals]
        new_owner, new_garrison = resolve_arrival_event(planet.owner, float(planet.ships), formatted)
        new_planets[idx] = Planet(planet.id, new_owner, planet.x, planet.y, planet.radius, int(new_garrison), planet.production)

    # ── 6. 更新元数据 ──
    new_step = state.step + 1
    remaining = max(1, state.episode_steps - new_step)

    my_planets = [p for p in new_planets if p.owner == player]
    enemy_planets = [p for p in new_planets if p.owner not in (-1, player)]
    neutral_planets = [p for p in new_planets if p.owner == -1]

    my_total_ships = sum(p.ships for p in my_planets) + sum(f.ships for f in surviving_fleets if f.owner == player)
    enemy_total_ships = sum(p.ships for p in enemy_planets) + sum(f.ships for f in surviving_fleets if f.owner not in (-1, player))
    my_total_production = sum(p.production for p in my_planets)
    enemy_total_production = sum(p.production for p in enemy_planets)

    return GameState(
        step=new_step,
        player=player,
        planets=new_planets,
        fleets=surviving_fleets,
        angular_velocity=state.angular_velocity,
        initial_by_id=state.initial_by_id,
        comets=comets,
        comet_ids=state.comet_ids,
        my_planets=my_planets,
        enemy_planets=enemy_planets,
        neutral_planets=neutral_planets,
        remaining_steps=remaining,
        episode_steps=state.episode_steps,
        num_players=state.num_players,
        my_total_ships=my_total_ships,
        enemy_total_ships=enemy_total_ships,
        my_total_production=my_total_production,
        enemy_total_production=enemy_total_production,
    )
