"""距离可靠性测试：测试 aim_at() 在不同距离下的命中率。

生成多局随机游戏，对所有 (source, target) 对调用 aim_at()，
然后用 validate_fleet_arrival() 验证，统计各距离区间的命中率。
"""

import sys
import io
import random
import math
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 使用游戏引擎生成随机棋盘
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Planet, Fleet, generate_planets, generate_comet_paths,
    BOARD_SIZE, CENTER, SUN_RADIUS, ROTATION_RADIUS_LIMIT,
    COMET_RADIUS, COMET_PRODUCTION, COMET_SPAWN_STEPS,
)

from src.engine.interception import aim_at
from src.engine.validation import validate_fleet_arrival
from src.engine.prediction import comet_remaining_life
from src.engine.physics import dist
from src.world.types import GameState


def create_game_state(planets, angular_velocity, comets, comet_ids, step=0):
    """从原始行星列表构建 GameState。"""
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet as RawPlanet

    planet_objs = []
    for p in planets:
        pid, owner, x, y, r, ships, prod = (
            p[0], p[1], p[2], p[3], p[4], p[5], p[6]
        )
        planet_objs.append(RawPlanet(pid, owner, x, y, r, int(ships), prod))

    initial_by_id = {}
    for p in planet_objs:
        init_x = p.x
        init_y = p.y
        initial_by_id[p.id] = RawPlanet(p.id, p.owner, init_x, init_y, p.radius, p.ships, p.production)

    my_planets = [p for p in planet_objs if p.owner == 0]
    enemy_planets = [p for p in planet_objs if p.owner == 1]
    neutral_planets = [p for p in planet_objs if p.owner == -1]

    # 计算 comet_ids 列表
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet as RawPlanet

    remaining = 500 - step
    episode_steps = 500

    my_total_ships = sum(p.ships for p in my_planets)
    enemy_total_ships = sum(p.ships for p in enemy_planets)
    my_total_production = sum(p.production for p in my_planets)
    enemy_total_production = sum(p.production for p in enemy_planets)

    return GameState(
        step=step,
        player=0,
        planets=planet_objs,
        fleets=[],
        angular_velocity=angular_velocity,
        initial_by_id=initial_by_id,
        comets=list(comets),
        comet_ids=set(comet_ids),
        my_planets=my_planets,
        enemy_planets=enemy_planets,
        neutral_planets=neutral_planets,
        remaining_steps=remaining,
        episode_steps=episode_steps,
        num_players=2,
        my_total_ships=my_total_ships,
        enemy_total_ships=enemy_total_ships,
        my_total_production=my_total_production,
        enemy_total_production=enemy_total_production,
    )


def test_distance_reliability(num_games=20, ships_to_send=20):
    """测试 num_games 局游戏中 aim_at 的距离-命中率关系。"""
    # 距离区间统计：[(min_dist, max_dist, label)]
    buckets = [
        (0, 20, "0-20"),
        (20, 30, "20-30"),
        (30, 40, "30-40"),
        (40, 50, "40-50"),
        (50, 60, "50-60"),
        (60, 70, "60-70"),
        (70, 80, "70-80"),
        (80, 90, "80-90"),
        (90, 200, "90+"),
    ]
    stats = {label: {"total": 0, "aim_ok": 0, "valid": 0, "fail_reasons": defaultdict(int)} for _, _, label in buckets}

    # 调试：记录前几个失败样例
    failure_samples = []

    for game_idx in range(num_games):
        rng = random.Random(game_idx * 1000 + 42)
        planets_raw = generate_planets(rng)
        angular_velocity = rng.uniform(0.025, 0.05)

        # 分配玩家
        num_groups = len(planets_raw) // 4
        if num_groups > 0:
            home_group = rng.randint(0, num_groups - 1)
            base = home_group * 4
            planets_raw[base][1] = 0
            planets_raw[base][5] = 10
            planets_raw[base + 3][1] = 1
            planets_raw[base + 3][5] = 10

        # 尝试生成彗星
        comets = []
        comet_ids = []
        for spawn_step in [50]:
            comet_paths = generate_comet_paths(
                planets_raw, angular_velocity, spawn_step,
                comet_planet_ids=comet_ids, comet_speed=4.0, rng=rng
            )
            if comet_paths:
                next_id = max(p[0] for p in planets_raw) + 1
                comet_ships = min(rng.randint(1, 99), rng.randint(1, 99))
                group = {"planet_ids": [], "paths": comet_paths, "path_index": 0}
                for i, p_path in enumerate(comet_paths):
                    pid = next_id + i
                    group["planet_ids"].append(pid)
                    comet_ids.append(pid)
                    planet = [pid, -1, p_path[0][0], p_path[0][1], COMET_RADIUS, comet_ships, COMET_PRODUCTION]
                    planets_raw.append(planet)
                comets.append(group)

        state = create_game_state(planets_raw, angular_velocity, comets, comet_ids, step=60)

        my_planets = state.my_planets
        all_targets = [p for p in state.planets if p.owner != 0]

        for src in my_planets:
            for tgt in all_targets:
                if tgt.id == src.id:
                    continue
                d = dist(src.x, src.y, tgt.x, tgt.y)

                # 确定所属区间
                bucket_label = None
                for lo, hi, label in buckets:
                    if lo <= d < hi:
                        bucket_label = label
                        break
                if bucket_label is None:
                    continue

                stats[bucket_label]["total"] += 1

                # aim_at
                result = aim_at(
                    src, tgt, ships_to_send,
                    state.initial_by_id, state.angular_velocity,
                    state.comets, state.comet_ids,
                )
                if result is None:
                    continue  # 路径被阻挡，不算失败

                stats[bucket_label]["aim_ok"] += 1
                angle, eta, _, _ = result

                # 验证
                valid, _, reason = validate_fleet_arrival(src, tgt, ships_to_send, angle, state)
                if valid:
                    stats[bucket_label]["valid"] += 1
                else:
                    stats[bucket_label]["fail_reasons"][reason] += 1
                    if len(failure_samples) < 10:
                        failure_samples.append({
                            "game": game_idx, "src": src.id, "tgt": tgt.id,
                            "dist": d, "angle": angle, "eta": eta,
                            "reason": reason,
                            "tgt_radius": tgt.radius,
                            "tgt_is_comet": tgt.id in state.comet_ids,
                        })

        print(f"  Game {game_idx + 1}/{num_games} done", flush=True)

    # 输出报告
    print("\n" + "=" * 72)
    print("Distance Reliability Report")
    print("=" * 72)
    print(f"{'Range':>8}  {'Total':>6}  {'Aim OK':>7}  {'Valid':>6}  {'Hit%':>7}  {'Status'}")
    print("-" * 72)

    for _, _, label in buckets:
        s = stats[label]
        if s["total"] == 0:
            continue
        aim_pct = s["aim_ok"] / s["total"] * 100
        if s["aim_ok"] > 0:
            hit_pct = s["valid"] / s["aim_ok"] * 100
        else:
            hit_pct = 0.0

        if hit_pct >= 99.9:
            status = "RELIABLE"
        elif hit_pct >= 95.0:
            status = "CAUTION"
        else:
            status = "UNRELIABLE"

        print(f"{label:>8}  {s['total']:>6}  {s['aim_ok']:>7}  {s['valid']:>6}  {hit_pct:>6.1f}%  {status}")

    print("-" * 72)
    print("\n结论：hit% ≥ 99.9% 的最大距离 = 推荐 MAX_SAFE_DISTANCE 阈值")

    # 找出可靠距离
    max_reliable = 0
    for lo, hi, label in buckets:
        s = stats[label]
        if s["aim_ok"] > 0:
            hit_pct = s["valid"] / s["aim_ok"] * 100
            if hit_pct >= 99.9:
                max_reliable = hi
    print(f"建议 MAX_SAFE_DISTANCE = {max_reliable}")

    # 失败原因汇总
    print("\n--- 失败原因分布 ---")
    all_reasons = defaultdict(int)
    for _, _, label in buckets:
        for reason, count in stats[label]["fail_reasons"].items():
            all_reasons[reason] += count
    for reason, count in sorted(all_reasons.items(), key=lambda x: -x[1]):
        print(f"  {count:>5} × {reason}")

    # 失败样例
    print("\n--- 失败样例 (前10) ---")
    for fs in failure_samples:
        print(f"  Game {fs['game']}  src={fs['src']} tgt={fs['tgt']}  dist={fs['dist']:.1f}"
              f"  r={fs['tgt_radius']:.1f}  comet={fs['tgt_is_comet']}"
              f"  angle={fs['angle']:.3f}  eta={fs['eta']}"
              f"  → {fs['reason']}")


if __name__ == "__main__":
    print("Testing aim_at distance reliability...")
    print(f"{'='*72}")
    test_distance_reliability(num_games=20, ships_to_send=20)
