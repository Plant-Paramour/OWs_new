# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Python**: `C:\ProgramData\anaconda3\envs\Orbit_Wars\python.exe` (conda environment)
- **Dependency**: `kaggle_environments` (provides `Planet`, `Fleet` types from `kaggle_environments.envs.orbit_wars.orbit_wars`)
- **Encoding**: Any script printing Chinese must include `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')` at the top
- **File paths**: Use full Windows absolute paths with backslashes for all file operations

## Project Overview

Orbit Wars search-based AI agent — a pure algorithmic decision system with **no neural network dependency**. It uses a physics engine + world model + value formula to enumerate feasible attack/defense/joint-strike actions, selecting the highest net-value combinations via greedy allocation.

Core entry point: `src/search/search.py` → `search_agent_act()`

## Architecture (3-layer stack)

### Layer 1: Physics Engine (`src/engine/`) — pure functions, no side effects
- `constants.py` — board size, sun radius, speed caps, simulation horizon
- `physics.py` — logarithmic speed curve (`fleet_speed()`), ETA estimation (`travel_time()`, `estimate_arrival_float()`), sun collision detection, path geometry
- `prediction.py` — planet orbit prediction, comet trajectory prediction, float-position interpolation for precision aiming
- `interception.py` — iterative intercept solver (`aim_at()`, max 50 iterations converging to floating-point ETA), path-blocking detection (`check_path_blocked()` with two-layer: ray-circle fast filter → per-turn confirmation)

### Layer 2: World Model (`src/world/`) — raw observations → structured state
- `types.py` — `GameState` dataclass with precomputed planet classifications and ship/production stats
- `observation.py` — `parse_observation()` converts raw obs (dict or attribute-based) to `GameState`
- `fleet_tracker.py` — `build_arrival_ledger()` uses ray-circle hit detection to map `{planet_id: [(eta, owner, ships)]}`
- `combat.py` — `simulate_planet_timeline()` runs full future-state simulation with combat resolution matching the game engine (top-2 attackers cancel → survivor vs garrison)

### Layer 3: Search Agent (`src/search/`) — core decision module
- `simulator.py` — what-if simulation: `simulate_fleet_launch()`, `simulate_multi_fleet_launch()`, `find_min_ships_to_capture()` (binary search, min 8 ships)
- `valuation.py` — value formulas: `value_of_capture()` (swing factors: neutral=1.0, contested=2.0, enemy=2.5), `value_of_reinforcement()`, `value_of_evacuation()`, `lookahead_adjustment()` (enemy counter-threat penalty + expansion chain bonus)
- `search.py` — main algorithm: `search_best_actions()` enumerates all (source, target) pairs with progressive ETA filtering, `beam_search()` with lookahead, `_find_multi_source_actions()` for joint strikes, `search_agent_act()` as the full decision pipeline

### Data flow
```
raw obs → parse_observation() → GameState
         → build_arrival_ledger() → arrival ledger
         → simulate_planet_timeline() × N → planet timelines
         → search_agent_act()
           ├── search_best_actions() (single-source attack + defense + evacuation)
           ├── _find_multi_source_actions() (joint strikes)
           ├── lookahead_adjustment() (long games >100 steps only)
           └── greedy ship allocation → aim_at() × N → [(source_id, angle, ships)]
```

## Key Constants

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `MAX_SPEED` | 6.0 | `engine/constants.py:16` | Fleet speed cap |
| `SUN_RADIUS` | 10.0 | `engine/constants.py:12` | Sun collision radius |
| `SIM_HORIZON` | 110 | `engine/constants.py:26` | Max lookahead turns |
| `INTERCEPT_TOLERANCE` | 1 | `engine/constants.py:28` | Intercept convergence tolerance |
| `SEARCH_MIN_SHIPS` | 5 | `search/search.py:22` | Min ships to consider an action |
| `aim_at` max iterations | 50 | `engine/interception.py:103` | Intercept solver limit |
| Lookahead threshold | >100 steps | `search/search.py:537` | Only enable lookahead adjustments in long games |

## Important Design Decisions

- **Float ETA for precision**: `aim_at()` uses floating-point ETA + linear interpolation for position prediction, eliminating ~1 unit aiming bias on distant targets
- **Defense-aware**: Each friendly planet computes `keep_needed` via binary search — ships needed to survive incoming enemy fleets. This reserve is subtracted before considering offensive actions.
- **Value-driven economy**: Only positive-value actions (value > 0) are executed. Ships stay idle when no profitable action exists, automatically banking for future turns.
- **Progressive ETA filtering**: Early-game restricts distant targets (`max_eta = 40 + progress × 85`), gradually opening up the full board.
- **Greedy allocation**: Same source planet can feed multiple actions; allocated in descending value order until ships run out.

### MCTS Agent (`src/mcts/`)

Monte Carlo Tree Search agent——完整动作集合 UCB1 搜索。每个 MCTS 节点 = 一个完整的本回合动作集合（多行星同时行动），而非逐行星决策。

**已验证：5局全胜 heuristic 对手。**

| 文件 | 导出 | 用途 |
|------|------|------|
| `state_transition.py` | `clone_state()`, `step_state()` | 严格复制游戏引擎回合推进（6步） |
| `action_space.py` | `Action` dataclass, `enumerate_actions()` | 动作枚举，船数基于攻占所需兵力(needed)而非可用比例 |
| `value.py` | `build_baseline_timelines()`, `player_value_from_state()` | 时间线投影价值评估 + 快速静态评估 |
| `search.py` | `MCTSNode`, `MCTSSearch` | 完整动作集合 UCB1 搜索，~1000次迭代/900ms |
| `agent.py` | `mcts_agent()`, `MCTSOpponent` | 标准 agent 协议入口，try/except 容错 |

**关键设计决策：**
- **船数规模**基于 `needed`（攻占所需兵力）的倍数 [0.9×, 1.0×, 1.2×, 1.5×, 2.0×] + 全押兜底，不产生无用小船
- **动作集基线**每个行星选最优动作（优先 sufficient + 近距离），而非简单取列表第一个
- **候选策略池**始终包含"全 pass"（`[]`），让 MCTS 能评估"等待积累再进攻"
- **Rollout 启发式**只派送 >= needed 的兵力，不足则跳过（不乱射）
- **零和价值**从 `player_value_from_state()` 自然涌现，不依赖显式 swing factor

**与 `src/search/` 的关键差异：**
- MCTS 搜索未来**游戏状态**而非单个 (source, target, ships) 三元组
- 使用 UCB1 探索-利用权衡的树搜索
- 评估整局状态而非单次 what-if 结果

## 对战与回放

```bash
# MCTS vs heuristic
python scripts/gen_replays.py --agent mcts --opponent heuristic -n 5

# MCTS vs 现有搜索智能体
python scripts/gen_replays.py --agent mcts --opponent search -n 5

# MCTS vs lb1200 / v4_hybrid
python scripts/gen_replays.py --agent mcts --opponent lb1200 -n 5
python scripts/gen_replays.py --agent mcts --opponent v4_hybrid -n 5

# 查看回放：浏览器打开 replays/viewer.html
```

对战结果保存到 `replays/wins/` 和 `replays/losses/`，可通过 `replays/viewer.html` 可视化回放。
