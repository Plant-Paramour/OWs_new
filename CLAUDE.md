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
- `validation.py` — `validate_fleet_arrival()` 使用 swept-pair 碰撞检测前向模拟验证舰队路径（命中/出界/撞太阳/被拦截）；供动作枚举过滤无效发射

### Layer 2: World Model (`src/world/`) — raw observations → structured state
- `types.py` — `GameState` dataclass with precomputed planet classifications and ship/production stats
- `observation.py` — `parse_observation()` converts raw obs (dict or attribute-based) to `GameState`
- `fleet_tracker.py` — `build_arrival_ledger_accurate()` 使用 swept-pair 前向模拟精确计算 ETA（与游戏引擎一致）；`build_arrival_ledger()` 为旧静态射线-圆方法（向后兼容）
- `combat.py` — `simulate_planet_timeline()` runs full future-state simulation with combat resolution matching the game engine (top-2 attackers cancel → survivor vs garrison)

### Layer 3: Search Agent (`src/search/`) — core decision module
- `simulator.py` — what-if simulation: `simulate_fleet_launch()`, `simulate_multi_fleet_launch()`, `find_min_ships_to_capture()` (binary search, min 8 ships)
- `valuation.py` — value formulas: `value_of_capture()` (swing factors: neutral=1.0, contested=2.0, enemy=2.5), `value_of_reinforcement()`, `value_of_evacuation()`, `lookahead_adjustment()` (enemy counter-threat penalty + expansion chain bonus)
- `search.py` — main algorithm: `search_best_actions()` enumerates all (source, target) pairs with progressive ETA filtering, `beam_search()` with lookahead, `_find_multi_source_actions()` for joint strikes, `search_agent_act()` as the full decision pipeline

### MCTS 数据流
```
raw obs → parse_observation() → GameState
         → enumerate_actions() → validate_fleet_arrival() × N → Action[]
         → MCTSSearch.search()
           ├── generate_action_sets() → 我方动作集池 (~50)
           ├── generate_action_sets(enemy) → 敌方动作集池 (~30)
           ├── for each iteration:
           │     ├── select (我方 max UCB → 敌方 min UCB)
           │     ├── expand (新敌方回应)
           │     ├── evaluate: clone → step_state → build_baseline_timelines
           │     │     └── build_arrival_ledger_accurate → swept-pair ETA
           │     └── backprop (敌方节点累加值, 我方节点 min over 敌方)
           └── best: maximin over 我方动作集
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
| `state_transition.py` | `clone_state()`, `step_state()` | 严格复制游戏引擎回合推进 |
| `action_space.py` | `Action` dataclass, `enumerate_actions()` | 动作枚举，船数基于 needed；`validate_fleet_arrival` 过滤无效发射 |
| `value.py` | `build_baseline_timelines()`, `player_value_from_timelines()` | 时间线投影价值评估（swept-pair 精确 ETA），终局舰船全额计入 |
| `search.py` | `MCTSNode`, `MCTSSearch` | 二层 minimax 博弈树搜索，~400 次迭代/900ms |
| `agent.py` | `mcts_agent()`, `MCTSOpponent` | 标准 agent 协议入口，try/except 容错 |

**关键设计决策：**
- **二层博弈树**：Level 0 (root, 我方回合) → 我方动作集 → Level 1 (敌方回合) → 敌方回应 → 时间线评估
- **Minimax UCB**：我方节点选 max(minimax_value + explore)，敌方节点选 min(avg_value - explore)；最优动作选 maximin
- **敌方回应建模**：预生成 ~30 敌方动作集，每次模拟随机采样新回应，最多 12 个/my_node，取最坏情况（min）
- **精确 ETA**：`build_arrival_ledger_accurate()` 使用 swept-pair 前向模拟（与游戏引擎一致），正确计入行星轨道和彗星运动；旧静态射线-圆方法 ETA 偏差 1-2 回合已修复
- **时间线评估**：`player_value_from_timelines()` 逐回合追踪行星归属 + 在途舰队到达 + 战斗结算（top-2 攻击者对消 → 幸存者 vs 驻军），终局舰船全额计入（直接对应胜负）
- **无 rollout**：时间线投影已覆盖未来 110 回合在途舰队影响，不需要额外启发式步进（避免噪声 + 提升性能）
- **船数随机采样**：15 个随机样本覆盖 [0.9×, 3.0×] × needed，探索速度 vs 成本最优权衡
- **动作集基线**每个行星选最优动作（优先 sufficient + 近距离），始终包含"全 pass"候选

**与 `src/search/` 的关键差异：**
- MCTS 使用 minimax 二层博弈树搜索，建模敌方最优回应（非单一启发式）
- 时间线投影用 swept-pair 精确 ETA，而非静态射线-圆近似
- 终局舰船全额估值（1.0），零和从底层自然涌现
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
