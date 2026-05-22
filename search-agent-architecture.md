# 搜索智能体 (Search Agent) 架构文档

## 概览

搜索智能体是 Orbit Wars 项目中**不依赖神经网络**的纯算法决策系统。它基于世界模型 + 物理引擎 + 价值公式，枚举所有可行的攻击/防御/合击动作，选取净价值最高的组合，通过贪心分配输出可执行的动作列表。

核心入口：`src/search/search.py` → `search_agent_act()`

---

## 代码文件清单（按层级）

### 第一层：物理引擎 `src/engine/`

纯函数模块，无副作用，无项目内部依赖。所有数学计算与 Kaggle 游戏引擎完全一致。

| 文件 | 导出 | 职责 |
|------|------|------|
| `constants.py` | 棋盘尺寸、太阳半径、速度上限、模拟视野等常量 | 全局配置 |
| `physics.py` | `fleet_speed()`, `travel_time()`, `estimate_arrival()`, `estimate_arrival_float()`, `dist()`, `segment_hits_sun()`, `actual_path_geometry()`, `safe_angle_and_distance()` | 对数速度曲线、ETA 估算、太阳碰撞检测、路径几何 |
| `prediction.py` | `predict_planet_position()`, `predict_comet_position()`, `predict_target_position()`, `predict_target_position_float()`, `comet_remaining_life()`, `is_static_planet()`, `target_can_move()` | 行星公转、彗星轨迹预测、浮点位置插值 |
| `interception.py` | `aim_at()`, `check_path_blocked()`, `search_safe_intercept()` | 迭代拦截求解器（50 次迭代收敛至浮点 ETA）、中途行星阻挡检测 |

**关键设计：**
- `aim_at()` 使用浮点 ETA + 线性插值位置预测，消除远距离目标的 ~1 单位瞄准偏差
- `check_path_blocked()` 两层检测：射线-圆快速过滤 → 逐回合精确确认（含轨道行星运动）

---

### 第二层：世界模型 `src/world/`

将原始观测转化为结构化状态、到达账本和未来时间线。

| 文件 | 导出 | 职责 |
|------|------|------|
| `types.py` | `GameState` dataclass | 统一状态容器：行星分类、舰船/产能统计、剩余回合 |
| `observation.py` | `parse_observation()` | 原始 obs (dict/attribute) → `GameState` |
| `fleet_tracker.py` | `build_arrival_ledger()`, `fleet_target_planet()` | 射线-圆命中判定 → 到达账本 `{planet_id: [(eta, owner, ships)]}` |
| `combat.py` | `simulate_planet_timeline()`, `resolve_arrival_event()`, `state_at_timeline()` | 同回合战斗结算、行星时间线模拟、`keep_needed` 二分搜索 |

**关键设计：**
- `simulate_planet_timeline()` 返回完整未来状态字典：`owner_at`, `ships_at`, `keep_needed`, `fall_turn`, `first_enemy`, `holds_full`
- `keep_needed` 通过二分搜索 `survives_with_keep(k)` 精确计算，用于防御决策
- 战斗结算完全匹配游戏引擎：前两名攻击者对消 → 幸存者 vs 驻军

---

### 第三层：搜索智能体 `src/search/`

核心决策模块。依赖物理引擎 + 世界模型，不依赖神经网络。

#### 3a. What-If 模拟器 `simulator.py`

| 导出 | 职责 |
|------|------|
| `LaunchOutcome` dataclass | 单次舰队发射的模拟结果：`aim`, `blocked`, `new_timeline`, `capture_turn`, `hold_until`, `source_at_risk` |
| `simulate_fleet_launch(src, tgt, ships, state, ledger, timelines)` | 注入候选舰队 → 重新模拟目标时间线 → 分析占领回合 |
| `simulate_multi_fleet_launch(sources, tgt, state, ledger, timelines)` | 多源合击模拟（2-3 个源行星联合进攻同一目标） |
| `find_min_ships_to_capture(src, tgt, max_ships, state, ledger, timelines)` | 二分搜索最小攻占舰船数（最低 8 艘） |

**调用链：**
```
simulate_fleet_launch()
  ├── aim_at()                    ← engine/interception.py
  ├── check_path_blocked()        ← engine/interception.py
  └── simulate_planet_timeline()  ← world/combat.py
```

#### 3b. 价值计算 `valuation.py`

| 导出 | 职责 |
|------|------|
| `value_of_capture(target, capture_turn, hold_until, remaining_steps, ships_sent, is_enemy, source_at_risk, neutral_contested)` | 占领价值公式：`productive_turns × prod × swing − cost` |
| `compute_action_value(outcome, target, player_id, remaining_steps, ships_sent, comet_ids, timelines)` | 一站式价值计算（含 contested 检测） |
| `is_comet_target(target, comet_ids)` | 彗星判断（彗星不使用占领公式） |
| `lookahead_adjustment(action, state, timelines)` | 多步前瞻调整（长局 >100 步启用） |
| `_enemy_counter_threat(target, action, state, timelines)` | 敌方反制威胁惩罚（上限 25%） |
| `_expansion_chain_bonus(target, action, state)` | 扩张链期权奖励（factor 0.06） |
| `value_of_reinforcement(src, tgt, ships, state, timelines, ledger)` | 防御增援价值：行星现值 × 风险覆盖 − 调动成本 |
| `value_of_evacuation(src, ships, nearest_safe_planet, state, timelines)` | 撤离价值：保存的舰船 − 运输时间成本 |

**价值公式核心：**

| 目标类型 | swing | 含义 |
|----------|-------|------|
| 中立 (safe) | 1.0 | 仅我方获得产能 |
| 中立 (contested) | 2.0 | 我方获得 + 剥夺对手 = 双倍收益 |
| 敌方 | 2.5 | 直接零和 2.0 + 剥夺复利溢价 0.5 |

contested 检测：检查原时间线中敌方是否会占领该中立星。

#### 3c. 搜索主算法 `search.py`

| 导出 | 职责 |
|------|------|
| `ScoredAction` dataclass | 评分后的候选动作 |
| `_try_ship_amounts(src, tgt, ship_amounts, state, ledger, timelines, delays)` | 对给定舰船数列表逐一模拟，返回最佳 |
| `_evaluate_candidate(src, tgt, available, state, ledger, timelines)` | 评估单个 (source, target) 对：二分搜索 → 尝试多种舰船规模 → 返回最佳 |
| `search_best_actions(state, ledger, timelines, top_k, include_comets, eta_progressive)` | **单源搜索**：枚举所有 (source, target) 对 + 防御增援 + 撤离 |
| `beam_search(state, ledger, timelines, beam_width, max_depth)` | 束搜索 + 前瞻调整 |
| `MultiSourceAction` dataclass | 多源合击动作 |
| `_find_multi_source_actions(state, ledger, timelines)` | 多源合击搜索：合并最近 2-3 个源行星联合进攻 |
| `search_agent_act(state, ledger, timelines)` | **纯搜索智能体入口**：完整决策管道 → `[(source_id, angle, ships), ...]` |

**`search_agent_act()` 全流程：**

```
1. 构建世界模型 (ledger + timelines)
2. 搜索最佳单源动作 (search_best_actions, top_k=60)
   ├── 攻击: 逐个评估 (source, target) 对
   │   ├── 二分搜索最小攻占舰船数
   │   ├── 尝试 [min, min×1.3, min×2, available]
   │   └── 计算占领价值
   ├── 防御增援: 识别 keep_needed > current_defense 的己方行星
   └── 撤离: 注定失守行星保存舰船
3. 搜索多源合击 (_find_multi_source_actions)
   ├── 单一源无法攻占 → 合并最近 2-3 个源行星联合进攻
   └── 计算联合占领价值
4. 前瞻调整 (长局 >100 步)
   ├── 敌方反制威胁惩罚
   └── 扩张链期权奖励
5. 贪心分配舰船 (解决同一源行星被多个动作争用)
   ├── 仅执行正价值动作 (value > 0)
   └── 按价值降序分配，舰船不足自动跳过
6. 瞄准 + 太阳阻挡检测 → 输出 [(source_id, angle, ships), ...]
```

**关键约束：**
- 仅执行正价值动作（`value > 0`），舰船不足时自动待机攒兵
- 评估与执行一致：只评估当前可用舰船数，不评估"等待 N 回合后的未来舰船"
- 同一源行星可分配多个动作（贪心分配）
- 渐进式 ETA 过滤：前期限制远距离目标（`max_eta = 40 + progress × 85`）

---

### 第四层：环境集成 `src/env/`

搜索智能体的结果通过 Logit Bias 机制集成到 PPO 训练中。

| 文件 | 导出 | 职责 |
|------|------|------|
| `wrapper.py` | `OrbitWarsEnv`, `_build_search_value_map()` | 环境封装：`collect_decisions()` 中可选 `search_alpha > 0` 时，运行搜索为策略提供 Logit Bias |

**搜索集成机制（Logit Bias）：**

```
augmented_logits = raw_target_logits + search_alpha × search_value

其中:
  search_value = (搜索占领价值 + 前瞻调整) / remaining_steps
  归一化到 [−1, 10] 范围，α 偏置约 0.1~1.5
```

`_build_search_value_map()` 返回 `{(source_id, target_id): normalized_value}` 映射。

---

### 验证与诊断脚本

| 文件 | 说明 |
|------|------|
| `scripts/verify_search.py` | 44 个单元测试：模拟器、价值计算、搜索算法、多舰队场景、集成测试 |
| `scripts/verify_search_vs_sniper.py` | 搜索智能体 vs 对手 (Sniper/Heuristic/V4Hybrid) 的独立评估 |
| `scripts/debug_search_decisions.py` | 单局逐步诊断：打印每步搜索决策详情、Top-10 动作、合击候选 |

---

## 完整依赖图

```
┌─────────────────────────────────────────────────────────────────┐
│                    search_agent_act()                            │
│                    src/search/search.py                          │
│  入口：纯搜索智能体，不依赖神经网络                                │
└──────────┬──────────┬──────────┬──────────┬─────────────────────┘
           │          │          │          │
    ┌──────▼──┐ ┌────▼───┐ ┌───▼────┐ ┌───▼──────┐
    │simulator│ │valuation│ │ search │ │interception│
    │  .py    │ │  .py    │ │  .py   │ │   .py      │
    │what-if  │ │价值公式 │ │搜索算法│ │aim_at()    │
    │模拟器   │ │前瞻调整 │ │束搜索  │ │check_block │
    └────┬────┘ └───┬─────┘ └───┬────┘ └─────┬──────┘
         │          │           │             │
    ┌────▼──────────▼───────────▼─────────────▼──────┐
    │              世界模型 src/world/                │
    │  ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
    │  │  types   │ │ fleet_   │ │    combat     │  │
    │  │GameState │ │ tracker  │ │timeline+战斗  │  │
    │  └──────────┘ └──────────┘ └───────────────┘  │
    │  ┌──────────────────────────────────────────┐  │
    │  │         observation (obs → state)         │  │
    │  └──────────────────────────────────────────┘  │
    └──────────────────────┬─────────────────────────┘
                           │
    ┌──────────────────────▼─────────────────────────┐
    │              物理引擎 src/engine/               │
    │  ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
    │  │constants │ │ physics  │ │  prediction   │  │
    │  │ .py      │ │速度/ETA  │ │  位置预测     │  │
    │  └──────────┘ └──────────┘ └───────────────┘  │
    │  ┌──────────────────────────────────────────┐  │
    │  │      interception (拦截+阻挡)            │  │
    │  └──────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────┘
```

---

## 数据流

```
原始 obs
  │
  ▼
parse_observation()  ──→  GameState
  │                        │
  ▼                        ├── my_planets / enemy_planets / neutral_planets
build_arrival_ledger()     ├── remaining_steps / episode_steps
  │                        ├── initial_by_id / angular_velocity
  ▼                        └── comets / comet_ids
到达账本 {pid: [(eta, owner, ships)]}
  │
  ▼
simulate_planet_timeline() × N
  │
  ▼
行星时间线 {pid: {owner_at, ships_at, keep_needed, fall_turn, ...}}
  │
  ▼
┌──────────────────────────────────────────┐
│          search_agent_act()              │
│                                          │
│  1. search_best_actions()               │
│     ├── 对每个 (src, tgt):              │
│     │   ├── find_min_ships_to_capture() │
│     │   ├── simulate_fleet_launch()     │
│     │   └── compute_action_value()      │
│     ├── 防御增援: value_of_reinforcement│
│     └── 撤离: value_of_evacuation       │
│                                          │
│  2. _find_multi_source_actions()        │
│     └── simulate_multi_fleet_launch()   │
│                                          │
│  3. lookahead_adjustment() (长局)       │
│     ├── _enemy_counter_threat()         │
│     └── _expansion_chain_bonus()        │
│                                          │
│  4. 贪心分配 → aim_at() × N            │
│     └── check_path_blocked() × N        │
│                                          │
│  5. 输出: [(source_id, angle, ships)]   │
└──────────────────────────────────────────┘
```

---

## 关键常数

| 常量 | 值 | 位置 | 说明 |
|------|-----|------|------|
| `SEARCH_MIN_SHIPS` | 5 | `search.py:22` | 最小搜索舰船数 |
| `SIM_HORIZON` | 110 | `engine/constants.py:26` | 时间线最大前瞻回合 |
| `INTERCEPT_TOLERANCE` | 1 | `engine/constants.py:28` | 拦截收敛容差 |
| `MAX_SPEED` | 6.0 | `engine/constants.py:16` | 舰队速度上限 |
| `SUN_RADIUS` | 10.0 | `engine/constants.py:12` | 太阳半径 |
| `SUN_SAFETY` | 1.5 | `engine/constants.py:13` | 太阳安全余量 |
| `aim_at` 最大迭代 | 50 | `interception.py:103` | 拦截收敛最大迭代次数 |
| `lookahead` 启用阈值 | >100 步剩余 | `search.py:537` | 前瞻调整仅长局启用 |

---

## 评估成绩

| 对手 | 胜率 | 平均 Φ | 说明 |
|------|------|--------|------|
| Sniper | 93.3% | +9.50 | 50 回合局 |
| Heuristic | 60.0% | +2.65 | 50 回合局 |
| V4Hybrid | 10.0% | -6.36 | 50 回合局 |

---

## 已知局限

1. **滚雪球效应未捕获**：线性价值公式无法体现早期扩张的复利价值（占领 A → 用 A 的产能占领 B）
2. **缺少进攻性反应**：敌方发兵后老家空虚，搜索智能体应能检测并偷袭
3. **终局奖励被 soft_clip 压缩**：wrapper.py 对大额奖励有压缩，终局 ±10 被压到 ~6.5-8
4. **全行星无可用舰船时 reward 丢失**：极端罕见，所有己方行星 available==0 时整步无 transition
