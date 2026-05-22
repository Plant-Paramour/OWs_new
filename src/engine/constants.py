"""Orbit Wars 游戏引擎常量。

所有值与游戏引擎完全一致，直接来自 Structured Baseline v11。
"""

# ── 棋盘 ──
BOARD_SIZE = 100.0
CENTER_X = 50.0
CENTER_Y = 50.0

# ── 太阳 ──
SUN_RADIUS = 10.0
SUN_SAFETY = 1.5          # 路径检查时额外增加的安全余量

# ── 舰队 ──
MAX_SPEED = 6.0            # 最大舰队速度 (units/turn)
LAUNCH_CLEARANCE = 0.1     # 舰队发射时超出行星半径的距离

# ── 行星 ──
ROTATION_LIMIT = 50.0      # orbital_radius + planet_radius < 此值 → 轨道行星

# ── 游戏 ──
TOTAL_STEPS = 500

# ── 模拟 ──
SIM_HORIZON = 110          # 舰队追踪/时间线的最大前瞻回合数
ROUTE_SEARCH_HORIZON = 60  # 安全路径扫描的最大回合数
INTERCEPT_TOLERANCE = 1    # 拦截收敛容差 (回合)
