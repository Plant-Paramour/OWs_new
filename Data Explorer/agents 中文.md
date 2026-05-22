# Orbit Wars：入门指南

本指南将带您逐步构建智能体、在本地测试并提交到 Kaggle 上的 Orbit Wars 竞赛。

## 游戏概述

Orbit Wars 是一款在 100x100 棋盘上进行的实时策略游戏，棋盘中央有一颗太阳。玩家通过向行星派遣舰队来征服它们。

- **行星**每回合生产舰船（与半径成正比）
- **内层行星**绕中央太阳旋转；外层行星是静态的
- **舰队**以给定角度从源行星沿直线飞行
- **舰队速度**随舰队规模变化（1 艘舰船 = 1/回合，较大舰队最高可达 6/回合）
- **战斗**：到达的舰队舰船数从行星驻军中扣除。若驻军降至 0 以下，所有权翻转
- **太阳**：撞到太阳的舰队会被摧毁
- **彗星**：沿椭圆路径飞越棋盘的临时行星
- **胜利条件**：时间耗尽时拥有最高舰船数（行星 + 舰队），或成为最后存活的玩家

完整规则和配置默认值请参阅 [README.md](README.md)。

## 您的智能体

您的智能体是一个接收观测并返回行动列表的函数。

**观测字段：**
- `player` — 您的玩家 ID（0-3）
- `planets` — 列表，每项为 `[id, owner, x, y, radius, ships, production]`（owner 为 -1 表示中立）
- `fleets` — 列表，每项为 `[id, owner, x, y, angle, from_planet_id, ships]`
- `angular_velocity` — 内层行星的旋转速度（弧度/回合）

**行动格式：**
每个行动为 `[from_planet_id, angle_in_radians, num_ships]`。

**示例 — 最近行星狙击手：**

```python
import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

def agent(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    for mine in my_planets:
        # 找到最近的非我方行星
        nearest = min(targets, key=lambda t: math.hypot(mine.x - t.x, mine.y - t.y))

        # 派遣刚好足够的舰船来占领它
        ships_needed = nearest.ships + 1
        if mine.ships >= ships_needed:
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])

    return moves
```

## 本地测试

从 PyPI 安装环境（Orbit Wars 需要 1.28.0 或更高版本）：

```bash
pip install "kaggle-environments>=1.28.0"
```

从 Python 或 notebook 运行一局游戏：

```python
from kaggle_environments import make

env = make("orbit_wars", configuration={"seed": 42}, debug=True)
env.run(["main.py", "random"])

# 查看结果
final = env.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")

# 在 notebook 中渲染
env.render(mode="ipython", width=800, height=600)
```

## 设置 Kaggle CLI

安装 CLI：

```bash
pip install kaggle
```

您需要一个 Kaggle 账户——如果还没有，请前往 https://www.kaggle.com 注册。然后在 https://www.kaggle.com/settings/api 下载您的 API 凭据，点击 **"API"** 部分下的 **"Generate New Token"**。

**推荐方式：API token 文件。** 将 token 字符串保存到 `~/.kaggle/access_token`：

```bash
mkdir -p ~/.kaggle
# 将 Kaggle 设置界面中的 token 粘贴到此文件
nano ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

其他认证方式：
- **OAuth（浏览器流程）：** `kaggle auth login`
- **环境变量：** `export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx`

验证 CLI 已正确配置：

```bash
kaggle competitions list -s "orbit wars"
```

## 查找竞赛

```bash
kaggle competitions list -s "orbit wars"
kaggle competitions pages orbit-wars
kaggle competitions pages orbit-wars --content
```

## 接受竞赛规则

提交之前，您**必须**在 Kaggle 网站上接受规则。前往 `https://www.kaggle.com/competitions/orbit-wars` 并点击 **"Join Competition"**。

验证您已加入：

```bash
kaggle competitions list --group entered
```

## 下载竞赛数据

```bash
kaggle competitions download orbit-wars -p orbit-wars-data
```

## 提交您的智能体

您的提交必须在根目录包含一个带有 `agent` 函数的 `main.py`。

**单文件智能体：**

```bash
kaggle competitions submit orbit-wars -f main.py -m "Nearest planet sniper v1"
```

**多文件智能体** — 打包为 tar.gz，其中 `main.py` 在根目录：

```bash
tar -czf submission.tar.gz main.py helper.py model_weights.pkl
kaggle competitions submit orbit-wars -f submission.tar.gz -m "Multi-file agent v1"
```

**Notebook 提交：**

```bash
kaggle competitions submit orbit-wars -k YOUR_USERNAME/orbit-wars-agent -f submission.tar.gz -v 1 -m "Notebook agent v1"
```

## 监控您的提交

查看提交状态：

```bash
kaggle competitions submissions orbit-wars
```

记下输出中的提交 ID——查看对局时需要用到。

## 列出对局

当您的提交完成一些对局后：

```bash
kaggle competitions episodes <SUBMISSION_ID>
```

用于脚本的 CSV 输出：

```bash
kaggle competitions episodes <SUBMISSION_ID> -v
```

## 下载回放和日志

下载某场对局的回放 JSON（用于可视化或分析）：

```bash
kaggle competitions replay <EPISODE_ID>
kaggle competitions replay <EPISODE_ID> -p ./replays
```

下载智能体日志以调试您的智能体行为：

```bash
# 第一个智能体的日志（索引 0）
kaggle competitions logs <EPISODE_ID> 0

# 第二个智能体的日志（索引 1）
kaggle competitions logs <EPISODE_ID> 1 -p ./logs
```

## 查看排行榜

```bash
kaggle competitions leaderboard orbit-wars -s
```

## 典型工作流

```bash
# 本地测试
python -c "
from kaggle_environments import make
env = make('orbit_wars', debug=True)
env.run(['main.py', 'random'])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"

# 提交
kaggle competitions submit orbit-wars -f main.py -m "v1"

# 查看状态
kaggle competitions submissions orbit-wars

# 查看对局
kaggle competitions episodes <SUBMISSION_ID>

# 下载回放和日志
kaggle competitions replay <EPISODE_ID>
kaggle competitions logs <EPISODE_ID> 0

# 查看排行榜
kaggle competitions leaderboard orbit-wars -s
```
