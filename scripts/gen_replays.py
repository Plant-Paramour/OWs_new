"""生成对战回放并保存到 replays/ 目录。

支持两种模式:
  --agent policy  (默认): 策略网络 vs 对手, 使用 OrbitWarsEnv 包装器
  --agent search  : 搜索智能体 vs 对手, 使用 Kaggle env.run() 直接对战
"""
import io, json, os, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.opponents import (SniperOpponent, HeuristicOpponent, LB1200Opponent,
                            V4HybridOpponent, SearchOpponent, MCTSOpponent)
from src.world.observation import parse_observation


def record_game(env, policy, candidate_count=20, deterministic=True):
    """Run one game and return replay data.

    使用 OrbitWarsEnv 包装器接口：
      - env.reset() → (state, raw_obs)
      - env.collect_decisions(policy, state) → (decisions, transitions)
      - env.step(decisions, state) → (next_state, raw_obs, reward, done, info)
    """
    state, raw_obs = env.reset()
    frames = []
    done = False

    while not done:
        # 记录当前帧
        planets_by_id = {p.id: p for p in state.planets}
        frame = {
            'step': state.step,
            'planets': [],
            'fleets': [],
            'our_actions': [],
            'their_actions': [],
        }
        for p in state.planets:
            frame['planets'].append([p.id, p.owner, round(p.x, 4), round(p.y, 4),
                                     round(p.radius, 4), p.ships, p.production])
        for fl in state.fleets:
            frame['fleets'].append([fl.id, fl.owner, round(fl.x, 4), round(fl.y, 4),
                                    round(fl.angle, 4), fl.from_planet_id, fl.ships])

        # 收集决策并执行
        decisions, _ = env.collect_decisions(policy, state, deterministic=deterministic)

        # 记录本方的动作（用于 viewer 高亮）
        for source_id, target_id, ships, angle in decisions:
            frame['our_actions'].append([source_id, angle, ships])

        state, raw_obs, reward, done, info = env.step(decisions, state)
        frames.append(frame)

    # 构建汇总数据（从终局状态判断胜负）
    our_planets_count = sum(1 for p in state.planets if p.owner == 0)
    their_planets_count = sum(1 for p in state.planets if p.owner == 1)
    our_ships = sum(p.ships for p in state.planets if p.owner == 0)
    their_ships = sum(p.ships for p in state.planets if p.owner == 1)

    # 胜负判定：按行星数，平局按舰船数
    if our_planets_count > their_planets_count:
        our_reward, their_reward = 1, -1
    elif our_planets_count < their_planets_count:
        our_reward, their_reward = -1, 1
    elif our_ships > their_ships:
        our_reward, their_reward = 1, -1
    elif our_ships < their_ships:
        our_reward, their_reward = -1, 1
    else:
        our_reward, their_reward = 0, 0

    data = {
        'total_steps': len(frames),
        'our_reward': our_reward,
        'their_reward': their_reward,
        'our_planets': our_planets_count,
        'their_planets': their_planets_count,
        'our_ships': our_ships,
        'their_ships': their_ships,
        'frames': frames,
    }
    return data


def record_game_direct(env, agent0, agent1, episode_steps=500):
    """使用 env.run() 直接对战两个 agent，返回回放数据。

    agent0 = 己方 (player 0), agent1 = 对手 (player 1)。
    帧格式与 record_game() 完全兼容。
    """
    players = [agent0, agent1]
    env.reset()
    env.run(players)

    frames = []
    our_actions_all = []
    their_actions_all = []

    for step_data in env.steps:
        obs0 = step_data[0].observation
        obs1 = step_data[1].observation
        action0 = step_data[0].action or []
        action1 = step_data[1].action or []

        our_actions_all.append(action0)
        their_actions_all.append(action1)

        planets = obs0.get("planets", [])
        fleets = obs0.get("fleets", [])
        frame = {
            "step": obs0.get("step", 0),
            "planets": [[p[0], p[1], round(p[2], 4), round(p[3], 4),
                         round(p[4], 4), p[5], p[6]] for p in planets],
            "fleets": [[f[0], f[1], round(f[2], 4), round(f[3], 4),
                       round(f[4], 4), f[5], f[6]] for f in fleets],
            "our_actions": [],
            "their_actions": [],
        }
        frames.append(frame)

    # 填充每个帧的动作
    for i, frame in enumerate(frames):
        if i < len(our_actions_all):
            for act in our_actions_all[i]:
                if len(act) >= 3:
                    frame["our_actions"].append([act[0], act[1], act[2]])
        if i < len(their_actions_all):
            for act in their_actions_all[i]:
                if len(act) >= 3:
                    frame["their_actions"].append([act[0], act[1], act[2]])

    # 终局状态
    final_obs = env.steps[-1][0].observation
    final_state = parse_observation(final_obs, episode_steps=episode_steps)
    our_planets_count = sum(1 for p in final_state.planets if p.owner == 0)
    their_planets_count = sum(1 for p in final_state.planets if p.owner == 1)
    our_ships = sum(p.ships for p in final_state.planets if p.owner == 0)
    their_ships = sum(p.ships for p in final_state.planets if p.owner == 1)

    if our_planets_count > their_planets_count:
        our_reward, their_reward = 1, -1
    elif our_planets_count < their_planets_count:
        our_reward, their_reward = -1, 1
    elif our_ships > their_ships:
        our_reward, their_reward = 1, -1
    elif our_ships < their_ships:
        our_reward, their_reward = -1, 1
    else:
        our_reward, their_reward = 0, 0

    return {
        "total_steps": len(frames),
        "our_reward": our_reward,
        "their_reward": their_reward,
        "our_planets": our_planets_count,
        "their_planets": their_planets_count,
        "our_ships": our_ships,
        "their_ships": their_ships,
        "frames": frames,
    }


def update_list(base_dir):
    """Update replays/list.json index."""
    result = {'wins': [], 'losses': []}
    for sub in ['wins', 'losses']:
        d = os.path.join(base_dir, sub)
        if os.path.isdir(d):
            result[sub] = sorted([f for f in os.listdir(d) if f.endswith('.json')], reverse=True)
    with open(os.path.join(base_dir, 'list.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"list.json: {len(result['wins'])} wins, {len(result['losses'])} losses")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Orbit Wars replays')
    parser.add_argument('-n', type=int, default=1, help='Number of games')
    parser.add_argument('--agent', type=str, default='policy',
                       choices=['policy', 'search', 'mcts'],
                       help='Player 0 agent type (default: policy)')
    parser.add_argument('--opponent', type=str, default='random', help='Opponent type')
    parser.add_argument('--ckpt', type=str, default=None, help='Policy checkpoint to load')
    parser.add_argument('--steps', type=int, default=500, help='Max episode steps')
    parser.add_argument('--candidates', type=int, default=20, help='Candidate count')
    args = parser.parse_args()

    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(project, 'replays')
    os.makedirs(os.path.join(base_dir, 'wins'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'losses'), exist_ok=True)

    opponent = _build_opponent(args.opponent)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    if args.agent == 'search':
        # 搜索智能体 vs 对手 (直接用 Kaggle env.run)
        from kaggle_environments import make
        agent0 = SearchOpponent(episode_steps=args.steps)
        for i in range(1, args.n + 1):
            env = make("orbit_wars", debug=True,
                      configuration={"episodeSteps": args.steps})
            data = record_game_direct(env, agent0, opponent, episode_steps=args.steps)
            our_rew = data['our_reward']
            sub = 'wins' if our_rew > 0 else 'losses'
            fname = f"game_{timestamp}_{i:02d}_{data['our_planets']+data['their_planets']}p_{data['our_ships']+data['their_ships']}s.json"
            fpath = os.path.join(base_dir, sub, fname)
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            print(f"[{i}/{args.n}] {'赢' if our_rew > 0 else '输'} → {sub}/{fname}")
    elif args.agent == 'mcts':
        # MCTS 智能体 vs 对手 (直接用 Kaggle env.run)
        from kaggle_environments import make
        agent0 = MCTSOpponent(episode_steps=args.steps)
        for i in range(1, args.n + 1):
            env = make("orbit_wars", debug=True,
                      configuration={"episodeSteps": args.steps})
            data = record_game_direct(env, agent0, opponent, episode_steps=args.steps)
            our_rew = data['our_reward']
            sub = 'wins' if our_rew > 0 else 'losses'
            fname = f"game_{timestamp}_{i:02d}_{data['our_planets']+data['their_planets']}p_{data['our_ships']+data['their_ships']}s.json"
            fpath = os.path.join(base_dir, sub, fname)
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            print(f"[{i}/{args.n}] {'赢' if our_rew > 0 else '输'} → {sub}/{fname}")
    else:
        # 策略网络 vs 对手 (使用 OrbitWarsEnv 包装器)
        from src.env.wrapper import OrbitWarsEnv
        from src.policy.model import PolicyNetwork
        import torch
        policy = PolicyNetwork(hidden=256)
        if args.ckpt:
            ckpt = torch.load(args.ckpt, map_location='cpu')
            policy.load_state_dict(ckpt['policy_state_dict'])
            print(f"加载 {args.ckpt}")

        env = OrbitWarsEnv(opponent=opponent, candidate_count=args.candidates,
                           episode_steps=args.steps)

        for i in range(1, args.n + 1):
            data = record_game(env, policy, deterministic=False)
            our_rew = data['our_reward']
            sub = 'wins' if our_rew > 0 else 'losses'
            fname = f"game_{timestamp}_{i:02d}_{data['our_planets']+data['their_planets']}p_{data['our_ships']+data['their_ships']}s.json"
            fpath = os.path.join(base_dir, sub, fname)
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            print(f"[{i}/{args.n}] {'赢' if our_rew > 0 else '输'} → {sub}/{fname}")

    update_list(base_dir)
    print(f"\n用浏览器打开 replays/viewer.html 即可观看回放")


def _build_opponent(opponent_str: str):
    """根据字符串构建对手对象。"""
    if opponent_str == "random":
        return "random"
    elif opponent_str == "sniper":
        return SniperOpponent()
    elif opponent_str == "heuristic":
        return HeuristicOpponent()
    elif opponent_str == "lb1200":
        return LB1200Opponent()
    elif opponent_str == "v4_hybrid" or opponent_str == "V4HybridOpponent":
        return V4HybridOpponent()
    elif opponent_str == "search":
        from src.opponents.search_opponent import SearchOpponent
        return SearchOpponent()
    elif opponent_str == "mcts":
        return MCTSOpponent()
    else:
        print(f"未知对手类型 '{opponent_str}'，回退到 random")
        return "random"


if __name__ == '__main__':
    main()
