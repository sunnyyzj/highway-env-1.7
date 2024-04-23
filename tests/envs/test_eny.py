import highway_env
import gymnasium as gym
# from pathlib import Path

# import highway_env.envs
# from rl_agents.agents.common.factory import load_environment, load_agent
# from rl_agents.trainer.evaluation import Evaluation
# from rl_agents.agents.common.factory import agent_factory
# import sys
# from tqdm.notebook import trange
# from datetime import datetime
# sys.path.insert(0, './highway-env/scripts/')
# # from utils import record_videos, show_videos

highway_env.register_highway_envs()
env1 = gym.make("highway-fast-v0", render_mode="rgb_array")
env = gym.make("highway-bs-v0")
# env=gym.make("highway-bs-v0", render_mode="rgb_array")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.render()