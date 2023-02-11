import time
import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.envs import highway
from mo_gymnasium.utils import MORecordEpisodeStatistics
from gymnasium.wrappers import FlattenObservation
from morl_baselines.common.scalarization import tchebicheff
from morl_baselines.single_policy.ser.mo_q_learning import MOQLearning

import highway_env
# from highway_env.envs.mo_highway_env import MOHighwayEnvFastBS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == "__main__":
    # env = MORecordEpisodeStatistics(DeepSeaTreasure(dst_map=CONCAVE_MAP), gamma=0.9)
    env = mo_gym.make("mo-highwaybs-fast-v0")
    env = MORecordEpisodeStatistics(env, gamma=0.9)
    # print(type(env.observation_space))
    # print( env.observation_space.shape)
    env = FlattenObservation(env)
    # print( env.observation_space.shape)
    # eval_env = gym.wrappers.TimeLimit(DeepSeaTreasure(dst_map=CONCAVE_MAP), 500)
    eval_env = gym.wrappers.TimeLimit(env, 500)
    eval_env = FlattenObservation(eval_env)
    scalarization = tchebicheff(tau=4.0, reward_dim=3)
    weights = np.array([0.5, 0.5])

    agent = MOQLearning(env, scalarization=scalarization, weights=weights, log=True)
    agent.train(
        total_timesteps=int(5e4), #1e5
        start_time=time.time(),
        eval_freq=10, #100
        eval_env=eval_env,
    )

    print(mo_gym.eval_mo(agent, env=eval_env, w=weights))
