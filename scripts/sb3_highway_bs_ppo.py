import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import highway_env  # noqa: F401

TRAIN = True

if __name__ == "__main__":
    # Create the environment
    env = gym.make("highway-bs-v0") # highway-fast-v0 , render_mode="rgb_array"
    obs, info = env.reset()

    # Create the model
    # Parallel environments
    vec_env = make_vec_env("CartPole-v1", n_envs=4)

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=250000)
    model.save("ppo_highway_bs")

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(2e4))
        model.save("highway_dqn/model")
        del model

    # # Run the trained model and record video
    # model = DQN.load("highway_dqn/model", env=env)
    # env = RecordVideo(
    #     env, video_folder="highway_dqn/videos", episode_trigger=lambda e: True
    # )
    # env.unwrapped.set_record_video_wrapper(env)
    # env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    # for videos in range(10):
    #     done = truncated = False
    #     obs, info = env.reset()
    #     while not (done or truncated):
    #         # Predict
    #         action, _states = model.predict(obs, deterministic=True)
    #         # Get reward
    #         obs, reward, done, truncated, info = env.step(action)
    #         # Render
    #         env.render()
    # env.close()