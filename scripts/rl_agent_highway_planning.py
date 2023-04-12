import gymnasium as gym
import highway_env
import sys
from tqdm.notebook import trange
sys.path.insert(0, './highway-env/scripts/')
# from utils import record_videos, show_videos
from rl_agents.agents.common.factory import agent_factory
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from gymnasium.wrappers import RecordVideo
from pathlib import Path
import base64


display = Display(visible=0, size=(1400, 900))
display.start()


def record_videos(env, video_folder="videos"):
    wrapped = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)

    # Capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped


def show_videos(path="videos"):
    html = []
    for mp4 in Path(path).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


# Make environment
env = gym.make("highway-fast-v0", render_mode="rgb_array")
env = record_videos(env)
(obs, info), done = env.reset(), False

# Make agent
agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
    "env_preprocessors": [{"method":"simplify"}],
    "budget": 50,
    "gamma": 0.7,
}
agent = agent_factory(env, agent_config)

# Run episode
for step in trange(env.unwrapped.config["duration"], desc="Running..."):
    action = agent.act(obs)
    obs, reward, done, truncated, info = env.step(action)
    
env.close()
show_videos()