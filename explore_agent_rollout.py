import gym
import os
import shutil
import ray
from ray.tune.registry import register_env
from explore_agent.envs.exploring_gym import ExploreDrone
import ray.rllib.agents.ppo as ppo
ray.init(ignore_reinit_error=True)
from time import sleep

chkpt_root = "tmp/exa"
ray_results = "{}/ray_results/".format(os.getenv("HOME"))


select_env = "ExploreAgent-v0"
register_env(select_env, lambda config: ExploreDrone())
config = ppo.DEFAULT_CONFIG.copy()

config["log_level"] = "WARN"
agent = ppo.PPOTrainer(config, env=select_env)
chkpt_file = "tmp/exa/checkpoint_000082"

agent.restore(chkpt_file)
env = gym.make(select_env)
state = env.reset()

sum_reward = 0
n_step = 500
for step in range(n_step):
    action = agent.compute_single_action(state)
    sleep(0.0416)
    state, reward, done, info = env.step(action)
    sum_reward += reward
    env.render()
    if done == 1:
        print("cumulative reward", sum_reward)
        state = env.reset()
        sum_reward = 0
