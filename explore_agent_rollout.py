import gym
import os
import shutil
import ray
from ray.tune.registry import register_env
from explore_agent.envs.exploring_gym import ExploreDrone
import ray.rllib.algorithms.ppo.ppo as ppo
from ray.rllib.algorithms.ppo.ppo import PPOConfig
ray.init(ignore_reinit_error=True)
from time import sleep

chkpt_root = "tmp/ppo"
ray_results = "{}/ray_results/".format(os.getenv("HOME"))


select_env = "ExploreAgent-v0"
register_env(select_env, lambda config: ExploreDrone())
config = PPOConfig().resources(num_gpus=0).rollouts(num_rollout_workers=24)


config["log_level"] = "WARN"
agent = ppo.PPO(config, env=select_env)
chkpt_file = "tmp/ppo/checkpoint_best"

agent.restore(chkpt_file)
env = gym.make(select_env)
state,_ = env.reset()

sum_reward = 0
n_step = 500
for step in range(n_step):
    action = agent.compute_single_action(state)
    sleep(0.0416)
    state, reward, done, _,info = env.step(action)
    sum_reward += reward
    env.render()
    if done == 1:
        print("cumulative reward", sum_reward)
        state = env.reset()
        sum_reward = 0
