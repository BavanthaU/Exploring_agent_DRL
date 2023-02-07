import os
import shutil
import ray

from ray.tune.registry import register_env
from explore_agent.envs.exploring_gym import ExploreDrone
from ray.rllib.algorithms.ppo.ppo import PPOConfig
ray.init(ignore_reinit_error=True)



chkpt_root = "tmp/ppo"
#shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
ray_results = "{}/ray_results/".format(os.getenv("HOME"))
#shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


select_env = "ExploreAgent-v0"
register_env(select_env, lambda config: ExploreDrone())
config = PPOConfig().resources(num_gpus=0).rollouts(num_rollout_workers=24)

agent = config.build(env=select_env)
status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f}"
n_iter = 55555n in range(n_iter):
    result = agent.train()
    if n%50 ==0:
        chkpt_file = agent.save(chkpt_root)
    print(status.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"]
            ))
