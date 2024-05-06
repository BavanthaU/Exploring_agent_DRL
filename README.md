How to use the updated version :

Main Reuquirments: 
```
pip install -U tensorflow ray ray[rllib]
```

How to train: 
Updated only PPO to follow the gymnasium chnages (library update from gym to gymnasium), please update other files (mostly about how to import the alogorithm) 
```
python Explore_PPO_agent_training.py
```

How to test trained agent:
```
python explore_agent_rollout.py

make sure to update :
chkpt_file = "tmp/ppo/checkpoint_best"
```

Visualise Results:
```
pip install tensorboard
python3 -m tensorboard.main --logdir=ray_results/
```
