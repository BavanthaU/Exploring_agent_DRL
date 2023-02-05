from gym.envs.registration import register


register(
    id='ExploreAgent-v0',
    entry_point='explore_agent.envs:ExploreDrone',
)
