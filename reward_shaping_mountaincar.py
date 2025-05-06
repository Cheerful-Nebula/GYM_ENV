# reward_shaping_mountaincar.py
from reward_shaping import RewardShaper


class MountainCarProgressRewardShaper(RewardShaper):
    def __init__(self):
        super().__init__()
        self.max_position = -float('inf')  # start lower than any env position

    def reset(self):
        self.max_position = -float('inf')

    def shape(self, obs, reward, done):
        position = obs[0]
        shaped_reward = reward

        if position > self.max_position:
            shaped_reward += (position - self.max_position) * 10  # scale bonus
            self.max_position = position

        return shaped_reward
