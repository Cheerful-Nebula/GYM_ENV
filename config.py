# File where all configuration settings are stored

class Config:
    num_evaluation_episodes = 10
    num_rollouts = 10
    network_policy = "MlpPolicy"  # "MlpPolicy"/ "CnnPolicy" for ALE envs
    learning_algo_list = ['DQN', 'A2C', 'PPO']  # ['DQN', 'A2C', 'PPO']
    training_timestep_increment = 2048
    env_id = 'ALE/Breakout-v5'  # CartPole-v1 'LunarLander-v3' MountainCar-v0 Acrobot-v1 Pendulum-v1(continous, no DQN)
    experiment_num = 1
    target_obs_indices = [0]