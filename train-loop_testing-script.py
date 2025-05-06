import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from my_helper_funcs import save_reward_plot_os_specific, plot_all_curves_with_note, evaluate_model
from reward_shaping import RewardShaper
from datetime import date
from reward_shaping_mountaincar import MountainCarProgressRewardShaper # noqa

# Settings
num_evaluation_episodes = 5
network_policy = "MlpPolicy"
learning_algo_list = ['DQN', 'A2C', 'PPO']
training_increment = 20000
total_training_steps = 500000
env_id = 'MountainCar-v0'  # CartPole-v1 'LunarLander-v3' MountainCar-v0 Acrobot-v1 Pendulum-v1(continous, no DQN)

# Define shaping conditions
shaping_conditions = [
    {'index': 1, 'operation': 'abs>=', 'threshold': 0.01, 'reward_modifier': 2, 'description': '+2 : abs(velocity) >= 0.01'},
    {'index': 1, 'operation': 'abs>=', 'threshold': 0.019, 'reward_modifier': 4, 'description': '+6 : abs(velocity) >= 0.019'},
    {'index': 1, 'operation': 'abs>=', 'threshold': 0.029, 'reward_modifier': 8, 'description': '+14 : abs(velocity) >= 0.029'},
    {'index': 1, 'operation': 'abs>=', 'threshold': 0.039, 'reward_modifier': 16, 'description': '+30 : abs(velocity) >= 0.039'},
    {'index': 1, 'operation': 'abs>=', 'threshold': 0.049, 'reward_modifier': 32, 'description': '+62 : abs(velocity) >= 0.049'},
    {'index': 0, 'operation': '>=', 'threshold': -0.2, 'reward_modifier': 20, 'description': '+20 : position >= -0.2'},
    {'index': 0, 'operation': '>=', 'threshold': 0.15, 'reward_modifier': 60, 'description': '+60 : position >= 0.15'},
    {'index': 0, 'operation': '>=', 'threshold': 0.5, 'reward_modifier': 100, 'description': '+100 : position >= 0.5'}
]

# Initialize the reward shaper object
'''if env_id == 'MountainCar-v0':
    reward_shaper = MountainCarProgressRewardShaper()
else:
    reward_shaper = RewardShaper(shaping_conditions)'''
reward_shaper = RewardShaper(shaping_conditions)
# Initialize lists to store curves
all_curves = []
final_curves = []

# Main Loop
for learning_algo in learning_algo_list:
    evaluation_rewards = []
    timesteps_recorded = []
    env = gym.make(env_id)
    model = None  # Initialize model outside the loop

    print(f"Starting training for {learning_algo}...")

    for current_timesteps in range(0, total_training_steps + training_increment, training_increment):
        # Choose or load model
        if model is None:
            if learning_algo == 'DQN':
                model = DQN(network_policy, env, verbose=0)
            elif learning_algo == 'A2C':
                model = A2C(network_policy, env, verbose=0)
            elif learning_algo == 'PPO':
                model = PPO(network_policy, env, verbose=0)
        else:
            # The model already exists, so we continue training it
            pass

        # Run evaluation loop, returns average reward of all episodes final rewards
        avg_reward = evaluate_model(
            model=model,
            env_id=env_id,
            num_episodes=num_evaluation_episodes,
            reward_shaper=reward_shaper,
            render=False,
            show_applied_conditions=False
            )
        evaluation_rewards.append(avg_reward)
        timesteps_recorded.append(current_timesteps)

        all_curves.append({
            "x": timesteps_recorded.copy(),
            "y": evaluation_rewards.copy(),
            "label": learning_algo
        })

        print(f"Training {learning_algo} for {training_increment} additional timesteps (total: {current_timesteps})...")
        model.learn(total_timesteps=training_increment, reset_num_timesteps=False)

    # Saving last entry in all_curves
    # dictionary of all points for current algo
    final_curves.append(all_curves[-1])
    all_curves.clear()
    print(final_curves, "\n")
# Plot all the curves, then Save final combined plot
plot_all_curves_with_note(final_curves, shaping_conditions=shaping_conditions)
today = date.today()
plot_filename = f"{env_id}_{today}_experiment02"
save_reward_plot_os_specific(plot_filename, env_id)
