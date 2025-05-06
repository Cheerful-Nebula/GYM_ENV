import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
import numpy as np
from my_helper_funcs import save_reward_plot_os_specific, plot_all_curves_with_note

# Settings
num_evaluation_episodes = 10
network_policy = "MlpPolicy"
learning_algo_list = ['DQN', 'A2C', 'PPO']
training_increment = 20000
total_training_steps = 1000000
env_id = 'LunarLander-v3'

all_curves = []
final_curves = []

# Main Loop
for learning_algo in learning_algo_list:
    evaluation_rewards = []
    timesteps_recorded = []
    env = gym.make(env_id)
    model = None  # Initialize model outside the loop

    print(f"Starting training for {learning_algo}...")

    for current_timesteps in range(training_increment, total_training_steps + training_increment, training_increment):
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

        print(f"Training {learning_algo} for {training_increment} additional timesteps (total: {current_timesteps})...")
        model.learn(total_timesteps=training_increment, reset_num_timesteps=False)

        # Create Evaluation Environment
        eval_env = gym.make(env_id)
        all_episode_rewards = []

        for episode in range(num_evaluation_episodes):
            episode_reward = 0
            terminated = False
            truncated = False
            obs, info = eval_env.reset()

            while not terminated and not truncated:
                # Initial values for this episode
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                position = obs[0]
                # Reward shaping

                episode_reward += reward

            all_episode_rewards.append(episode_reward)

        avg_reward = np.mean(all_episode_rewards)
        evaluation_rewards.append(avg_reward)
        timesteps_recorded.append(current_timesteps)
        eval_env.close()

        all_curves.append({
            "x": timesteps_recorded.copy(),
            "y": evaluation_rewards.copy(),
            "label": learning_algo
        })

    env.close()
    # Saving last entry in all_curves
    # dictionary of all points for current algo
    final_curves.append(all_curves[-1])
    all_curves.clear()
    print(final_curves, "\n")
# Save final combined plot
plot_all_curves_with_note(final_curves, show_note=True)
plot_filename = f"{env_id}_AllAlgorithms_incremental_reward03_04"
save_reward_plot_os_specific(plot_filename)
