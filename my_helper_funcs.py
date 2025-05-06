# my_helper_funcs.py
# This module contains helper functions for plotting and saving reward curves.
# It includes functions to save plots in a platform-specific manner and to plot multiple curves with annotations.
import matplotlib.pyplot as plt
import os
import platform
import numpy as np
import time
import gymnasium as gym


# Helper function to save plots
def save_reward_plot_os_specific(plot_filename, env_id: str):
    save_directory = os.path.join(env_id, "reward_plots")
    os.makedirs(save_directory, exist_ok=True)
    extension = ".png" if platform.system() != "Windows" else ".jpg"
    plot_path = os.path.join(save_directory, f"{plot_filename}{extension}")
    try:
        plt.savefig(plot_path)
        plt.close()
        plt.clf()
        print(f"Plot saved as: {plot_path}")
        return True
    except Exception as e:
        print(f"Error saving plot at {plot_path}: {e}")
        return False


# Function to plot all curves with a not
def plot_all_curves_with_note(curves, shaping_conditions=None):
    plt.figure(figsize=(12, 8))

    for curve in curves:

        plt.plot(curve["x"], curve["y"], marker='o', label=curve["label"])
        # Highlight best point for each curve
        # best_idx = np.argmax(curve["y"])
        # best_timestep = curve["x"][best_idx]
        # best_reward = curve["y"][best_idx]

        # plt.scatter(best_timestep, best_reward, s=60, label=f"{curve['label']} Best", zorder=5)

    plt.xlabel("Total Training Timesteps")
    plt.ylabel("Average Evaluation Reward")
    plt.title("Comparison of Algorithms and Reward Shaping Variants", fontsize=16)
    plt.grid(True)
    # Deduplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    print(by_label)
    plt.legend(by_label.values(), by_label.keys())

    if shaping_conditions:
        shaping_description = 'Reward Shaping Conditions:\n'
        for i in range(len(shaping_conditions)):
            shaping_description += shaping_conditions[i]['description']+"\n"

        plt.gcf().text(
            0.02, 0.02, shaping_description,
            fontsize=8, color="black", ha="left", va="bottom",
            bbox=dict(facecolor="white", alpha=1, edgecolor="black")
        )


def add_noise_to_obs(obs, noise_std=0.01):
    return obs + np.random.normal(0, noise_std, size=np.shape(obs))

# use this in your training loop
# obs = add_noise_to_obs(obs, noise_std=0.01)


def evaluate_model(model, env_id, num_episodes, reward_shaper=None, render=False, show_applied_conditions=False):
    ''' Function to evaluate the model
    This function evaluates the model over a specified number of episodes
    and returns the average reward.'''

    # Create Evaluation Environment
    if render:
        env = gym.make(env_id, render_mode='human')
    else:
        env = gym.make(env_id)

    all_episode_final_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        if reward_shaper:
            reward_shaper.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if reward_shaper:
                reward, applied_conditions = reward_shaper.shape(reward, obs)
            # Optional: log condition triggers
                if applied_conditions and show_applied_conditions:
                    for cond in applied_conditions:
                        print(f"Condition met: {cond['description']} | Obs Value: {cond['obs_value']:.2f} | Reward Modifier: {cond['reward_modifier']} | Reward: {reward}") # noqa

            episode_reward += reward
            if render:
                env.render()
                time.sleep(0.02)

        all_episode_final_rewards.append(episode_reward)
    env.close()
    # Return the average of all episode final rewards
    return np.mean(all_episode_final_rewards)
