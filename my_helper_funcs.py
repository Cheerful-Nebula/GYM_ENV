# my_helper_funcs.py
# This module contains helper functions for plotting and saving reward curves.
# It includes functions to save plots in a platform-specific manner and to plot multiple curves with annotations.
import matplotlib.pyplot as plt
import os
import platform
import numpy as np
import time
import gymnasium as gym
from position_tracking import PositionTracker
import torch


# ------------------------------
# SAVE FUNCTION FOR PLOTS
# ------------------------------
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


# Function to plot all reward curves with a note
def plot_all_curves_with_note(curves, env_id, shaping_conditions=None):
    plt.figure(figsize=(12, 8))

    for curve in curves:
        plt.plot(curve["x"], curve["y"], marker='+', label=curve["label"])

    plt.xlabel("Total Training Timesteps")
    plt.ylabel("Average Evaluation Reward")
    plt.title(f"Comparison of Algorithms and Reward Shaping Variants on {env_id}", fontsize=16)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=10)

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


# ------------------------------
# EVALUATION FUNCTION
# ------------------------------
def evaluate_model(model, env_id, num_episodes, reward_shaper=None, render=False,
                   show_applied_conditions=False, tracker: PositionTracker = None):
    """
    Evaluates a trained RL model over multiple episodes and tracks selected position stats (e.g., x, y).

    Returns:
        avg_reward (float): Mean reward over all episodes
        avg_position_stats (dict): A dictionary for each tracked observation index with average, min, max stats
    """
    all_episode_final_rewards = []
    if tracker:
        tracker.start_rollout()

    for _ in range(num_episodes):
        if tracker:
            tracker.start_episode()

        render_mode = "human" if render else None
        env = gym.make(env_id, render_mode=render_mode)

        obs, _ = env.reset()
        if reward_shaper:
            reward_shaper.reset()
        if tracker:
            tracker.track_episode(obs)

        done = False
        episode_reward = 0

        while not done:

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if tracker:
                tracker.track_episode(obs)

            # Optional reward shaping
            if reward_shaper:
                reward, applied_conditions = reward_shaper.shape(reward, obs)
                if applied_conditions and show_applied_conditions:
                    for cond in applied_conditions:
                        print(f"Condition met: {cond['description']} | Obs Value: {cond['obs_value']:.2f} | "
                              f"Reward Modifier: {cond['reward_modifier']} | Reward: {reward}")

            episode_reward += reward

            if render:
                env.render()
                time.sleep(0.02)

        # Store episode total reward
        all_episode_final_rewards.append(episode_reward)

        # Summarize episode position stats (mean, min, max of each tracked observation)
        if tracker:
            tracker.end_episode(terminated)
        env.close()

    if tracker:
        tracker.end_rollout(model.num_timesteps)

    return np.mean(all_episode_final_rewards), tracker


# ------------------------------
# UTILITY FUNCTIONS
# ------------------------------

def get_device():
    """
    Returns the most powerful available device.
    Priority: CUDA (Nvidia) > MPS (Apple Metal) > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
