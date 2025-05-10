# my_helper_funcs.py
# This module contains helper functions for plotting and saving reward curves.
# It includes functions to save plots in a platform-specific manner and to plot multiple curves with annotations.
import matplotlib.pyplot as plt
import os
import platform
import numpy as np
import time
import gymnasium as gym
from datetime import date


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


# Function to plot all curves with a note
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
                   show_applied_conditions=False, record_video=False, experiment_num=None,
                   position_indices=None):
    """
    Evaluates a trained RL model over multiple episodes and tracks selected position stats (e.g., x, y).

    Returns:
        avg_reward (float): Mean reward over all episodes
        avg_position_stats (dict): A dictionary for each tracked observation index with average, min, max stats
    """
    all_episode_final_rewards = []

    # Prepare dictionary to collect per-index position stats
    position_stats = {i: [] for i in (position_indices or [])}

    today = str(date.today())
    video_directory = os.path.join(env_id, "eval_videos", today)
    os.makedirs(video_directory, exist_ok=True)

    for ep in range(num_episodes):
        # Handle rendering or video recording for last episode
        if record_video and ep == num_episodes - 1:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_directory,
                name_prefix=f"experiment{experiment_num}_{model.num_timesteps}steps_eval",
                episode_trigger=lambda i: True,
                disable_logger=True
            )
        else:
            render_mode = "human" if render else None
            env = gym.make(env_id, render_mode=render_mode)

        obs, _ = env.reset()
        if reward_shaper:
            reward_shaper.reset()

        done = False
        episode_reward = 0
        episode_positions = {i: [] for i in (position_indices or [])}  # Per-episode position tracker

        while not done:
            # Track specific observation values (e.g., x, y) across timesteps
            if position_indices:
                for i in position_indices:
                    episode_positions[i].append(obs[i])

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Optional reward shaping
            if reward_shaper:
                reward, applied_conditions = reward_shaper.shape(reward, obs)
                if applied_conditions and show_applied_conditions:
                    for cond in applied_conditions:
                        print(f"Condition met: {cond['description']} | Obs Value: {cond['obs_value']:.2f} | "
                              f"Reward Modifier: {cond['reward_modifier']} | Reward: {reward}")

            episode_reward += reward

            if render and not record_video:
                env.render()
                time.sleep(0.02)

        # Store episode total reward
        all_episode_final_rewards.append(episode_reward)

        # Summarize episode position stats (mean, min, max of each tracked observation)
        if position_indices:
            for i in position_indices:
                episode_arr = np.array(episode_positions[i])
                position_stats[i].append({
                    "avg": float(np.mean(episode_arr)),
                    "min": float(np.min(episode_arr)),
                    "max": float(np.max(episode_arr))
                })

        env.close()

    # Restructure data so that we average across all episodes per stat type
    avg_position_stats = {}
    for i, stats_list in position_stats.items():
        avg_position_stats[i] = {
            "avg": float(np.mean([s["avg"] for s in stats_list])),
            "min": float(np.mean([s["min"] for s in stats_list])),
            "max": float(np.mean([s["max"] for s in stats_list]))
        }

    return np.mean(all_episode_final_rewards), avg_position_stats

# ------------------------------
# GENERALIZED POSITION PLOTTING FUNCTION
# ------------------------------


def plot_position_curves_with_note(curves, y_label="Position Stat", title="Position Statistic Trends", notes=None):
    """
    Plots average or other statistic (min, max) over time for position-based metrics.

    Args:
        curves: list of dicts with 'x', 'y', and 'label' fields
        y_label: string label for Y-axis (e.g., 'Avg X Pos')
        title: title of the plot
        notes: Optional string note or metadata (e.g., shaping rules) to display on figure
    """
    plt.figure(figsize=(12, 8))

    for curve in curves:
        plt.plot(curve["x"], curve["y"], marker='o', label=curve["label"])

    plt.xlabel("Total Training Timesteps")
    plt.ylabel(y_label)
    plt.title(title, fontsize=16)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=10)

    if notes:
        plt.gcf().text(
            0.02, 0.02, notes,
            fontsize=8, color="black", ha="left", va="bottom",
            bbox=dict(facecolor="white", alpha=1, edgecolor="black")
        )
    plt.show()
