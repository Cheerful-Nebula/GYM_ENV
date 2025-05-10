import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from my_helper_funcs import save_reward_plot_os_specific, plot_all_curves_with_note, evaluate_model, plot_position_curves_with_note
from reward_shaping import RewardShaper
from datetime import date
from reward_shaping_mountaincar import MountainCarProgressRewardShaper # noqa

# Settings
num_evaluation_episodes = 4
network_policy = "MlpPolicy"
learning_algo_list = ['DQN', 'A2C', 'PPO']
training_increment = 10000
total_training_steps = 50000
env_id = 'LunarLander-v3'  # CartPole-v1 'LunarLander-v3' MountainCar-v0 Acrobot-v1 Pendulum-v1(continous, no DQN)
experiment_num = 1
target_obs_indices = [0, 1]

# Define shaping conditions, use 'prev_obs' for previous observation as value to compare against in 'threshold'
shaping_conditions = [
    {'index': 0, 'operation': '>', 'threshold': 'prev_obs', 'reward_modifier': 2, 'description': '+2 : position > prev_position\ni.e. moving to right'}, # noqa
    {'index': 0, 'operation': '<', 'threshold': 'prev_obs', 'reward_modifier': -0.5, 'description': '-0.5 : position > prev_position\ni.e. moving to left'} # noqa
]

# Initialize the reward shaper object
reward_shaper = RewardShaper(shaping_conditions)

# Initialize lists to store curves
eval_curves = []

# Main Loop
for learning_algo in learning_algo_list:
    evaluation_rewards = []
    timesteps_recorded = []
    all_position_stats = []
    position_curves = []
    # Initialize the environment
    env = gym.make(env_id)
    model = None  # Initialize model outside the loop

    print(f"Starting training for {learning_algo}...\n")

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

        print(f"Training {learning_algo} for {training_increment} additional timesteps (total: {current_timesteps})...\n")
        model.learn(total_timesteps=training_increment, reset_num_timesteps=False)

        # Run evaluation loop, returns average reward of all episodes final rewards
        avg_reward, avg_position_stats = evaluate_model(
            model=model,
            env_id=env_id,
            num_episodes=num_evaluation_episodes,
            reward_shaper=reward_shaper,
            render=False,
            show_applied_conditions=False,  # print statement when a reward condition is triggered
            record_video=False,
            experiment_num=experiment_num,
            position_indices=target_obs_indices  # List of indices to track (e.g., [0] for x, [0,1] for x,y)
            )
        print(f"Average reward after {current_timesteps} timesteps: {avg_reward}\nmodel timestep count: {model.num_timesteps}\n")
        # Build curve for average X position
        avg_x = avg_position_stats[0]["avg"]
        position_curves.append({
            "x": current_timesteps,
            "y": avg_x,
            "label": "Avg X Pos"
        })

        evaluation_rewards.append(avg_reward)
        timesteps_recorded.append(current_timesteps)

    eval_curves.append({
        "x": timesteps_recorded.copy(),
        "y": evaluation_rewards.copy(),
        "label": learning_algo
    })
    # Group x/y values from position_curves by label for plotting
    plot_ready_data = {}
    print("position_curves\n", position_curves)
    for point in position_curves:
        label = point["label"]
        if label not in plot_ready_data:
            plot_ready_data[label] = {"x": [], "y": [], "label": label}
        plot_ready_data[label]["x"].append(point["x"])
        plot_ready_data[label]["y"].append(point["y"])

    curves_list = list(plot_ready_data.values())

    plot_position_curves_with_note(
        curves=curves_list,
        y_label="Avg X Position (per eval)",
        title=f"{learning_algo} in {env_id}: Avg X Position over Training Progress",
        notes=f"Tracked average x-position over {num_evaluation_episodes} episodes at each checkpoint"
        )

    # This line is where training/evaluation loop for each algorithm ends --------------------

# Plot all the curves, then Save final combined plot
plot_all_curves_with_note(eval_curves, shaping_conditions=shaping_conditions)
today = date.today()
plot_filename = f"{env_id}_{today}_experiment{experiment_num}"
save_reward_plot_os_specific(plot_filename, env_id)
