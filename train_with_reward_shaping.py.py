import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from my_helper_funcs import plot_all_curves_with_note, evaluate_model
from my_helper_funcs import save_reward_plot_os_specific
from reward_shaping import RewardShaper
from position_tracking import PositionTracker
from datetime import date


# Settings
num_evaluation_episodes = 10
num_rollouts = 10
network_policy = "MlpPolicy"
learning_algo_list = ['DQN', 'A2C', 'PPO']  # ['DQN', 'A2C', 'PPO']
training_timestep_increment = 2048
env_id = 'Acrobot-v1'  # CartPole-v1 'LunarLander-v3' MountainCar-v0 Acrobot-v1 Pendulum-v1(continous, no DQN)
experiment_num = 1
target_obs_indices = [0, 2]

# Define shaping conditions, use 'prev_obs' to use previous observation as value to compare against in 'threshold'
shaping_conditions = [
    {'index': 0, 'operation': '>', 'threshold': 'prev_obs', 'reward_modifier': 0, 'description': 'Environment Base Reward Only'}
 ] # noqa


# Initialize the reward shaper object
reward_shaper = RewardShaper(shaping_conditions)

# Initialize lists to store curves
eval_curves = []
today = date.today()

# Main Loop
for learning_algo in learning_algo_list:
    evaluation_rewards = []
    timesteps_recorded = []
    position_curves = []

    # Initialize the environment
    env = gym.make(env_id)
    model = None  # Initialize model outside the loop

    print(f"Starting training for {learning_algo}...\n")

    for current_timesteps in range(0, training_timestep_increment * num_rollouts, training_timestep_increment):
        # Choose or load model
        if model is None:
            if learning_algo == 'DQN':
                model = DQN(network_policy, env, verbose=0)
                tracker = PositionTracker(target_obs_indices, experiment_num, learning_algo, env_id)

            elif learning_algo == 'A2C':
                model = A2C(network_policy, env, verbose=0)
                tracker = PositionTracker(target_obs_indices, experiment_num, learning_algo, env_id)

            elif learning_algo == 'PPO':
                model = PPO(network_policy, env, verbose=0)
                tracker = PositionTracker(target_obs_indices, experiment_num, learning_algo, env_id)
        else:
            # The model already exists, so we continue training it
            pass

        print(f"Training {learning_algo} for additional {training_timestep_increment} timesteps (total: {model.num_timesteps + training_timestep_increment})...")  # noqa
        model.learn(total_timesteps=training_timestep_increment, reset_num_timesteps=False)

        # Run evaluation loop, returns average reward of all episodes final rewards
        avg_reward, tracker = evaluate_model(
            model=model,
            env_id=env_id,
            num_episodes=num_evaluation_episodes,
            reward_shaper=reward_shaper,
            render=False,
            show_applied_conditions=False,  # print statement when a reward condition is triggered
            tracker=tracker
            )
        print(f"Average reward after {model.num_timesteps} timesteps: {avg_reward}\n")

        evaluation_rewards.append(avg_reward)
        timesteps_recorded.append(model.num_timesteps)

    eval_curves.append({
        "x": timesteps_recorded.copy(),
        "y": evaluation_rewards.copy(),
        "label": learning_algo
    })
    # Plot position/ episode outcome values which have been recorded in tracker object after each rollout during training
    tracker.plot_position()
    tracker.plot_outcome()

    # Save position and outcome plot for each algorithm
    tracker.save_plot("position")
    tracker.save_plot("outcomes")

    # Delete tracker to free up memory
    del tracker
    print(f"Finished training for {learning_algo}.\n")
    # This line is where training/evaluation loop for each algorithm ends --------------------

# Plot all the curves, then Save final combined plot
plot_all_curves_with_note(eval_curves, env_id, shaping_conditions=shaping_conditions)
plot_filename = f"{today}_experiment{experiment_num}"
save_reward_plot_os_specific(plot_filename, env_id)
