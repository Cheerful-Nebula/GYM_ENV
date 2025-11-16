import gymnasium as gym
import ale_py # ignore
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3 import DQN, PPO, A2C
from my_helper_funcs import plot_all_curves_with_note, evaluate_model
from my_helper_funcs import save_reward_plot_os_specific
from reward_shaping import RewardShaper
from position_tracking import PositionTracker
from datetime import date
from config import Config


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
for learning_algo in Config.learning_algo_list:
    evaluation_rewards = []
    timesteps_recorded = []
    position_curves = []

    # Initialize the environment
    env = gym.make(Config.env_id)
    model = None  # Initialize model outside the loop

    if Config.env_id.startswith('ALE/'):
        # Models 'look' at pixels for 'ALE' environments, so we need to wrap the env 
        # in this wrapper, which automatically grayscales, resizes, and stacks frames
        env = AtariWrapper(env)
        Config.network_policy = "CnnPolicy"
    
    print(f"Starting training for {learning_algo}...\n")

    for current_timesteps in range(0, Config.training_timestep_increment * Config.num_rollouts, Config.training_timestep_increment):
        # Choose or load model
        if model is None:
            if learning_algo == 'DQN':
                model = DQN(Config.network_policy, env, verbose=1, buffer_size=100000)
                tracker = PositionTracker(Config.target_obs_indices, Config.experiment_num, learning_algo, Config.env_id)

            elif learning_algo == 'A2C':
                model = A2C(Config.network_policy, env, verbose=1)
                tracker = PositionTracker(Config.target_obs_indices, Config.experiment_num, learning_algo, Config.env_id)

            elif learning_algo == 'PPO':
                model = PPO(Config.network_policy, env, verbose=1)
                tracker = PositionTracker(Config.target_obs_indices, Config.experiment_num, learning_algo, env_id)
        else:
            # The model already exists, so we continue training it
            pass

        print(f"Training {learning_algo} for additional {Config.training_timestep_increment} timesteps (total: {model.num_timesteps + training_timestep_increment})...")  # noqa
        model.learn(total_timesteps=Config.training_timestep_increment, reset_num_timesteps=False)

        # Run evaluation loop, returns average reward of all episodes final rewards
        avg_reward, tracker = evaluate_model(
            model=model,
            env_id=Config.env_id,
            num_episodes=Config.num_evaluation_episodes,
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
plot_all_curves_with_note(eval_curves, Config.env_id, shaping_conditions=shaping_conditions)
plot_filename = f"{today}_experiment{Config.experiment_num}"
save_reward_plot_os_specific(plot_filename, Config.env_id)
