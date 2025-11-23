import ast
import json
from typing import Any, List

import gymnasium as gym
import streamlit as st
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
import torch

# Ensure these modules exist in your project structure
from config import Config
from reward_shaping import RewardShaper
from my_helper_funcs import evaluate_model, get_device
from position_tracking import PositionTracker

# --- Helpers ---
DEFAULT_ENV_LIST = [
    "CartPole-v1",
    "LunarLander-v3",
    "MountainCar-v0",
    "ALE/Breakout-v5",
    Config.env_id,
]


def parse_threshold(text: str, operation: str) -> Any:
    """
    Parse threshold input from the UI.
    """
    text = text.strip()
    if not text:
        return None
    if text.lower() == "prev_obs":
        return "prev_obs"
    if operation == "between":
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 2:
            raise ValueError("Between requires two comma-separated values")
        return (float(parts[0]), float(parts[1]))

    try:
        val = ast.literal_eval(text)
        return val
    except Exception:
        return float(text)


class RandomAgent:
    """A simple random agent for fallback purposes."""
    def __init__(self, action_space):
        self.action_space = action_space
        self.num_timesteps = 0

    def predict(self, obs, deterministic=True):
        return self.action_space.sample(), None

# --- Streamlit App ---


def main():
    st.set_page_config(page_title="RL Reward Shaper", layout="wide")
    st.title("RL Reward Shaping — Quick Runner")

    # Check device immediately
    device_type = get_device()

    st.sidebar.header("Hardware Status")
    if device_type == "cuda":
        st.sidebar.success(f"⚡ Running on NVIDIA GPU ({torch.cuda.get_device_name(0)})")
    elif device_type == "mps":
        st.sidebar.success("⚡ Running on Apple Metal (MPS)")
    else:
        st.sidebar.warning("🐢 Running on CPU")
    # -- Sidebar: Environment & Config --
    st.sidebar.header("Environment & Evaluation")
    env_choice = st.sidebar.selectbox("Choose environment", DEFAULT_ENV_LIST)
    custom_env_id = st.sidebar.text_input("Custom env id (overrides selection)", value="")

    # Logic to determine final Env ID
    env_id = custom_env_id.strip() if custom_env_id.strip() else env_choice

    num_rollouts = st.sidebar.number_input(
        "Number of rollouts", min_value=1, value=int(Config.num_rollouts)
    )
    num_eval_eps = st.sidebar.number_input(
        "Eval episodes", min_value=1, value=int(Config.num_evaluation_episodes)
    )
    render_env = st.sidebar.checkbox("Render env (Human)", value=False)

    st.sidebar.markdown("---")
    st.sidebar.header("Model Loading")
    model_path = st.sidebar.text_input(
        "SB3 Model Path (blank = Random Agent)", value=""
    )

    # -- Main Section: Conditions --
    st.header("Reward Shaping Conditions")
    st.markdown(
        "Define rules to modify the reward signal based on observation indices."
    )

    # Initialize session state for conditions
    if "conditions" not in st.session_state:
        st.session_state.conditions = []

    with st.form("add_condition", clear_on_submit=True):
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            c_index = st.number_input("Obs Index", min_value=0, value=0, step=1)
        with col2:
            c_operation = st.selectbox(
                "Operation",
                options=[
                    ">", "<", ">=", "<=", "==", "!=",
                    "abs>", "abs<", "abs>=", "abs<=", "between"
                ],
            )
        with col3:
            c_threshold = st.text_input("Threshold", placeholder="e.g., 0.5 or prev_obs")

        col4, col5 = st.columns([1, 2])
        with col4:
            c_reward_modifier = st.number_input("Reward Mod", value=0.0, format="%.4f")
        with col5:
            c_description = st.text_input("Description", placeholder="e.g., Avoid tilting")

        add_btn = st.form_submit_button("Add Condition")

        if add_btn:
            try:
                parsed_thresh = parse_threshold(c_threshold, c_operation)
                cond = {
                    "index": int(c_index),
                    "operation": c_operation,
                    "threshold": parsed_thresh,
                    "reward_modifier": float(c_reward_modifier),
                    "description": c_description,
                }
                st.session_state.conditions.append(cond)
                st.success("Condition added!")
            except Exception as e:
                st.error(f"Error parsing input: {e}")

    # -- Display Current Conditions --
    st.subheader(f"Active Conditions ({len(st.session_state.conditions)})")

    if st.session_state.conditions:
        # Iterate with index to allow deletion
        for i, c in enumerate(st.session_state.conditions):
            with st.container():
                c1, c2 = st.columns([5, 1])
                c1.code(json.dumps(c), language="json")
                # Best Practice: Unique keys for buttons in loops are mandatory
                if c2.button("Remove", key=f"rm_{i}"):
                    st.session_state.conditions.pop(i)
                    st.rerun()  # Use rerun() instead of experimental_rerun()
    else:
        st.info("No shaping conditions applied yet.")

    st.markdown("---")

    # -- Execution --
    if st.button("Run Evaluation", type="primary"):
        # Update Global Config (Note: Modifying globals in Streamlit can be tricky in multi-user apps,
        # but works for local single-user tools)
        Config.env_id = env_id
        Config.num_rollouts = int(num_rollouts)
        Config.num_evaluation_episodes = int(num_eval_eps)

        reward_shaper = RewardShaper(st.session_state.conditions)

        # 1. Environment Setup
        try:
            # Creating a dummy env just to get action space for the agent check
            env_for_agent = gym.make(env_id)
        except Exception as e:
            st.error(f"Failed to create environment '{env_id}': {e}")
            st.stop()

        # 2. Model Setup
        model = None
        if model_path.strip():
            try:
                # Attempt generic load; might need specific algo class (PPO.load) depending on file
                model = BaseAlgorithm.load(model_path, env=env_for_agent, device=device_type)
                st.info(f"Loaded model from {model_path} on {device_type.upper()}")
                st.success(f"Loaded model: {model_path}")
            except Exception as e:
                st.warning(f"Could not load model at '{model_path}': {e}")
                st.info("Falling back to Random Agent.")
                model = RandomAgent(env_for_agent.action_space)
        else:
            # If creating a fresh agent for testing (if you added that feature)
            model = PPO("MlpPolicy", env_for_agent, device=device_type)
            # model = RandomAgent(env_for_agent.action_space) # Random agent doesn't need devicemodel = RandomAgent(env_for_agent.action_space)

        # 3. Setup Position Tracker
            # We need to tell it which indices to track.
            # Defaulting to [0] or checking if we have conditions on specific indices.
            indices_to_track = [0, 1] if env_id == "LunarLander-v3" else [0]

            # If user added conditions, let's track those indices too!
            if st.session_state.conditions:
                indices_to_track = list(set(indices_to_track + [c['index'] for c in st.session_state.conditions]))

            tracker = PositionTracker(
                obs_idxs=indices_to_track,
                expt_num=999,  # Temporary ID for UI runs
                learning_algo="Streamlit-Eval",
                env_id=env_id
            )
# 4. Run Eval
    with st.spinner(f"Running {num_eval_eps} episodes on {env_id}..."):
        try:
            # Pass the tracker to your helper function
            avg_reward, filled_tracker = evaluate_model(
                model=model,
                env_id=env_id,
                num_episodes=int(num_eval_eps),
                reward_shaper=reward_shaper,
                render=render_env,
                show_applied_conditions=True,
                tracker=tracker
            )
            st.balloons()
            st.success(f"**Average Reward:** {avg_reward:.4f}")

            # 5. Visualize Results using YOUR existing scripts (NEW)
            st.markdown("---")
            st.subheader("Position Tracking Analysis")

            # Generate the plots internally in the tracker
            filled_tracker.plot_position()
            filled_tracker.plot_outcome()

            # Display Position Plots
            # Your class stores figures in a list of tuples: [(index, fig), ...]
            if hasattr(filled_tracker, 'pos_fig_list'):
                for idx, fig in filled_tracker.pos_fig_list:
                    st.write(f"**Observation Index {idx}**")
                    st.pyplot(fig)  # <--- This renders your matplotlib figure in the UI

            # Display Outcome Plots
            st.subheader("Episode Outcomes")
            if hasattr(filled_tracker, 'outcome_fig_list'):
                for idx, fig in filled_tracker.outcome_fig_list:
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Runtime Error during evaluation: {e}")
            # Helpful for debugging
            st.exception(e)

        # 4. Export Options
        st.download_button(
            label="Download JSON Config",
            data=json.dumps(st.session_state.conditions, indent=2),
            file_name="shaping_conditions.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
