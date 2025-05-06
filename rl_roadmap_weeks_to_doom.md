
# 🧠 Reinforcement Learning Roadmap (6 Weeks to Doom Agent)

A structured, gradual learning path — balancing theory, code, and experimentation.

---

## ✅ Week 1: Deepen Core Concepts (Policy vs. Value)

**Goals:**
- Solidify understanding of value-based (e.g., DQN) vs. policy-based (e.g., PPO, A2C) methods
- Examine how reward shaping influences learning
- Continue improving modular code and experiment tracking

**Tasks:**
- Improve reward shaping on CartPole (which you're already doing)
- Refactor plotting + evaluation into utility files
- Read:
  - Sutton & Barto Chapters 1–6 (skim if already familiar)
  - Stable-Baselines3 docs (custom policies, callbacks)

---

## 🔁 Week 2: Continuous Control & Environments

**Goals:**
- Move from discrete to continuous action spaces
- Experiment with PPO on LunarLanderContinuous-v2 or Pendulum-v1
- Learn about normalization and scaling rewards

**Tasks:**
- Train agents on at least 2 continuous environments
- Compare PPO vs. A2C
- Visualize reward curves

---

## 🧠 Week 3: CNNs and Image Observations

**Goals:**
- Understand how RL agents process visual input
- Train a CNN-based policy (e.g., using Atari games like Breakout)
- Learn about frame stacking, grayscale, resizing

**Tasks:**
- Train PPO or DQN with `CnnPolicy` on `BreakoutNoFrameskip-v4`
- Use `VecFrameStack`, `VecTransposeImage`, `Monitor`
- Save training videos

**Read/Watch:**
- OpenAI Baselines paper on Atari
- SB3 docs on image input pipelines

---

## 🎯 Week 4: Transition to VizDoom (Basic Scenario)

**Goals:**
- Create a custom `VizDoom` gym-compatible environment
- Train PPO on the `basic.cfg` scenario
- Learn how to preprocess Doom frames and reward signals

**Tasks:**
- Write `vizdoom_env.py` wrapper
- Design simple shaping rewards (e.g., +1 for hitting, -1 for damage)
- Evaluate and plot results

---

## 🧱 Week 5: Complex VizDoom + Memory (RNNs)

**Goals:**
- Experiment with `defend_the_center` or `deadly_corridor` scenario
- Explore recurrent policies (LSTMs/GRUs with PPO)
- Analyze long-term dependencies in agent behavior

**Tasks:**
- Train with `RecurrentPPO` (from SB3-Contrib)
- Plot agent health, ammo, kill counts
- Introduce longer episodes, sparse rewards

---

## 🧪 Week 6: Evaluation, Curriculum, and Experimentation

**Goals:**
- Compare different algorithms, reward schemes, and hyperparameters
- Try curriculum learning (gradually increasing difficulty)
- Write final training loop with plotting, video saving, and model saving

**Tasks:**
- Build experiment manager or run grid of configs
- Record videos of agent performance
- Write a short report or presentation of findings

---

## 📌 Tools You’ll Use Along the Way

- **SB3**: PPO, A2C, DQN, RecurrentPPO
- **Gymnasium / VizDoom**: Environments
- **Matplotlib + TensorBoard**: Logging
- **Python**: Utility scripts, wrappers, and reproducibility

