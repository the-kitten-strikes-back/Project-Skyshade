# ⚡ AeroMania RL

> "Teach a plane to dogfight, they said. It'll be fun, they said. They were correct."

A reinforcement learning dogfight project built in Python, where a PPO agent learns to fly, chase, align, and shoot inside a live `Ursina` 3D combat sandbox. This is not a sterile little gridworld. This is an aircraft with missiles, guns, and violent opinions. Damage is dealt, rewards are given, and the agent learns to be a *slightly less* embarrassing pilot over time. DO NOT expect it to win any airshows, but it might just survive a few rounds of aerial chaos.




## ✨ Features

This project comes with enough moving parts to keep both RL people and game-dev people pleasantly stressed:

| Module | What It Does |
| --- | --- |
| ✈️ `rl_env.py` | Wraps the combat sandbox as a custom `Gymnasium` environment |
| 🧠 `train_rl.py` | Trains a `stable-baselines3` PPO agent with a Rich progress bar |
| 🎮 `evaluate_ai.py` | Loads a trained model and lets you watch it dogfight in visible 3D |
| 🔥 `entities.py` | Handles planes, missiles, flares, damage, and other aviation-adjacent chaos |
| 💾 `models/checkpoints/` | Stores training checkpoints every `10000` steps |
| 📈 `ppo_aeromania_logs/` | TensorBoard logs for people who enjoy graphs almost as much as explosions |

## 🛠️ One Repo, Two Moods

### 🧪 Training Mode — `train_rl.py`

For when you want the machine to suffer in silence and become stronger.

```bash
python train_rl.py
```

Runs the environment in headless mode, trains PPO for `100000` timesteps, saves checkpoints along the way, and writes the final model to `models/aeromania_rl_model_final.zip`.

### 👀 Watch It Fly — `evaluate_ai.py`

For when you want to stare directly at the consequences of your reward function.

```bash
python evaluate_ai.py
```

This opens the environment in `human` mode and lets the trained pilot fly in a visible `Ursina` scene like it pays rent there.

## 🚀 Quick Start

### 1. Install dependencies

There is no `requirements.txt` in the repo right now, so here is the direct approach:

```bash
pip install ursina gymnasium numpy stable-baselines3 rich torch tensorboard
```

### 2. Train the agent

```bash
python train_rl.py
```

What happens:

- Builds `AeroManiaEnv` in headless mode
- Creates a PPO agent with `MlpPolicy`
- Trains for `100000` timesteps
- Saves checkpoints every `10000` steps
- Saves the final model under `models/`

### 3. Watch the trained pilot

```bash
python evaluate_ai.py
```

If the final model exists, it loads that. If not, it falls back to a checkpoint. Because adaptability is a virtue and also a coping mechanism.

### 4. Peek at the training curves

```bash
tensorboard --logdir ppo_aeromania_logs
```

For those magical moments when you want visual proof that the line is, in fact, going somewhere.

## 🧠 Environment Breakdown

This environment puts an `RLPlane` against an `EnemyPlane` and asks the eternal question:

Can a policy network learn aerial violence with style?

### Action Space

The action space is a continuous `Box(-1, 1, shape=(4,))`:

| Control | What It Means |
| --- | --- |
| `pitch` | Nose up, nose down, hope for the best |
| `yaw` | Turn left or right like your life depends on it |
| `throttle slot` | Present in the interface, currently not actively used in training |
| `fire` | Guns at lower threshold, missiles at higher threshold, poor decisions at any threshold |

### Observation Space

The observation vector has 8 values:

| Signal | Why It Exists |
| --- | --- |
| Relative target position `x, y, z` | So the agent knows where the problem is |
| Own rotation `pitch, yaw` | So it knows which way it's currently being dramatic |
| Own speed | Because stalling is a lifestyle, not a strategy |
| Distance to target | Important if you wish to shoot things accurately |
| Alignment score | Measures whether the enemy is actually in front of the aircraft |

### Reward Function

The reward setup encourages behavior that vaguely resembles competent pursuit instead of interpretive aviation.

- Damage dealt gives positive reward
- Facing the enemy gives a small ongoing reward
- Getting within `1000m` earns a bonus
- Drifting beyond `5000m` earns a penalty
- Episodes terminate if someone dies, the planes get too close, or the target is basically gone

## 📦 Training Artifacts

The repo already includes some goodies:

- `models/aeromania_rl_model_final.zip`
- multiple saved checkpoints in `models/checkpoints/`
- TensorBoard logs in `ppo_aeromania_logs/`

So yes, there is already a pilot in here. Whether it is a good pilot is between you and the evaluation script.

## 🧭 Repo Tour

| File | Role |
| --- | --- |
| `rl_env.py` | Defines `AeroManiaEnv`, reset/step logic, observations, rewards, and headless vs human rendering |
| `train_rl.py` | Entry point for PPO training and checkpoint saving |
| `evaluate_ai.py` | Loads a saved policy and runs infinite evaluation episodes |
| `entities.py` | Implements `RLPlane`, `EnemyPlane`, missiles, flares, and combat behavior |
| `bootstrap.py` | Broader game bootstrap and non-RL project context |

## 🔧 Good Next Upgrades

Because every RL project is just a to-do list wearing sunglasses:

- add real throttle control to the action pipeline
- randomize spawn positions and headings on reset
- log episode reward, survival time, and win rate explicitly
- add vectorized environments for faster training
- track missile efficiency and gun accuracy
- compare PPO with SAC or TD3
- expand this into a multi-agent dogfight setup

## 💥 Final Pitch

If you want a project that sits right at the intersection of game dev, simulation weirdness, and reinforcement learning nonsense, this is a very solid little war machine.

Which, frankly, is the dream.
