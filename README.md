# Leatherback RL Environment

A reinforcement learning environment for autonomous vehicle navigation using NVIDIA Isaac Lab. Features a four-wheeled vehicle learning to navigate through waypoints with LiDAR-based obstacle detection.

## Features

- **Waypoint Navigation**: Vehicle learns to traverse 10 sequential waypoints
- **LiDAR Integration**: Multi-mesh ray casting for obstacle detection
- **Obstacle Avoidance**: Dynamic collision detection and avoidance
- **Direct Workflow**: Uses Isaac Lab's direct workflow for efficient simulation
- **RL Training**: PPO-based training with RSL-RL library

## Demo at 2x Speed

<img src="gif 1 fast.gif" width="400"/>
<img src="gif 2 fast.gif" width="400"/>
<img src="gif 3 fast.gif" width="400"/>
<img src="gif 4 fast.gif" width="400"/>

## Requirements

- Isaac Lab 2.0 or later
- Isaac Sim 4.5 or later
- Python 3.8+
- NVIDIA GPU with CUDA support

## Installation

1. **Install Isaac Lab** following the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

2. **Clone and install this project:**
   ```bash
   # Clone the repository
   cd Leatherback
   
   # Activate Isaac Lab environment
   conda activate isaaclab
   
   # Install in editable mode
   python -m pip install -e source/Leatherback
   ```

## Usage

### Training

Train with visualization:
```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Template-Leatherback-Direct-v0 --num_envs 32
```

Train headless (faster):
```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Template-Leatherback-Direct-v0 --num_envs 4096 --headless
```

### Evaluation

Run trained policy:
```bash
python scripts/reinforcement_learning/rsl_rl/play.py --task Template-Leatherback-Direct-v0 --num_envs 32
```

With specific checkpoint:
```bash
python scripts/reinforcement_learning/rsl_rl/play.py --task Template-Leatherback-Direct-v0 --checkpoint logs/rsl_rl/leatherback_direct/<RUN_DIR>/model_<STEP>.pt
```

### Monitoring Training with TensorBoard

RSL-RL automatically logs training metrics to TensorBoard. To monitor training progress:

1. **Start TensorBoard** (in a separate terminal):
   ```bash
   tensorboard --logdir=logs/rsl_rl/leatherback_direct
   ```

2. **View in browser**: Open `http://localhost:6006` (or the URL shown in the terminal)

3. **Key metrics to monitor**:
   - `train/mean_reward`: Average episode reward (main training progress indicator)
   - `train/episode_length`: Average episode length
   - `train/learning_rate`: Current learning rate (if using adaptive schedule)
   - `train/value_loss`: Value function loss
   - `train/policy_loss`: Policy gradient loss

4. **Viewing specific run**: To view a specific training run:
   ```bash
   tensorboard --logdir=logs/rsl_rl/leatherback_direct/<RUN_DIR>
   ```


## Project Structure

```
Leatherback/
├── source/Leatherback/Leatherback/tasks/direct/leatherback/
│   ├── leatherback_env.py          # Main environment implementation
│   ├── leatherback.py              # Robot configuration
│   ├── waypoint.py                 # Waypoint markers
│   ├── agents/
│   │   └── rsl_rl_ppo_cfg.py      # PPO hyperparameters
│   └── custom_assets/
│       └── leatherback_OG.usd
├── scripts/
│   └── list_envs.py               # List available environments
├── setup.py                        # Package setup
└── README.md
```

## Environment Details

- **Action Space**: Continuous (throttle, steering)
- **Observation Space**: 79 dimensions
  - 8 base state (position error, heading, velocities, throttle/steering state)
  - 8 shock proprioception (4 shock positions + 4 shock velocities)
  - 63 LiDAR ray distances (normalized to [0, 1])
- **Reward Function**: Distance to waypoints, collision penalties, velocity rewards
- **Episode Length**: 90 seconds (resets on completion or collision)

## Implicit Learning

The agent learns implicitly from two sensor modalities:

- **LiDAR**: The agent learns to interpret 63 normalized ray distances (360° coverage) to detect and avoid obstacles without explicit obstacle labels.

- **Shocks**: The agent learns to adapt its driving behavior (throttle and steering) based on passive suspension feedback (4 shock positions and 4 shock velocities), learning terrain adaptation and stability control without directly controlling the shocks.

## Configuration

Edit `agents/rsl_rl_ppo_cfg.py` to modify:
- Network architecture (hidden dimensions, activation functions)
- Learning rate and schedule
- Training hyperparameters (PPO clip parameter, entropy coefficient, etc.)
- Rollout settings (steps per environment, mini-batch size)

## License

BSD-3-Clause License (consistent with Isaac Lab)

## Acknowledgments

Built with NVIDIA Isaac Lab and Isaac Sim simulation platforms.

## Training Phases

- Phase 1.0 (8192 envs):  40 second episode, 10 waypoint,  20 meters, 1D planar LiDAR with 60 rays, 360° Horizontal FOV, Obstacles: 2 gaps 1 random wall per episode, penalty on collision, still continues episode. Reward and penalty for vertical movement for shocks. (FLAT)
    - +10 waypoint, -10 collision
    - \rsl_rl\leatherback_direct\2025-10-22_04-25-39\model_4550.pt
    - 95-100
- Phase 1.1 (8192 envs):  40 second episode, 10 waypoint,  20 meters, 1D planar LiDAR with 60 rays, 360° Horizontal FOV, Obstacles: 2 gaps 1 random wall per episode, penalty on collision, still continues episode.
    - 2025-10-28_02-33-13        
- Phase 2.0 (8192 envs):  40 second episode, 5waypoint,  20 meters, 1D planar LiDAR with 60 rays, 360° Horizontal FOV, Obstacles: 2 gaps 1 random wall per episode, penalty on collision, still continues episode.
    - Checkpoint: runs/rsl_rl/leatherback_direct/2025-10-28_18-46-10/model_7350.pt (best reward at step 7350, tracked in git)
    - Final metrics (step 11921):
      - Mean reward: 103.72
      - Episode length: 223.83
      - Value loss: 2.77
      - Entropy: 14.01
      - Training FPS: 38,373
    - 2025-10-28_18-46-10
