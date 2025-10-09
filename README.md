# Leatherback RL Environment

A reinforcement learning environment for autonomous vehicle navigation using NVIDIA Isaac Lab. Features a four-wheeled vehicle learning to navigate through waypoints with LiDAR-based obstacle detection.

## Features

- **Waypoint Navigation**: Vehicle learns to traverse 10 sequential waypoints
- **LiDAR Integration**: Multi-mesh ray casting for obstacle detection
- **Obstacle Avoidance**: Dynamic collision detection and avoidance
- **Direct Workflow**: Uses Isaac Lab's direct workflow for efficient simulation
- **RL Training**: PPO-based training with SKRL library

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
python scripts/skrl/train.py --task Template-Leatherback-Direct-v0 --num_envs 32
```

Train headless (faster):
```bash
python scripts/skrl/train.py --task Template-Leatherback-Direct-v0 --num_envs 4096 --headless
```

### Evaluation

Run trained policy:
```bash
python scripts/skrl/play.py --task Template-Leatherback-Direct-v0 --num_envs 32
```

With specific checkpoint:
```bash
python scripts/skrl/play.py --task Template-Leatherback-Direct-v0 --checkpoint logs/skrl/leatherback_direct/<RUN_DIR>/checkpoints/agent_<STEP>.pt
```

## Project Structure

```
Leatherback/
├── source/Leatherback/Leatherback/tasks/direct/leatherback/
│   ├── leatherback_env.py          # Main environment implementation
│   ├── leatherback.py              # Robot configuration
│   ├── waypoint.py                 # Waypoint markers
│   ├── agents/
│   │   └── skrl_ppo_cfg.yaml      # PPO hyperparameters
│   └── custom_assets/
│       └── leatherback_simple_better.usd
├── scripts/
│   ├── skrl/
│   │   ├── train.py               # Training script
│   │   └── play.py                # Evaluation script
│   └── list_envs.py               # List available environments
├── setup.py                        # Package setup
└── README.md
```

## Environment Details

- **Action Space**: Continuous (throttle, steering)
- **Observation Space**: Vehicle state, waypoint positions, LiDAR data
- **Reward Function**: Distance to waypoints, collision penalties, velocity rewards
- **Episode Length**: Variable (resets on completion or collision)

## Configuration

Edit `agents/skrl_ppo_cfg.yaml` to modify:
- Network architecture
- Learning rate
- Training hyperparameters
- Rollout settings

## Training History

### [32,32] Layers

- **Phase 1**: 20 second episode, implicit suspension, 10 waypoint, about 20 meters
    - `2025-09-30_20-51-01_ppo_torch`
    - REWARD NORMALIZES so 100 is max here

- **Phase 2 (BAD)**: 40 second episode, implicit suspension, 6 waypoints, about 50 meters
    - `2025-10-01_10-40-21_ppo_torch`
    - REWARD NORMALIZES so 70 is max here

- **Phase 2a (GOOD/KEPT)**: 40 second episode, implicit suspension, 7 waypoints, 20 meters
    - `2025-10-01_13-59-46_ppo_torch`
    - REWARD NORMALIZES so 70 is max here

- ~~**Phase 3**: 40 second episode, 35m, 7 waypoints, implicit suspension~~
    - ~~`2025-10-01_16-01-45_ppo_torch`~~
    - ~~REWARD NORMALIZES so 70 is max here~~

- ~~**Phase 3a**: 40 second episode, 35m, 7 waypoints, implicit suspension (New YAML)~~
    - ~~`2025-10-01_18-32-41_ppo_torch`~~
    - ~~REWARD NORMALIZES so 70 is max here~~
    - ~~NEW YAML~~

- **Phase 3b**: 40 second episode, 35m, 10 waypoints, implicit suspension (OG YAML)
    - `2025-10-01_19-20-50_ppo_torch`
    - REWARD NORMALIZES so 100 is max here
    - OG YAML configuration used

- **Phase 3b.1** (continuation of 3b but longer episodes): 80 second episode, 35m, 10 waypoints, implicit suspension
    - `2025-10-01_20-17-01_ppo_torch`
    - REWARD NORMALIZES so 100 is max here

### [128,128] Layers

- **Phase 1**: 20 second episode, implicit suspension, 10 waypoint, about 20 meters
    - `2025-10-01_21-28-34_ppo_torch`
    - REWARD NORMALIZES so 100 is max here

- **Phase 1a** (64 envs continuation): 20 second episode, implicit suspension, 10 waypoint, about 20 meters
    - `2025-10-01_22-23-21_ppo_torch`
    - REWARD NORMALIZES so 100 is max here

- **Phase 2**: 40 second episode, implicit suspension, 7 waypoints, 20 meters
    - REWARD NORMALIZES so 70 is max here

- **Phase 3**: 80 second episode, 35m, 10 waypoints, implicit suspension
    - REWARD NORMALIZES so 100 is max here

### [128,128] Layers with Suspension

- **Phase 1**: 20 second episode, 10 waypoint, about 20 meters
    - `2025-10-01_23-52-12_ppo_torch`
    - REWARD NORMALIZES so 100 is max here

- **Phase 2**: 40 second episode, 7 waypoints, 20 meters
    - `2025-10-02_11-12-47_ppo_torch`
    - REWARD NORMALIZES so 70 is max here

- **Phase 3**: 20 second episode, 10 waypoint, 20 meters, 1D planar LiDAR with 30–60 rays, ±90° FOV, Obstacles: 3–6 randomized per episode, Reset: on collision.
    - REWARD NORMALIZES so 100 is max here

## License

BSD-3-Clause License (consistent with Isaac Lab)

## Acknowledgments

Built with NVIDIA Isaac Lab and Isaac Sim simulation platforms.
