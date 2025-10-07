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

## License

BSD-3-Clause License (consistent with Isaac Lab)

## Acknowledgments

Built with NVIDIA Isaac Lab and Isaac Sim simulation platforms.
