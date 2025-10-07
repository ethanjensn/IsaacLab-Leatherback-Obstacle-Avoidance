# Leatherback Project Tutorial (Isaac Lab External Project)

## Click the images below to watch the tutorial videos:

### Part 1: Setup & Run
[![Part 1: Setup & Run](https://img.youtube.com/vi/bzHtZseHb34/0.jpg)](https://www.youtube.com/watch?v=bzHtZseHb34)

### Part 2.1: Code Explained
[![Part 2.1: Code Explained](https://img.youtube.com/vi/li56b5KVAPc/0.jpg)](https://www.youtube.com/watch?v=li56b5KVAPc)

### Part 2.2: Code Explained
[![Part 2.2: Code Explained](https://img.youtube.com/vi/Y9r7UyWpIAU/0.jpg)](https://www.youtube.com/watch?v=Y9r7UyWpIAU)

*Watch the full playlist: [Leatherback Tutorial Series](https://www.youtube.com/watch?v=bzHtZseHb34&list=PLQQ577DOyRN8jR7xs73WNDy98q05ElidM)*

## Overview

This repository contains the **Leatherback** project, a reinforcement learning environment demonstrating a driving navigation task within the NVIDIA Isaac Lab simulation platform. A four-wheeled vehicle learns to navigate through a series of waypoints.

This project was originally developed by Eric Bowman (StrainFlow) during Study Group Sessions in the [Omniverse Discord Community](https://discord.com/channels/827959428476174346/833873440418431017) and has been adapted into the recommended **external project** structure for Isaac Lab. This structure keeps the project isolated from the main Isaac Lab repository, simplifying updates and development.

**Task:** The robot must learn to drive through a sequence of 10 waypoints using throttle and steering commands.

**Versions:** Isaac Lab 2.0 - Isaac Sim 4.5 ([Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html))

## Contributors

- **Original Project:** [StrainFlow (Eric Bowman)](https://www.linkedin.com/in/strainflow/) & [Antoine Richard](https://github.com/AntoineRichard/)
- **Modified by:** [Pappachuck](https://www.linkedin.com/in/renan-monteiro-barbosa/)
- **Explained & Documented by:** [The Robotics Club](https://www.youtube.com/@madeautonomous) & [LycheeAI](https://www.youtube.com/@LycheeAI)

## Installation

1.  **Install Isaac Lab:** Follow the [official Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). The `conda` installation method is recommended as it simplifies running Python scripts.

2.  **Clone this repository:**
    ```bash
    git clone https://github.com/MuammerBay/Leatherback.git
    cd Leatherback
    ```

3.  **Install the project package:** Activate your Isaac Lab conda environment (e.g., `conda activate isaaclab`) and install this project package in editable mode:
    ```bash
    # On Linux
    python -m pip install -e source/Leatherback

    # On Windows
    python -m pip install -e source\Leatherback
    ```
    *The `-e` flag installs the package in "editable" mode, linking directly to your source code.*

## Usage

*(Ensure your Isaac Lab conda environment is activated before running these commands)*

1.  **List Available Tasks:** Verify that the Leatherback environment is registered.
    ```bash
    # On Linux
    python scripts/list_envs.py

    # On Windows
    python scripts\list_envs.py
    ```
    You should see `Template-Leatherback-Direct-v0` in the output.

2.  **Train the Agent:**
    *   To watch training with a small number of environments:
        ```bash
        # On Linux
        python scripts/skrl/train.py --task Template-Leatherback-Direct-v0 --num_envs 32

        # On Windows
        python scripts\skrl\train.py --task Template-Leatherback-Direct-v0 --num_envs 32
        ```
    *   To accelerate training (more environments, no graphical interface):
        ```bash
        # On Linux
        python scripts/skrl/train.py --task Template-Leatherback-Direct-v0 --num_envs 4096 --headless

        # On Windows
        python scripts\skrl\train.py --task Template-Leatherback-Direct-v0 --num_envs 4096 --headless
        ```
    *   Training logs and checkpoints are saved under the `logs/skrl/leatherback_direct/` directory.

3.  **Play/Evaluate the Agent:**
    *   Run the best-performing policy found during training:
        ```bash
        # On Linux
        python scripts/skrl/play.py --task Template-Leatherback-Direct-v0 --num_envs 32

        # On Windows
        python scripts\skrl\play.py --task Template-Leatherback-Direct-v0 --num_envs 32
        ```
    *   Run a specific checkpoint:
        ```bash
        # Example checkpoint path (replace with your actual path)
        # On Linux
        python scripts/skrl/play.py --task Template-Leatherback-Direct-v0 --checkpoint logs/skrl/leatherback_direct/<YOUR_RUN_DIR>/checkpoints/agent_<STEP>.pt

        # On Windows
        python scripts\skrl\play.py --task Template-Leatherback-Direct-v0 --checkpoint logs\skrl\leatherback_direct\<YOUR_RUN_DIR>\checkpoints\agent_<STEP>.pt
        ```

## Code Structure & Explanation

This project utilizes the **Direct Workflow** provided by Isaac Lab. Key files include:

-   `source/Leatherback/setup.py`: Defines the `Leatherback` Python package for installation.
-   `source/Leatherback/Leatherback/tasks/direct/leatherback/`: Contains the core environment logic.
    -   `__init__.py`: Registers the `Isaac-Leatherback-Direct-v0` Gymnasium environment.
    -   `leatherback_env.py`: Implements the `LeatherbackEnv` class (observation/action spaces, rewards, resets, simulation stepping) and its configuration `LeatherbackEnvCfg`.
    -   `leatherback.py`: Defines the `LEATHERBACK_CFG` articulation configuration (USD path, physics properties, actuators). Located under `isaaclab_assets` in a standard setup, but included here for completeness. *(Self-correction: The actual `leatherback.py` defining the asset config is typically in `isaaclab_assets`. This project likely references it or has a copy)*
    -   `waypoint.py`: Defines the configuration for the waypoint visualization markers.
    -   `agents/skrl_ppo_cfg.yaml`: Configures the PPO reinforcement learning algorithm (network architecture, hyperparameters) for the SKRL library.
    -   `custom_assets/leatherback_simple_better.usd`: The 3D model of the Leatherback robot.

**For a detailed code walkthrough, please refer to the YouTube videos:**

-   **Part 2.1:** [https://youtu.be/li56b5KVAPc](https://youtu.be/li56b5KVAPc)
-   **Part 2.2:** [https://www.youtube.com/watch?v=Y9r7UyWpIAU](https://www.youtube.com/watch?v=Y9r7UyWpIAU)

## Building Your Own Project

This project was created using the Isaac Lab template generator tool. You can create your own external projects or internal tasks using:

```bash
# On Linux
./isaaclab.sh --new

# On Windows
isaaclab.bat --new
```

Refer to the [official Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/main/source/workflows/template/index.html) for more details on building your own projects.

## License

This project uses the BSD-3-Clause license, consistent with Isaac Lab.