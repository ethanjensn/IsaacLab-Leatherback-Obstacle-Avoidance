"""
Isaac Sim Script Editor version for Leatherback waypoint following with LiDAR data collection.
Mirrors the exact waypoint and obstacle randomization from Isaac Lab RL training.

USAGE: Copy/paste this entire script into Isaac Sim Script Editor and click Run.
"""

import omni.usd
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, VisualSphere
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.articulations import Articulation
import numpy as np
import omni.kit.app
import asyncio
import torch
import torch.nn as nn

# ---------------------------------------------------------
# Configuration (matching Isaac Lab environment)
# ---------------------------------------------------------
ROBOT_USD_PATH = r"C:\IsaacLab\leatherback_car\Leatherback\leatherback - Copy.usd"
POLICY_PATH = r"C:\IsaacLab\logs\skrl\leatherback_direct\2025-10-02_11-12-47_ppo_torch\checkpoints\best_agent.pt"

NUM_WAYPOINTS = 10
ENV_SPACING = 32
COURSE_LENGTH_COEFF = 2.5
COURSE_WIDTH_COEFF = 2.0
GOAL_TOLERANCE = 0.15

# Obstacle config (matching Isaac Lab)
NUM_OBSTACLES = 2
OBSTACLE_WIDTH_RANGE = (0.5, 2.0)
OBSTACLE_HEIGHT_RANGE = (1.0, 2.0)
OBSTACLE_DEPTH_RANGE = (0.2, 0.6)

# RL Policy config (matching the checkpoint that was trained)
OBS_DIM = 12  # Checkpoint was trained with 12 obs (before lidar was added)
ACTION_DIM = 6  # From leatherback_env.py action_space

# ---------------------------------------------------------
# Neural Network Policy (matching skrl PPO architecture)
# ---------------------------------------------------------
class PolicyNetwork(nn.Module):
    """Policy network matching skrl's PPO architecture with shared net + policy head."""
    def __init__(self, obs_dim, action_dim, hidden_sizes=[128, 128]):
        super().__init__()
        # Shared feature extractor (net_container)
        layers = []
        in_dim = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ELU())
            in_dim = hidden_size
        self.net_container = nn.Sequential(*layers)
        
        # Policy head (outputs mean actions)
        self.policy_layer = nn.Linear(in_dim, action_dim)
    
    def forward(self, obs):
        features = self.net_container(obs)
        return self.policy_layer(features)

# ---------------------------------------------------------
# Global variables for Script Editor
# ---------------------------------------------------------
world = None
robot = None
waypoints = None
waypoint_markers = []
obstacles = []
current_waypoint_idx = 0
episode_count = 0
step_count = 0
policy = None
use_policy = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Robot state for observations
throttle_state = np.array([0.0])
steering_state = np.array([0.0])
shock_targets = np.array([-0.030, -0.030, 0.030, 0.030], dtype=np.float32)

# Joint indices (will be set after robot initialization)
throttle_joint_indices = None
steering_joint_indices = None
shock_joint_indices = None

# ---------------------------------------------------------
# Policy loading and inference
# ---------------------------------------------------------
def load_policy():
    """Load the trained PPO policy from checkpoint."""
    global policy
    try:
        print(f"Loading policy from: {POLICY_PATH}")
        checkpoint = torch.load(POLICY_PATH, map_location=device)
        
        # Debug: Show checkpoint structure
        print(f"  Checkpoint keys: {list(checkpoint.keys())}")
        
        # skrl saves policy under 'policy' key
        if 'policy' in checkpoint:
            policy_state = checkpoint['policy']
            print(f"  Loading from 'policy' key...")
        else:
            policy_state = checkpoint
            print(f"  Loading from root...")
        
        # Create policy network
        policy = PolicyNetwork(OBS_DIM, ACTION_DIM).to(device)
        
        # Load only the policy-related weights (ignore value_layer and log_std_parameter)
        policy_dict = policy.state_dict()
        filtered_state = {k: v for k, v in policy_state.items() 
                         if k in policy_dict and not k.startswith('value_layer') and k != 'log_std_parameter'}
        
        print(f"  Filtered state dict keys: {list(filtered_state.keys())}")
        policy.load_state_dict(filtered_state, strict=True)
        
        policy.eval()
        print(f"✓ Policy loaded successfully! Using device: {device}")
        return True
    except Exception as e:
        print(f"✗ Failed to load policy!")
        print(f"  Error: {e}")
        import traceback
        print(f"  Traceback:")
        traceback.print_exc()
        print("  Simulation will not run without a valid policy.")
        return False

def get_observation(robot_pos, robot_quat, robot_lin_vel, robot_ang_vel, target_pos, 
                    throttle_state, steering_state, shock_targets):
    """Construct observation matching the checkpoint training (12 dims - no lidar)."""
    # Position error
    position_error_vec = target_pos - robot_pos[:2]
    position_error = np.linalg.norm(position_error_vec)
    
    # Heading error
    w, x, y, z = robot_quat[3], robot_quat[0], robot_quat[1], robot_quat[2]
    robot_yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    target_angle = np.arctan2(position_error_vec[1], position_error_vec[0])
    heading_error = np.arctan2(np.sin(target_angle - robot_yaw), np.cos(target_angle - robot_yaw))
    
    # Observation vector (12 dims - trained before lidar was added)
    obs = np.array([
        position_error,                  # 0: distance to waypoint
        np.cos(heading_error),           # 1: cos(heading error)
        np.sin(heading_error),           # 2: sin(heading error)
        robot_lin_vel[0],                # 3: forward velocity
        robot_lin_vel[1],                # 4: lateral velocity
        robot_ang_vel[2],                # 5: yaw rate
        throttle_state,                  # 6: throttle state
        steering_state,                  # 7: steering state
        shock_targets[0],                # 8: rear right shock
        shock_targets[1],                # 9: rear left shock
        shock_targets[2],                # 10: front right shock
        shock_targets[3],                # 11: front left shock
    ], dtype=np.float32)
    
    return obs

def initialize_joint_indices():
    """Find joint indices for throttle, steering, and shock joints."""
    global throttle_joint_indices, steering_joint_indices, shock_joint_indices
    
    # Joint names from leatherback.py
    throttle_joint_names = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left"
    ]
    steering_joint_names = [
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left"
    ]
    shock_joint_names = [
        "Shock__Rear_Right",
        "Shock__Rear_Left",
        "Shock__Front_Right",
        "Shock__Front_Left"
    ]
    
    # Get all joint names from robot
    all_joint_names = robot.dof_names
    
    # Find indices
    throttle_joint_indices = [all_joint_names.index(name) for name in throttle_joint_names if name in all_joint_names]
    steering_joint_indices = [all_joint_names.index(name) for name in steering_joint_names if name in all_joint_names]
    shock_joint_indices = [all_joint_names.index(name) for name in shock_joint_names if name in all_joint_names]
    
    print(f"Found joint indices:")
    print(f"  Throttle (wheels): {throttle_joint_indices}")
    print(f"  Steering: {steering_joint_indices}")
    print(f"  Shocks: {shock_joint_indices}")
    
    return len(throttle_joint_indices) == 4 and len(steering_joint_indices) == 2 and len(shock_joint_indices) == 4

# ---------------------------------------------------------
# Waypoint generation (matching Isaac Lab _reset_idx)
# ---------------------------------------------------------
def generate_waypoints():
    """Generate waypoints exactly like Isaac Lab RL environment."""
    spacing = 2.0 / NUM_WAYPOINTS
    # X positions: along course direction
    x_positions = np.arange(-0.8, 1.1, spacing, dtype=np.float64) * ENV_SPACING / COURSE_LENGTH_COEFF
    # Y positions: random across course width
    y_positions = np.random.rand(NUM_WAYPOINTS).astype(np.float64) + COURSE_LENGTH_COEFF
    # Z positions: ground level
    z_positions = np.zeros(NUM_WAYPOINTS, dtype=np.float64)
    
    waypoints = np.stack([x_positions, y_positions, z_positions], axis=1)
    print(f"Generated {NUM_WAYPOINTS} waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"  WP{i}: [{wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}]")
    return waypoints

# ---------------------------------------------------------
# Obstacle generation (matching Isaac Lab)
# ---------------------------------------------------------
def spawn_obstacles():
    """Spawn obstacles with same randomization as Isaac Lab."""
    obstacles = []
    for i in range(NUM_OBSTACLES):
        # Randomize size (matching Isaac Lab ranges) - convert to float explicitly
        width = float(np.random.uniform(*OBSTACLE_WIDTH_RANGE))
        height = float(np.random.uniform(*OBSTACLE_HEIGHT_RANGE))
        depth = float(np.random.uniform(*OBSTACLE_DEPTH_RANGE))
        
        # Position along course (matching Isaac Lab reset logic)
        x_pos = float(np.random.uniform(-0.8, 1.1) * ENV_SPACING / COURSE_LENGTH_COEFF)
        y_pos = float(np.random.uniform(0, 1) * COURSE_WIDTH_COEFF * 2 + 1.0)
        z_pos = 0.55  # At LiDAR height
        
        # Create obstacle with uniform scale=1.0, then set size after creation
        obstacle = world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/Obstacle_{i}",
                name=f"Obstacle_{i}",
                position=np.array([x_pos, y_pos, z_pos]),
                scale=np.array([width, depth, height]),  # Use scale instead of size
                color=np.array([0.0, 1.0, 0.0] if i == 0 else [0.0, 0.0, 1.0]),
            )
        )
        obstacles.append(obstacle)
        print(f"Spawned Obstacle_{i}: scale=[{width:.2f}, {depth:.2f}, {height:.2f}], pos=[{x_pos:.2f}, {y_pos:.2f}, {z_pos:.2f}]")
    
    return obstacles

# ---------------------------------------------------------
# Waypoint visualization
# ---------------------------------------------------------
def create_waypoint_markers(waypoints):
    """Create visual sphere markers for waypoints."""
    markers = []
    for i, wp in enumerate(waypoints):
        marker = world.scene.add(
            VisualSphere(
                prim_path=f"/World/Waypoint_{i}",
                name=f"Waypoint_{i}",
                position=wp,
                radius=0.3,  # 30cm radius spheres
                color=np.array([1.0, 1.0, 0.0]),  # Yellow color
            )
        )
        markers.append(marker)
    print(f"Created {len(markers)} waypoint markers (yellow spheres)")
    return markers

# ---------------------------------------------------------
# Initialize scene (called once at startup)
# ---------------------------------------------------------
def setup_scene():
    """Initialize the world, robot, waypoints, and obstacles."""
    global world, robot, waypoints, waypoint_markers, obstacles, current_waypoint_idx, episode_count, use_policy
    
    print("\n" + "="*60)
    print("Setting up Leatherback waypoint following environment")
    print("="*60 + "\n")
    
    # Load trained policy
    use_policy = load_policy()
    
    # Remove previous simulation objects only (keep world, ground plane, lights)
    from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim
    
    print("Removing previous objects...")
    
    # Remove robot
    if get_prim_at_path("/World/Robot"):
        delete_prim("/World/Robot")
    
    # Remove waypoint markers (up to 20 to be safe)
    for i in range(20):
        waypoint_path = f"/World/Waypoint_{i}"
        if get_prim_at_path(waypoint_path):
            delete_prim(waypoint_path)
    
    # Remove obstacles (up to 10 to be safe)
    for i in range(10):
        obstacle_path = f"/World/Obstacle_{i}"
        if get_prim_at_path(obstacle_path):
            delete_prim(obstacle_path)
    
    print("Previous objects removed.\n")
    
    # Get or create world (don't clear instance to preserve rendering)
    world = World.instance()
    if world is None:
        world = World(stage_units_in_meters=1.0)
    
    stage = omni.usd.get_context().get_stage()
    
    # Add robot
    add_reference_to_stage(ROBOT_USD_PATH, "/World/Robot")
    robot = Articulation(prim_path="/World/Robot")
    world.scene.add(robot)
    
    # Generate waypoints and obstacles
    waypoints = generate_waypoints()
    waypoint_markers = create_waypoint_markers(waypoints)
    obstacles = spawn_obstacles()
    current_waypoint_idx = 0
    episode_count = 0
    
    print("Scene setup complete!\n")

# ---------------------------------------------------------
# Main simulation loop (async for Script Editor)
# ---------------------------------------------------------
async def run_simulation():
    """Main simulation loop - runs asynchronously in Script Editor."""
    global world, robot, waypoints, waypoint_markers, obstacles, current_waypoint_idx, episode_count, step_count
    global throttle_state, steering_state, shock_targets, use_policy
    
    print("\n" + "="*60)
    print("Starting waypoint following simulation")
    if use_policy:
        print("Using trained RL policy for control")
    else:
        print("ERROR: Policy failed to load! Cannot run simulation.")
        print("Please check the policy path and try again.")
        return
    print("="*60 + "\n")
    
    # Initialize physics context and start timeline (REQUIRED before robot.initialize())
    print("Initializing physics and starting simulation...")
    await world.initialize_simulation_context_async()
    await world.play_async()
    
    # Wait a frame for physics to be ready
    await omni.kit.app.get_app().next_update_async()
    
    # Now initialize robot (requires physics context to be ready and timeline playing)
    print("Initializing robot...")
    robot.initialize()
    
    # Find joint indices after robot is initialized
    if not initialize_joint_indices():
        print("ERROR: Failed to find all required joints!")
        return
    
    # Reset world to start simulation
    print("Resetting world...")
    world.reset()
    
    print("\nSimulation running! (Press Stop button in Isaac Sim to end)\n")
    
    step_count = 0
    
    while True:
        world.step(render=False)  # render=False to avoid blocking
        await omni.kit.app.get_app().next_update_async()  # Yield to Isaac Sim's event loop
        
        # Get robot state
        robot_pos, robot_quat = robot.get_world_pose()
        robot_lin_vel = robot.get_linear_velocity()
        robot_ang_vel = robot.get_angular_velocity()
        
        # Convert quaternion to yaw (assuming robot's forward is +X)
        w, x, y, z = robot_quat[3], robot_quat[0], robot_quat[1], robot_quat[2]
        robot_yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
        
        # Get current target waypoint
        target = waypoints[current_waypoint_idx]
        
        # Compute distance to target
        dx = target[0] - robot_pos[0]
        dy = target[1] - robot_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Get observation
        obs = get_observation(robot_pos, robot_quat, robot_lin_vel, robot_ang_vel, target[:2],
                            throttle_state[0], steering_state[0], shock_targets)
        
        # Run policy inference
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            action_tensor = policy(obs_tensor)
            actions = action_tensor.cpu().numpy()[0]
        
        # Update state for next observation
        throttle_state[0] = actions[0]
        steering_state[0] = actions[1]
        shock_targets = actions[2:6]
        
        # Scale and clamp actions (matching Isaac Lab _pre_physics_step)
        # Action scaling from leatherback_env.py
        throttle_scale = 10.0
        throttle_max = 50.0
        steering_scale = 0.1
        steering_max = 0.75
        shock_scale = 0.01
        shock_max = 0.1
        
        # Scale throttle (single action for all 4 wheels)
        throttle_vel = np.clip(actions[0] * throttle_scale, -throttle_max, throttle_max)
        throttle_vels = np.array([throttle_vel] * 4, dtype=np.float32)
        
        # Scale steering (single action for both steering joints)
        steering_pos = np.clip(actions[1] * steering_scale, -steering_max, steering_max)
        steering_positions = np.array([steering_pos] * 2, dtype=np.float32)
        
        # Scale shocks (4 separate actions)
        shock_positions = np.clip(actions[2:6] * shock_scale, -shock_max, shock_max)
        
        # Apply actions to robot joints using Isaac Sim API
        try:
            # Set wheel velocities (throttle) - Isaac Sim uses set_joint_velocities
            robot.set_joint_velocities(throttle_vels, joint_indices=throttle_joint_indices)
            
            # Set steering positions - Isaac Sim uses set_joint_positions  
            robot.set_joint_positions(steering_positions, joint_indices=steering_joint_indices)
            
            # Set shock positions
            robot.set_joint_positions(shock_positions, joint_indices=shock_joint_indices)
        except Exception as e:
            if step_count == 0:
                print(f"Warning: Failed to apply actions: {e}")
                print("  Actions will be computed but not applied to robot")
        
        # Check if waypoint reached
        if distance < GOAL_TOLERANCE:
            print(f"✓ Reached waypoint {current_waypoint_idx} (distance: {distance:.3f}m)")
            current_waypoint_idx += 1
            
            # Check if completed all waypoints
            if current_waypoint_idx >= NUM_WAYPOINTS:
                episode_count += 1
                print(f"\n{'='*60}")
                print(f"Episode {episode_count} completed! Randomizing environment...")
                print(f"{'='*60}\n")
                
                # Reset environment with new randomization
                waypoints = generate_waypoints()
                
                # Remove old waypoint markers and create new ones
                for marker in waypoint_markers:
                    world.scene.remove_object(marker.name)
                waypoint_markers = create_waypoint_markers(waypoints)
                
                # Remove old obstacles and spawn new ones
                for obs in obstacles:
                    world.scene.remove_object(obs.name)
                obstacles = spawn_obstacles()
                
                current_waypoint_idx = 0
                
                # Reset robot position (matching Isaac Lab spawn logic)
                reset_pos = np.array([-ENV_SPACING/2, 
                                     2.0 * np.random.rand() * COURSE_WIDTH_COEFF,
                                     0.05], dtype=np.float64)
                reset_angle = float(np.random.rand() * np.pi / 6.0)
                reset_quat = np.array([np.cos(reset_angle/2), 0, 0, np.sin(reset_angle/2)], dtype=np.float64)
                robot.set_world_pose(reset_pos, reset_quat)
                robot.set_linear_velocity(np.zeros(3, dtype=np.float64))
                robot.set_angular_velocity(np.zeros(3, dtype=np.float64))
        
        # Optional: Print status every 100 steps
        if step_count % 100 == 0:
            print(f"[Policy] Step {step_count}: WP {current_waypoint_idx}/{NUM_WAYPOINTS}, "
                  f"Distance: {distance:.2f}m, Throttle Vel: {throttle_vel:.1f}, Steering Pos: {steering_pos:.3f}")
        
        step_count += 1

# ---------------------------------------------------------
# Entry point for Script Editor
# ---------------------------------------------------------
# Setup the scene first
setup_scene()

# Then run the simulation loop using Isaac Sim's async system
asyncio.ensure_future(run_simulation())

