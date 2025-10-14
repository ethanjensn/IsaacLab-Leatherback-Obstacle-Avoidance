from __future__ import annotations

import torch
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCaster, MultiMeshRayCasterCfg, patterns
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG
from .waypoint import WAYPOINT_CFG
from .leatherback import LEATHERBACK_CFG
from isaaclab.markers import VisualizationMarkers

@configclass
class LeatherbackSceneCfg(InteractiveSceneCfg):
    """Configuration for the Leatherback environment scene."""
    
    # Contact sensors for obstacle detection - specific sensors for each wheel and chassis
    # Filter out ground plane to only detect obstacles and robot parts
    contact_chassis = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Rigid_Bodies/Chassis",  # Chassis sensor
        update_period=0.0,  # Update every physics step
        history_length=1,  # Store latest contact data
        debug_vis=False,  # Disable visualization to reduce overhead
        track_pose=False,
        track_contact_points=False,
        track_air_time=False,
        force_threshold=0.1,  # Low threshold to detect all contacts
        filter_prim_paths_expr=["/World/ground"],  # Exclude ground plane from contact detection
    )
    
    contact_wheel_front_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Rigid_Bodies/Wheel_Front_Left",  # Front left wheel
        update_period=0.0,  # Update every physics step
        history_length=1,  # Store latest contact data
        debug_vis=False,  # Disable visualization to reduce overhead
        track_pose=False,
        track_contact_points=False,
        track_air_time=False,
        force_threshold=0.1,  # Low threshold to detect all contacts
        filter_prim_paths_expr=["/World/ground"],  # Exclude ground plane from contact detection
    )
    
    contact_wheel_front_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Rigid_Bodies/Wheel_Front_Right",  # Front right wheel
        update_period=0.0,  # Update every physics step
        history_length=1,  # Store latest contact data
        debug_vis=False,  # Disable visualization to reduce overhead
        track_pose=False,
        track_contact_points=False,
        track_air_time=False,
        force_threshold=0.1,  # Low threshold to detect all contacts
        filter_prim_paths_expr=["/World/ground"],  # Exclude ground plane from contact detection
    )
    
    contact_wheel_rear_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Rigid_Bodies/Wheel_Rear_Right",  # Rear right wheel
        update_period=0.0,  # Update every physics step
        history_length=1,  # Store latest contact data
        debug_vis=False,  # Disable visualization to reduce overhead
        track_pose=False,
        track_contact_points=False,
        track_air_time=False,
        force_threshold=0.1,  # Low threshold to detect all contacts
        filter_prim_paths_expr=["/World/ground"],  # Exclude ground plane from contact detection
    )
    
    contact_wheel_rear_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Rigid_Bodies/Wheel_Rear_Left",  # Rear left wheel
        update_period=0.0,  # Update every physics step
        history_length=1,  # Store latest contact data
        debug_vis=False,  # Disable visualization to reduce overhead
        track_pose=False,
        track_contact_points=False,
        track_air_time=False,
        force_threshold=0.1,  # Low threshold to detect all contacts
        filter_prim_paths_expr=["/World/ground"],  # Exclude ground plane from contact detection
    )
    
    # Lidar sensor attached to chassis using Isaac Lab's MultiMeshRayCaster
    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Rigid_Bodies/Chassis",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.3)),  # 0.3m up from chassis
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,  # Single horizontal plane
            vertical_fov_range=(0.0, 0.0),  # 0 degrees vertical FOV
            horizontal_fov_range=(0.0, 360.0),  # Full 360 degrees
            horizontal_res=2.0,  # 2 degree resolution (fewer rays, cleaner visualization)
        ),
        max_distance=20.0,  # 20m maximum range
        debug_vis=False,  # Disabled initially, enabled after first reset
        visualizer_cfg=BLUE_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/LidarRayCaster",  # Global visualization path
            markers={
                "hit": sim_utils.SphereCfg(
                    radius=0.15,  # Medium-sized red hit points
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red color
                ),
                # Only show hit points, no ray lines to avoid stranded rays
            },
        ),
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                target_prim_expr="/World/ground",
                track_mesh_transforms=False,  # Ground is static
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                target_prim_expr="{ENV_REGEX_NS}/TestObstacle_.*",
                track_mesh_transforms=True,  # Obstacles move during reset
            ),
            # Test wall commented out - no walls to detect
            # MultiMeshRayCasterCfg.RaycastTargetCfg(
            #     target_prim_expr="{ENV_REGEX_NS}/TestWall",
            #     track_mesh_transforms=True,  # Wall moves during reset
            # ),
            # Exclude robot from lidar detection to avoid self-collision
            # MultiMeshRayCasterCfg.RaycastTargetCfg(
            #     target_prim_expr="{ENV_REGEX_NS}/Robot/.*",
            #     track_mesh_transforms=True,
            # ),
        ],
    )

@configclass
class LeatherbackEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 40.0
    action_space = 2  # Only throttle + steering (STAGE 1 - was 6)
    observation_space = 9  # STAGE 1: removed 4 shock observations (was 13)
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)
    robot_cfg: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    waypoint_cfg = WAYPOINT_CFG

    throttle_dof_name = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left"
    ]
    steering_dof_name = [
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left",
    ]
    shock_dof_name = [
        "Shock__Rear_Right",
        "Shock__Rear_Left",
        "Shock__Front_Right",
        "Shock__Front_Left"
    ]

    env_spacing = 32
    scene: LeatherbackSceneCfg = LeatherbackSceneCfg(
        num_envs=4096, 
        env_spacing=env_spacing, 
        replicate_physics=True
    )
    
    # Obstacle configuration
    num_obstacles_per_env = (3, 4)  # Random 3-4 obstacles per environment
    obstacle_width_range = (1.75, 3.0)  # Width across course (m) - minimum 1.75m
    obstacle_height_range = (1.0, 2.0)  # Height (m)
    obstacle_depth_range = (0.2, 0.6)  # Depth along course (m)
    obstacle_cfg: CuboidCfg = CuboidCfg(
        size=(1.0, 0.4, 1.5),  # Default size, will be randomized
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,  # Static obstacles
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.8, 0.2, 0.2),  # Red color
        ),
    )

class LeatherbackEnv(DirectRLEnv):
    cfg: LeatherbackEnvCfg

    def __init__(self, cfg: LeatherbackEnvCfg, render_mode: str | None = None, **kwargs):
        # Initialize obstacle tensors as None - will be created during scene setup
        # Type checkers will see them as torch.Tensor after initialization check
        self._obstacle_positions = None
        self._obstacle_sizes = None
        
        super().__init__(cfg, render_mode, **kwargs)
        
        # Obstacle tensors should have been initialized during scene setup
        # Safety check in case they weren't
        if self._obstacle_positions is None:
            self._obstacle_positions = torch.zeros((self.num_envs, 2, 3), device=self.device, dtype=torch.float32)
        if self._obstacle_sizes is None:
            self._obstacle_sizes = torch.zeros((self.num_envs, 2, 3), device=self.device, dtype=torch.float32)
        
        # Assert for type checker - these are definitely tensors now
        assert self._obstacle_positions is not None
        assert self._obstacle_sizes is not None
        
        self._throttle_dof_idx, _ = self.leatherback.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.leatherback.find_joints(self.cfg.steering_dof_name)
        self._shock_dof_idx, _ = self.leatherback.find_joints(self.cfg.shock_dof_name)
        self._throttle_state = torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32)
        self._steering_state = torch.zeros((self.num_envs,2), device=self.device, dtype=torch.float32)
        # STAGE 1: Shock control disabled
        # self._shock_targets = torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32)
        # self._shock_targets[:, 0:2] = -0.030  # Rear shocks
        # self._shock_targets[:, 2:4] = 0.030   # Front shocks
        # self._shock_action = torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32)  # Initialize for rewards
        self._goal_reached = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self.task_completed = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
        self._num_goals = 10
        self._target_positions = torch.zeros((self.num_envs, self._num_goals, 2), device=self.device, dtype=torch.float32)
        self._markers_pos = torch.zeros((self.num_envs, self._num_goals, 3), device=self.device, dtype=torch.float32)
        self.env_spacing = self.cfg.env_spacing
        self.course_length_coefficient = 2.5
        self.course_width_coefficient = 2.0
        self.position_tolerance = 0.15
        self.goal_reached_bonus = 10.0
        self.course_completion_bonus = 100.0  # NEW: bonus for finishing all waypoints
        self.progress_reward_weight = 1.0     # Reward per meter of progress
        self._target_index = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        
        # Store effective timestep for reward normalization
        # Physics dt * decimation = control timestep
        self.control_dt = self.cfg.sim.dt * self.cfg.decimation
        
        # STAGE 1: Suspension rewards disabled - re-enable for STAGE 2
        # self.shock_pos_weight = -0.001
        # self.shock_vel_weight = -0.0005
        # self.shock_action_penalty = -0.0001
        
        # Lidar-based reward parameters for obstacle avoidance
        self.lidar_danger_distance = 0.5      # Proximity warning threshold (not used when penalty is 0)
        self.lidar_proximity_penalty = 0.0    # Disabled - robot learns gap navigation implicitly from collisions
        self.collision_penalty = -10.0        # Hard penalty for collision
        
        # Obstacle-waypoint separation
        self.min_obstacle_waypoint_distance = 2.0  # Minimum 2.0m between obstacles and waypoints

    def _setup_scene(self):
        # Add Physics Scene for Lidar to work (required by Isaac Sim 5.0.0)
        import omni.kit.commands
        import omni
        stage = omni.usd.get_context().get_stage()
        omni.kit.commands.execute('AddPhysicsSceneCommand', stage=stage, path='/World/PhysicsScene')
        
        # Create a large ground plane without grid
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                size=(5000.0, 5000.0),  # Large ground plane (5km x 5km for 4096 envs at 70m spacing)
                color=(0.2, 0.2, 0.2),  # Dark gray color
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )

        # Setup rest of the scene
        self.leatherback = Articulation(self.cfg.robot_cfg)
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)
        self.lidar = MultiMeshRayCaster(self.cfg.scene.lidar)
        self.object_state = []
        
        # Create obstacles for source environment BEFORE cloning
        self._create_obstacles_for_source_env()
        
        self.scene.clone_environments(copy_from_source=False)
        # Don't filter collisions - obstacles are per-environment and need to collide with robot
        # self.scene.filter_collisions(global_prim_paths=[])  # Disabled - prevents obstacle collisions
        self.scene.articulations["leatherback"] = self.leatherback
        
        # Register sensors with scene AFTER cloning to ensure proper multi-env initialization
        # This prevents race conditions and ensures all environments have valid sensor data
        self.scene.sensors["lidar"] = self.lidar
        
        # Contact sensors and lidar are now configured in the scene configuration
        
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_obstacles(self):
        """Spawn obstacles for all environments."""
        # Initialize obstacle tensors if not already done (max 4 obstacles)
        if not hasattr(self, '_obstacle_sizes') or self._obstacle_sizes is None:
            self._obstacle_positions = torch.zeros((self.num_envs, 4, 3), device=self.device, dtype=torch.float32)
            self._obstacle_sizes = torch.zeros((self.num_envs, 4, 3), device=self.device, dtype=torch.float32)
        
        # Type guard assertions
        assert self._obstacle_positions is not None
        assert self._obstacle_sizes is not None
        
        for env_idx in range(self.num_envs):
            env_origin = self.scene.env_origins[env_idx]
            # Randomly choose 3-5 obstacles for this environment
            num_obstacles = torch.randint(3, 6, (1,), device=self.device).item()
            for obs_idx in range(int(num_obstacles)):
                prim_path = f"/World/envs/env_{env_idx}/Obstacle_{obs_idx}"
                
                # Randomize obstacle size
                width = torch.rand(1, device=self.device) * (self.cfg.obstacle_width_range[1] - self.cfg.obstacle_width_range[0]) + self.cfg.obstacle_width_range[0]
                height = torch.rand(1, device=self.device) * (self.cfg.obstacle_height_range[1] - self.cfg.obstacle_height_range[0]) + self.cfg.obstacle_height_range[0]
                depth = torch.rand(1, device=self.device) * (self.cfg.obstacle_depth_range[1] - self.cfg.obstacle_depth_range[0]) + self.cfg.obstacle_depth_range[0]
                
                # Store sizes for later use
                self._obstacle_sizes[env_idx, obs_idx] = torch.tensor([width.item(), depth.item(), height.item()])
                
                # Create obstacle config with randomized size - EXACTLY like test obstacle
                obstacle_cfg = CuboidCfg(
                    size=(width.item(), depth.item(), height.item()),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,  # Static obstacles
                        disable_gravity=True,    # No gravity for static obstacles
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True,  # Enable collision detection for Lidar
                        contact_offset=0.01,     # Small contact offset for better detection
                        rest_offset=0.0,        # No rest offset
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.8, 0.2, 0.2),  # Red color
                    ),
                )
                
                # Calculate initial position (will be updated during reset)
                # Use environment origin since robot data isn't available yet during initial setup
                env_origin = self.scene.env_origins[env_idx]
                initial_pos = (env_origin[0].item(), env_origin[1].item(), 0.5)  # Position at Lidar height
                
                # Spawn obstacle with initial position - EXACTLY like test obstacle
                obstacle_cfg.func(prim_path, obstacle_cfg, translation=initial_pos)
                # print(f"[DEBUG] Created obstacle {obs_idx} for env {env_idx} with size {width.item():.2f}x{depth.item():.2f}x{height.item():.2f}")


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        throttle_scale = 10
        throttle_max = 50
        steering_scale = 0.1
        steering_max = 0.75
        shock_scale = 0.01
        shock_max = 0.1

        self._throttle_action = actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * throttle_scale
        self.throttle_action = torch.clamp(self._throttle_action, -throttle_max, throttle_max)
        self._throttle_state = self._throttle_action
        
        self._steering_action = actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * steering_scale
        self._steering_action = torch.clamp(self._steering_action, -steering_max, steering_max)
        self._steering_state = self._steering_action
        
        # STAGE 1: Shock control disabled - re-enable for STAGE 2
        # self._shock_action = actions[:, 2:6] * shock_scale
        # self._shock_action = torch.clamp(self._shock_action, -shock_max, shock_max)
        # self._shock_targets = self._shock_action

    def _apply_action(self) -> None:
        self.leatherback.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        self.leatherback.set_joint_position_target(self._steering_state, joint_ids=self._steering_dof_idx)
        # STAGE 1: Shock control disabled
        # self.leatherback.set_joint_position_target(self._shock_targets, joint_ids=self._shock_dof_idx)

    def _get_observations(self) -> dict:
        # Detect environments with corrupted physics data
        pos_nan = torch.isnan(self.leatherback.data.root_pos_w).any(dim=1)
        vel_nan = torch.isnan(self.leatherback.data.root_lin_vel_b).any(dim=1)
        ang_nan = torch.isnan(self.leatherback.data.root_ang_vel_w).any(dim=1)
        
        # Also detect extreme positions (robots that have "teleported" to infinity)
        # Use much lower threshold - robots should stay within reasonable course bounds
        pos_extreme = torch.any(torch.abs(self.leatherback.data.root_pos_w) > 1000.0, dim=1)
        
        # Detect extreme shock velocities (sign of physics instability)
        shock_velocities_abs = torch.abs(self.leatherback.data.joint_vel[:, self._shock_dof_idx])
        shock_extreme = torch.any(shock_velocities_abs > 100.0, dim=1)  # Shocks shouldn't move > 100 m/s
        
        corrupted_envs = pos_nan | vel_nan | ang_nan | pos_extreme | shock_extreme
        
        if torch.any(corrupted_envs):
            corrupted_env_ids_tensor = torch.where(corrupted_envs)[0]
            corrupted_env_ids = corrupted_env_ids_tensor.cpu().numpy().tolist()
            print(f"[PHYSICS RESET] Resetting {len(corrupted_env_ids)} environments due to corrupted/extreme physics")
            self._reset_idx(corrupted_env_ids)
        
        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions - self.leatherback.data.root_pos_w[:, :2]
        
        # Keep previous distance for progress calculation
        if hasattr(self, '_position_error') and hasattr(self, '_previous_position_error'):
            self._previous_position_error = self._position_error.clone()
        
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        
        # Initialize on first call
        if not hasattr(self, '_previous_position_error'):
            self._previous_position_error = self._position_error.clone()

        heading = self.leatherback.data.heading_w
        target_heading_w = torch.atan2(
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 1] - self.leatherback.data.root_link_pos_w[:, 1],
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 0] - self.leatherback.data.root_link_pos_w[:, 0],
        )
        self.target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        
        # Guard against NaN in heading calculations
        if torch.any(torch.isnan(self.target_heading_error)):
            self.target_heading_error = torch.where(torch.isnan(self.target_heading_error), torch.zeros_like(self.target_heading_error), self.target_heading_error)

        # Get Lidar data from RayCaster sensor
        lidar_data = self.lidar.data.ray_hits_w  # Shape: (num_envs, num_rays, 3) - hit positions
        lidar_distances = torch.norm(lidar_data - self.lidar.data.pos_w.unsqueeze(1), dim=-1)  # Calculate distances
        
        # Handle inf values (when ray doesn't hit anything)
        # Replace inf with max_distance to avoid numerical issues
        lidar_distances = torch.where(
            torch.isinf(lidar_distances),
            torch.full_like(lidar_distances, self.cfg.scene.lidar.max_distance),
            lidar_distances
        )
        
        # Store minimum lidar distance per environment
        self.lidar_min_distance = torch.min(lidar_distances, dim=1)[0]  # Min distance per environment
        
        # # Periodic debug: Print lidar status every 2 seconds (120 steps at 60fps)
        # if not hasattr(self, '_lidar_debug_counter'):
        #     self._lidar_debug_counter = 0
        # self._lidar_debug_counter += 1
        # 
        # if self._lidar_debug_counter % 120 == 0:
        #     # Print lidar statistics for first 2 environments
        #     for env_idx in range(min(2, self.num_envs)):
        #         env_distances = lidar_distances[env_idx]
        #         valid_hits = env_distances < 10.0  # Within max range
        #         num_valid = torch.sum(valid_hits).item()
        #         min_dist = self.lidar_min_distance[env_idx].item()
        #         
        #         print(f"[LIDAR] Env {env_idx}: {num_valid}/180 rays hit obstacles, min distance: {min_dist:.2f}m")
        
        obs = torch.cat(
            (
                self._position_error.unsqueeze(dim=1),
                torch.cos(self.target_heading_error).unsqueeze(dim=1),
                torch.sin(self.target_heading_error).unsqueeze(dim=1),
                self.leatherback.data.root_lin_vel_b[:, 0].unsqueeze(dim=1),
                self.leatherback.data.root_lin_vel_b[:, 1].unsqueeze(dim=1),
                self.leatherback.data.root_ang_vel_w[:, 2].unsqueeze(dim=1),
                self._throttle_state[:, 0].unsqueeze(dim=1),
                self._steering_state[:, 0].unsqueeze(dim=1),
                # STAGE 1: Shock observations disabled (4 values removed)
                # self._shock_targets[:, 0].unsqueeze(dim=1),  # Rear right shock
                # self._shock_targets[:, 1].unsqueeze(dim=1),  # Rear left shock
                # self._shock_targets[:, 2].unsqueeze(dim=1),  # Front right shock
                # self._shock_targets[:, 3].unsqueeze(dim=1),  # Front left shock
                self.lidar_min_distance.unsqueeze(dim=1),  # Lidar minimum distance
            ),
            dim=-1,
        )
        
        # Replace any remaining NaN values with zeros to prevent training crash
        # This allows training to continue while we identify the root cause
        if torch.any(obs.isnan()):
            nan_count = torch.sum(obs.isnan()).item()
            nan_mask = obs.isnan()
            
            # Identify which observation components have NaNs
            if nan_count > 0:
                print(f"[NaN DEBUG] Step {self.common_step_counter}: {nan_count} NaNs in observations")
                print(f"  Position error: {torch.sum(nan_mask[:, 0]).item()}")
                print(f"  Heading cos: {torch.sum(nan_mask[:, 1]).item()}")
                print(f"  Heading sin: {torch.sum(nan_mask[:, 2]).item()}")
                print(f"  Vel X: {torch.sum(nan_mask[:, 3]).item()}")
                print(f"  Vel Y: {torch.sum(nan_mask[:, 4]).item()}")
                print(f"  Ang vel: {torch.sum(nan_mask[:, 5]).item()}")
                print(f"  Throttle: {torch.sum(nan_mask[:, 6]).item()}")
                print(f"  Steering: {torch.sum(nan_mask[:, 7]).item()}")
                print(f"  Shocks: {torch.sum(nan_mask[:, 8:12]).item()}")
                print(f"  Lidar: {torch.sum(nan_mask[:, 12]).item()}")
            
            obs = torch.where(nan_mask, torch.zeros_like(obs), obs)
            
            # Original debug code (commented out)
            # print("=" * 80)
            # print("NaN DETECTED IN OBSERVATIONS!")
            # print("=" * 80)
            # print(f"Position error NaN: {torch.any(self._position_error.isnan())}")
            # print(f"Target heading error NaN: {torch.any(self.target_heading_error.isnan())}")
            # print(f"Root lin vel x NaN: {torch.any(self.leatherback.data.root_lin_vel_b[:, 0].isnan())}")
            # print(f"Root lin vel y NaN: {torch.any(self.leatherback.data.root_lin_vel_b[:, 1].isnan())}")
            # print(f"Root ang vel z NaN: {torch.any(self.leatherback.data.root_ang_vel_w[:, 2].isnan())}")
            # print(f"Throttle state NaN: {torch.any(self._throttle_state.isnan())}")
            # print(f"Steering state NaN: {torch.any(self._steering_state.isnan())}")
            # print(f"\nRoot position: {self.leatherback.data.root_pos_w[obs.isnan().any(dim=1)]}")
            # print(f"Root velocity: {self.leatherback.data.root_lin_vel_b[obs.isnan().any(dim=1)]}")
            # print(f"Heading: {self.leatherback.data.heading_w[obs.isnan().any(dim=1)]}")
            # print("=" * 80)
            # raise ValueError("Observations cannot be NAN")

        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        # Check for extreme shock velocities before computing rewards
        shock_velocities_abs = torch.abs(self.leatherback.data.joint_vel[:, self._shock_dof_idx])
        if torch.any(shock_velocities_abs > 100.0):
            extreme_shock_envs = torch.any(shock_velocities_abs > 100.0, dim=1)
            extreme_shock_ids = torch.where(extreme_shock_envs)[0].cpu().numpy().tolist()
            print(f"[PHYSICS RESET IN REWARDS] Resetting {len(extreme_shock_ids)} environments due to extreme shock velocities")
            self._reset_idx(extreme_shock_ids)
        
        # One-way progress: only reward decreases in distance, ignore increases (detours)
        distance_change = self._previous_position_error - self._position_error
        R_progress = self.progress_reward_weight * torch.clamp(distance_change, min=0.0) * self.control_dt
        
        # Waypoint reached bonus
        goal_reached = self._position_error < self.position_tolerance
        R_waypoint = goal_reached.float() * self.goal_reached_bonus
        
        # Course completion bonus (all 10 waypoints)
        next_target_index = self._target_index + goal_reached
        course_complete = next_target_index >= self._num_goals
        R_completion = course_complete.float() * self.course_completion_bonus
        
        # Collision penalty (hard)
        obstacle_collision = self._check_obstacle_collisions()
        R_collision = self.collision_penalty * obstacle_collision.float()
        
        # Proximity warning (smooth, scaled with distance)
        # Penalty increases as robot gets closer: 0 at danger_distance, -0.5 at collision
        min_dist = self.lidar_min_distance
        proximity_mask = min_dist < self.lidar_danger_distance
        distance_ratio = torch.clamp(min_dist / self.lidar_danger_distance, min=0.0, max=1.0)
        R_proximity = torch.where(
            proximity_mask,
            self.lidar_proximity_penalty * (1.0 - distance_ratio) * self.control_dt,  # Smooth scaling
            torch.zeros_like(min_dist)
        )
        
        composite_reward = R_progress + R_waypoint + R_completion + R_collision + R_proximity
        
        # Update waypoint tracking
        self._target_index = self._target_index + goal_reached
        self.task_completed = self._target_index >= self._num_goals
        self._target_index = torch.clamp(self._target_index, max=self._num_goals - 1)
        
        # Visualization
        one_hot_encoded = torch.nn.functional.one_hot(self._target_index.long(), num_classes=self._num_goals)
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)
        
        # Replace any NaN rewards with zeros to prevent training crash
        if torch.any(composite_reward.isnan()):
            nan_count = torch.sum(composite_reward.isnan()).item()
            print(f"[NaN DEBUG] Step {self.common_step_counter}: {nan_count} NaNs in rewards")
            print(f"  Progress NaN: {torch.sum(R_progress.isnan()).item()}")
            print(f"  Waypoint NaN: {torch.sum(R_waypoint.isnan()).item()}")
            print(f"  Completion NaN: {torch.sum(R_completion.isnan()).item()}")
            print(f"  Collision NaN: {torch.sum(R_collision.isnan()).item()}")
            print(f"  Proximity NaN: {torch.sum(R_proximity.isnan()).item()}")
            composite_reward = torch.where(torch.isnan(composite_reward), torch.zeros_like(composite_reward), composite_reward)

        return composite_reward

    def _check_obstacle_collisions(self):
        """Check if the robot has collided with any obstacles using all contact sensors."""
        collision_detected = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # Check all contact sensors (all 4 wheels + chassis)
        sensor_names = ['contact_chassis', 'contact_wheel_front_left', 'contact_wheel_front_right', 'contact_wheel_rear_right', 'contact_wheel_rear_left']
        
        if hasattr(self.scene, 'sensors'):
            for sensor_name in sensor_names:
                if sensor_name in self.scene.sensors:
                    contact_sensor = self.scene.sensors[sensor_name]
                    
                    if hasattr(contact_sensor, 'data') and hasattr(contact_sensor.data, 'net_forces_w'):
                        contact_data = contact_sensor.data
                        
                        if contact_data.net_forces_w is not None:
                            # Check for NaN in contact forces (can happen during initialization)
                            forces = contact_data.net_forces_w
                            if torch.any(torch.isnan(forces)):
                                # Skip this sensor if data is invalid
                                continue
                            
                            # Get force magnitude
                            force_magnitudes = torch.norm(forces, dim=-1)
                            max_forces = torch.max(force_magnitudes, dim=1)[0]
                            
                            # Different thresholds for different sensors
                            if sensor_name == 'contact_chassis':
                                # Chassis should never touch anything - low threshold
                                threshold = 5.0
                            else:
                                # Wheels touch ground normally (~10-15N), obstacles will be much higher
                                threshold = 30.0
                            
                            sensor_collision = max_forces > threshold
                            collision_detected = collision_detected | sensor_collision
                            
                            # Debug output - print collisions for first environment only
                            if torch.any(sensor_collision):
                                if sensor_collision[0]:  # Only print env 0
                                    print(f"[COLLISION] Env 0: {sensor_name} hit obstacle! Force: {max_forces[0]:.1f}N")
        
        return collision_detected

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_failed = self.episode_length_buf > self.max_episode_length
        # Don't terminate on collision - let robot recover and continue learning
        # Collision penalty (-10.0) is still applied in _get_rewards()
        return task_failed, self.task_completed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.leatherback._ALL_INDICES
        super()._reset_idx(env_ids)

        # At this point env_ids is guaranteed to be Sequence[int]
        assert env_ids is not None
        
        # Convert to tensor if it's a list (needed for PhysX API calls)
        if isinstance(env_ids, list):
            env_ids_tensor = torch.tensor(env_ids, dtype=torch.int32, device=self.device)
        else:
            # env_ids is already a tensor (from leatherback._ALL_INDICES or torch.where)
            env_ids_tensor = env_ids  # type: ignore[assignment]
        
        num_reset = len(env_ids_tensor)
        default_state = self.leatherback.data.default_root_state[env_ids_tensor]
        leatherback_pose = default_state[:, :7]
        leatherback_velocities = default_state[:, 7:]
        joint_positions = self.leatherback.data.default_joint_pos[env_ids_tensor]
        joint_velocities = self.leatherback.data.default_joint_vel[env_ids_tensor]

        leatherback_pose[:, :3] += self.scene.env_origins[env_ids_tensor]
        leatherback_pose[:, 0] -= self.env_spacing / 2
        leatherback_pose[:, 1] += 2.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device) * self.course_width_coefficient

        angles = torch.pi / 6.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device)
        leatherback_pose[:, 3] = torch.cos(angles * 0.5)
        leatherback_pose[:, 6] = torch.sin(angles * 0.5)

        self.leatherback.write_root_pose_to_sim(leatherback_pose, env_ids_tensor)
        self.leatherback.write_root_velocity_to_sim(leatherback_velocities, env_ids_tensor)
        self.leatherback.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids_tensor)
        
        # Create and initialize rigid prim views on first reset (after simulation starts)
        if not hasattr(self, '_prims_initialized'):
            from isaacsim.core.prims import RigidPrim
            # Create views for all 4 obstacles (all created in source env)
            self._obstacle_views = []
            for i in range(4):
                view = RigidPrim(f"/World/envs/env_.*/TestObstacle_{i}", reset_xform_properties=False)
                view.initialize()
                self._obstacle_views.append(view)
            self._prims_initialized = True
            # print(f"[DEBUG] Initialized obstacle views (5 obstacle views)")
        
        # Reset contact sensors after episode reset
        self._reset_contact_sensors(env_ids_tensor)
        
        # # Debug: Check sensors once at startup
        # if len(env_ids_tensor) > 0 and env_ids_tensor[0] == 0 and not hasattr(self, '_sensors_debugged'):
        #     self._debug_contact_sensors()
        #     self._debug_robot_bodies()
        #     self._sensors_debugged = True

        self._target_positions[env_ids_tensor, :, :] = 0.0
        self._markers_pos[env_ids_tensor, :, :] = 0.0

        spacing = 2 / self._num_goals
        target_positions = torch.arange(-0.8, 1.1, spacing, device=self.device) * self.env_spacing / self.course_length_coefficient
        
        # Add X randomization to make waypoints less predictable
        x_positions = target_positions.unsqueeze(0) + torch.rand((num_reset, self._num_goals), device=self.device) * 2.0 - 1.0
        self._target_positions[env_ids_tensor, :len(target_positions), 0] = x_positions
        
        self._target_positions[env_ids_tensor, :, 1] = torch.rand((num_reset, self._num_goals), dtype=torch.float32, device=self.device) + self.course_length_coefficient
        self._target_positions[env_ids_tensor, :] += self.scene.env_origins[env_ids_tensor, :2].unsqueeze(1)

        self._target_index[env_ids_tensor] = 0
        self._markers_pos[env_ids_tensor, :, :2] = self._target_positions[env_ids_tensor]
        visualize_pos = self._markers_pos.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)
        
        # Reset obstacle positions AFTER prims are initialized
        if hasattr(self, '_prims_initialized'):
            self._reset_obstacle_positions(env_ids_tensor)
        
        # Disable lidar visualization to avoid multi-env instability
        # Enable lidar visualization after first reset (when sensors are properly initialized)
        # if not hasattr(self, '_lidar_vis_enabled'):
        #     self.lidar.set_debug_vis(True)
        #     self._lidar_vis_enabled = True

        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions[:, :2] - self.leatherback.data.root_pos_w[:, :2]
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        
        # Initialize progress tracking
        self._previous_position_error = self._position_error.clone()

        heading = self.leatherback.data.heading_w[:]
        target_heading_w = torch.atan2( 
            self._target_positions[:, 0, 1] - self.leatherback.data.root_pos_w[:, 1],
            self._target_positions[:, 0, 0] - self.leatherback.data.root_pos_w[:, 0],
        )
        self._heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        
        # Contact sensor is now managed by the scene

    def _randomize_obstacle_positions(self, env_ids: Sequence[int] | None):
        """Randomize obstacle positions for the given environment IDs."""
        import isaacsim.core.utils.prims as prim_utils
        from pxr import UsdGeom
        
        # Type guard assertions
        assert self._obstacle_positions is not None
        assert self._obstacle_sizes is not None
        
        if env_ids is None:
            env_ids = self.leatherback._ALL_INDICES
        
        # At this point env_ids is guaranteed to be Sequence[int]
        assert env_ids is not None
        for env_idx in env_ids:
            env_origin = self.scene.env_origins[env_idx]
            
            for obs_idx in range(self.cfg.num_obstacles_per_env):
                prim_path = f"/World/envs/env_{env_idx}/Obstacle_{obs_idx}"
                
                # Position obstacles relative to waypoints
                # Choose a random waypoint to place obstacle near
                waypoint_idx = int(torch.randint(0, self._num_goals, (1,), device=self.device).item())
                waypoint_pos = self._target_positions[env_idx, waypoint_idx]
                
                # Add random offset around the waypoint
                # X: small random offset along course direction
                obs_x_offset = (torch.rand(1, device=self.device) - 0.5) * 2.0  # ±1m along course
                obs_x = waypoint_pos[0] + obs_x_offset
                
                # Y: random offset across course width
                obs_y_offset = (torch.rand(1, device=self.device) - 0.5) * self.course_width_coefficient * 2
                obs_y = waypoint_pos[1] + obs_y_offset
                
                # Z: position at Lidar height for better detection (like test obstacle)
                robot_pos = self.leatherback.data.root_pos_w[env_idx]
                obs_height = self._obstacle_sizes[env_idx, obs_idx, 2]
                obs_z = robot_pos[2] + 0.5  # Same height as Lidar (like test obstacle)
                
                # Store position
                self._obstacle_positions[env_idx, obs_idx] = torch.tensor([obs_x.item(), obs_y.item(), obs_z.item()])
                
                # Update obstacle position and rotation in simulation using USD API
                if prim_utils.is_prim_path_valid(prim_path):
                    prim = prim_utils.get_prim_at_path(prim_path)
                    if prim.IsValid():
                        xform = UsdGeom.Xform(prim)
                        # Clear existing transform operations
                        xform.ClearXformOpOrder()
                        # Add translation operation
                        translate_op = xform.AddTranslateOp()
                        translate_op.Set((obs_x.item(), obs_y.item(), obs_z.item()))
                        # Add rotation operation (90 degrees around Z-axis to make obstacles horizontal)
                        rotate_op = xform.AddRotateZOp()
                        rotate_op.Set(90.0)  # 90 degrees rotation
                        
                        # print(f"[DEBUG] Positioned obstacle {obs_idx} at ({obs_x.item():.2f}, {obs_y.item():.2f}, {obs_z.item():.2f})")


    def _create_obstacles_for_source_env(self):
        """Create obstacle templates in source environment - actual obstacles created during reset."""
        # print(f"[DEBUG] Creating obstacle templates for source environment...")
        
        if self._obstacle_sizes is None:
            self._obstacle_sizes = torch.zeros((self.num_envs, 4, 3), device=self.device, dtype=torch.float32)
        if self._obstacle_positions is None:
            self._obstacle_positions = torch.zeros((self.num_envs, 4, 3), device=self.device, dtype=torch.float32)
        
        # Create template obstacles (hidden at origin) - will be positioned during reset
        # Only create obstacles for env_0 (source environment)
        env_idx = 0
        for obs_idx in range(4):  # Create templates for max 4 obstacles
            prim_path = f"/World/envs/env_{env_idx}/TestObstacle_{obs_idx}"
            
            # Randomize obstacle size based on configuration ranges
            width = (torch.rand(1, device=self.device) * (self.cfg.obstacle_width_range[1] - self.cfg.obstacle_width_range[0]) + self.cfg.obstacle_width_range[0]).item()
            height = (torch.rand(1, device=self.device) * (self.cfg.obstacle_height_range[1] - self.cfg.obstacle_height_range[0]) + self.cfg.obstacle_height_range[0]).item()
            depth = (torch.rand(1, device=self.device) * (self.cfg.obstacle_depth_range[1] - self.cfg.obstacle_depth_range[0]) + self.cfg.obstacle_depth_range[0]).item()
            
            obstacle_cfg = CuboidCfg(
                size=(width, depth, height),  # Randomized size from config ranges
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True, max_linear_velocity=0.0, max_angular_velocity=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.02, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),  # Red color
            )
            # Place template at origin - will be repositioned during reset
            obstacle_cfg.func(prim_path, obstacle_cfg, translation=(0.0, 0.0, 0.55))
        
        # print(f"[DEBUG] Created 5 obstacle templates for source environment")
    
    def _reset_obstacle_positions(self, env_ids: torch.Tensor | Sequence[int]):
        """Reset obstacle positions for given environments by moving them using RigidPrimView."""
        # Type guard assertions
        assert self._obstacle_positions is not None
        assert self._obstacle_sizes is not None
        
        if env_ids is None:
            env_ids = self.leatherback._ALL_INDICES
        
        num_reset = len(env_ids)
        
        # Reset obstacle positions to zero for resetting environments
        self._obstacle_positions[env_ids, :, :] = 0.0
        
        # Position obstacles along the course with waypoint and obstacle collision avoidance
        for env_i, env_id in enumerate(env_ids):
            # Randomly choose 3-4 obstacles for this reset
            num_obstacles = torch.randint(3, 5, (1,), device=self.device).item()
            
            # Get waypoints for this environment
            env_waypoints = self._target_positions[env_id, :, :2]  # Shape: (num_goals, 2)
            
            for obs_idx in range(int(num_obstacles)):
                # Try up to 50 times to find a valid position
                valid_position_found = False
                for attempt in range(50):
                    # X position: random position along course
                    x_pos = (torch.rand(1, device=self.device) * 1.9 - 0.8) * self.env_spacing / self.course_length_coefficient
                    
                    # Y position: random across course width (1.2-4.8m, slightly narrower for centering)
                    y_pos = torch.rand(1, device=self.device) * self.course_width_coefficient * 1.8 + 1.2
                    
                    # Create obstacle position tensor
                    obstacle_pos = torch.tensor([x_pos.item(), y_pos.item()], device=self.device)
                    
                    # Check distance to all waypoints
                    distances_to_waypoints = torch.norm(env_waypoints - obstacle_pos.unsqueeze(0), dim=-1)
                    min_waypoint_distance = torch.min(distances_to_waypoints)
                    
                    # Check distance to previously placed obstacles in this environment
                    min_obstacle_distance = float('inf')
                    if obs_idx > 0:
                        for prev_obs_idx in range(obs_idx):
                            prev_obstacle_pos = self._obstacle_positions[env_id, prev_obs_idx, :2]
                            # Only check if this obstacle was actually placed (not at origin)
                            if not torch.allclose(prev_obstacle_pos, torch.tensor([0.0, 0.0], device=self.device), atol=1e-6):
                                distance_to_prev = torch.norm(obstacle_pos - prev_obstacle_pos)
                                min_obstacle_distance = min(min_obstacle_distance, distance_to_prev.item())
                    
                    # If far enough from all waypoints AND other obstacles, use this position
                    if (min_waypoint_distance > self.min_obstacle_waypoint_distance and 
                        min_obstacle_distance > self.min_obstacle_waypoint_distance):
                        valid_position_found = True
                        self._obstacle_positions[env_id, obs_idx, 0] = x_pos.item()
                        self._obstacle_positions[env_id, obs_idx, 1] = y_pos.item()
                        self._obstacle_positions[env_id, obs_idx, 2] = 0.55  # Lidar height
                        break
                
                # If no valid position found after 50 attempts, reduce obstacle count instead
                if not valid_position_found:
                    # Reduce target obstacle count and break out of loop
                    num_obstacles = obs_idx  # Only spawn obstacles that succeeded
                    break
            
            # Move unused obstacles extremely far away (so lidar won't detect them and no env can reach them)
            # With 1024 envs at 32m spacing = ~1km grid, 5000m is safely beyond any environment
            for obs_idx in range(int(num_obstacles), 4):
                # Place unused obstacles 5km away (world coordinates, will be offset by env origin later)
                self._obstacle_positions[env_id, obs_idx, 0] = 5000.0  # 5km away
                self._obstacle_positions[env_id, obs_idx, 1] = 5000.0  # 5km away
                self._obstacle_positions[env_id, obs_idx, 2] = 0.55  # Same height
        
        # Add environment origins to all obstacles
        self._obstacle_positions[env_ids, :, :2] += self.scene.env_origins[env_ids, :2].unsqueeze(1)
        
        # Quaternion for 90 degree rotation around Z (to make obstacles horizontal)
        # Format: (w, x, y, z) - create as float32 tensor directly
        quat_90z = torch.tensor([0.7071, 0.0, 0.0, 0.7071], dtype=torch.float32, device=self.device)
        
        # Convert env_ids to tensor if it's not already
        if not isinstance(env_ids, torch.Tensor):
            env_ids_tensor = torch.tensor(list(env_ids), dtype=torch.int32, device=self.device)
        else:
            env_ids_tensor = env_ids
        
        # Move all obstacles (up to 4) using views
        for obs_idx in range(4):
            if obs_idx < len(self._obstacle_views):
                obs_positions = self._obstacle_positions[env_ids, obs_idx, :]
                obs_orientations = quat_90z.unsqueeze(0).repeat(num_reset, 1)
                self._obstacle_views[obs_idx].set_world_poses(obs_positions, obs_orientations, indices=env_ids_tensor)
        
        # Test walls commented out - no walls to reset
        # robot_positions = self.leatherback.data.root_pos_w[env_ids]  # Shape: (num_reset, 3)
        # wall_positions = robot_positions.clone()
        # wall_positions[:, 0] += 3.0  # 3m in front
        # wall_positions[:, 2] = 1.0   # Ground level (wall is 2m tall)
        # wall_orientations = quat_90z.unsqueeze(0).repeat(num_reset, 1)
        # self._wall_view.set_world_poses(wall_positions, wall_orientations, indices=env_ids_tensor)
        
        # print(f"[OBSTACLES] Reset {num_reset} environments: Obstacles repositioned")

    def _debug_robot_bodies(self):
        """Debug method to list all rigid bodies (simplified, no PhysX API calls)."""
        pass
        # import isaacsim.core.utils.prims as prim_utils
        # import omni
        # 
        # print(f"[DEBUG] === ROBOT BODIES DEBUG ===")
        # 
        # try:
        #     stage = omni.usd.get_context().get_stage()
        #     rigid_bodies_path = f"/World/envs/env_0/Robot/Rigid_Bodies"
        #     rigid_bodies_prim = stage.GetPrimAtPath(rigid_bodies_path)
        #     
        #     if rigid_bodies_prim.IsValid():
        #         print(f"[DEBUG] Found Rigid_Bodies prim at {rigid_bodies_path}")
        #         rigid_body_names = [child.GetName() for child in rigid_bodies_prim.GetChildren()]
        #         print(f"[DEBUG] Rigid bodies: {rigid_body_names}")
        #     else:
        #         print(f"[DEBUG] Rigid_Bodies prim not found at {rigid_bodies_path}")
        # except Exception as e:
        #     print(f"[DEBUG] Error debugging robot bodies: {e}")
        # 
        # print(f"[DEBUG] === END ROBOT BODIES DEBUG ===")

    def _reset_contact_sensors(self, env_ids: torch.Tensor | Sequence[int]):
        """Reset and reinitialize contact sensors after episode reset."""
        if not hasattr(self.scene, 'sensors'):
            return
        
        contact_sensor_names = [
            'contact_chassis',
            'contact_wheel_front_left', 
            'contact_wheel_front_right',
            'contact_wheel_rear_right',
            'contact_wheel_rear_left'
        ]
        
        for sensor_name in contact_sensor_names:
            if sensor_name in self.scene.sensors:
                contact_sensor = self.scene.sensors[sensor_name]
                
                # Reset the sensor's internal state
                if hasattr(contact_sensor, 'reset'):
                    contact_sensor.reset(env_ids)
                
                # Re-enable the sensor if it has an enable method
                if hasattr(contact_sensor, 'enable'):
                    contact_sensor.enable(True)
                
                # Clear any cached data
                if hasattr(contact_sensor, 'data') and contact_sensor.data is not None:
                    # Reset force data to zero
                    if hasattr(contact_sensor.data, 'net_forces_w') and contact_sensor.data.net_forces_w is not None:
                        contact_sensor.data.net_forces_w.zero_()
                
                # print(f"[DEBUG] Reset contact sensor: {sensor_name}")

    def _debug_contact_sensors(self):
        """Debug method to check contact sensor status."""
        pass
        # print(f"[DEBUG] === CONTACT SENSORS DEBUG ===")
        # 
        # if hasattr(self.scene, 'sensors'):
        #     print(f"[DEBUG] Scene sensors created: {list(self.scene.sensors.keys())}")
        #     expected_sensors = ['contact_chassis', 'contact_wheel_front_left', 'contact_wheel_front_right', 'contact_wheel_rear_right', 'contact_wheel_rear_left']
        #     
        #     for sensor_name in expected_sensors:
        #         if sensor_name in self.scene.sensors:
        #             contact_sensor = self.scene.sensors[sensor_name]
        #             print(f"[DEBUG] ✓ {sensor_name} successfully created by scene")
        #             print(f"[DEBUG]   - Initialized: {hasattr(contact_sensor, 'is_initialized') and contact_sensor.is_initialized}")
        #             print(f"[DEBUG]   - Has data: {hasattr(contact_sensor, 'data') and contact_sensor.data is not None}")
        #             
        #             if hasattr(contact_sensor, 'data') and hasattr(contact_sensor.data, 'net_forces_w'):
        #                 forces = contact_sensor.data.net_forces_w
        #                 print(f"[DEBUG]   - Forces shape: {forces.shape if forces is not None else 'None'}")
        #                 if forces is not None:
        #                     print(f"[DEBUG]   - Max force: {torch.max(torch.norm(forces, dim=-1)).item():.6f}N")
        #         else:
        #             print(f"[DEBUG] ✗ {sensor_name} NOT created by scene!")
        # else:
        #     print(f"[DEBUG] WARNING: Scene has no sensors attribute!")
        # 
        # print(f"[DEBUG] === END CONTACT SENSORS DEBUG ===")




