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
    action_space = 6
    observation_space = 13  # Updated to include Lidar data
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
    num_obstacles_per_env = 3
    obstacle_width_range = (0.5, 2.0)  # Width across course (m)
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
        self._shock_targets = torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32)
        self._shock_targets[:, 0:2] = -0.030  # Rear shocks
        self._shock_targets[:, 2:4] = 0.030   # Front shocks
        self._shock_action = torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32)  # Initialize for rewards
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
        self.position_progress_weight = 1.0
        self.heading_coefficient = 0.25
        self.heading_progress_weight = 0.05
        self._target_index = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        
        # Suspension reward parameters (small compared to waypoint rewards)
        self.shock_pos_weight = -0.001
        self.shock_vel_weight = -0.0005
        self.wheel_contact_bonus = 0.01
        self.shock_action_penalty = -0.0001
        
        # Lidar-based reward parameters (minimal like other rewards)
        self.lidar_safe_distance = 2.0  # Safe distance threshold (meters)
        self.lidar_danger_distance = 1.0  # Danger zone threshold (meters)
        self.lidar_safe_reward = 0.01  # Small reward for maintaining safe distance
        self.lidar_danger_penalty = -0.05  # Small penalty for getting too close
        self.lidar_collision_penalty = -0.5  # Moderate penalty for collision

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
        # Initialize obstacle tensors if not already done
        if not hasattr(self, '_obstacle_sizes') or self._obstacle_sizes is None:
            self._obstacle_positions = torch.zeros((self.num_envs, self.cfg.num_obstacles_per_env, 3), device=self.device, dtype=torch.float32)
            self._obstacle_sizes = torch.zeros((self.num_envs, self.cfg.num_obstacles_per_env, 3), device=self.device, dtype=torch.float32)
        
        # Type guard assertions
        assert self._obstacle_positions is not None
        assert self._obstacle_sizes is not None
        
        for env_idx in range(self.num_envs):
            env_origin = self.scene.env_origins[env_idx]
            for obs_idx in range(self.cfg.num_obstacles_per_env):
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
        
        # Shock control - actions 2-5 for 4 shocks
        self._shock_action = actions[:, 2:6] * shock_scale
        self._shock_action = torch.clamp(self._shock_action, -shock_max, shock_max)
        self._shock_targets = self._shock_action

    def _apply_action(self) -> None:
        self.leatherback.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        self.leatherback.set_joint_position_target(self._steering_state, joint_ids=self._steering_dof_idx)
        self.leatherback.set_joint_position_target(self._shock_targets, joint_ids=self._shock_dof_idx)

    def _get_observations(self) -> dict:
        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions - self.leatherback.data.root_pos_w[:, :2]
        # Store previous error first if it exists, otherwise use current error
        if hasattr(self, '_position_error'):
            self._previous_position_error = self._position_error.clone()
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        if not hasattr(self, '_previous_position_error'):
            self._previous_position_error = self._position_error.clone()

        heading = self.leatherback.data.heading_w
        target_heading_w = torch.atan2(
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 1] - self.leatherback.data.root_link_pos_w[:, 1],
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 0] - self.leatherback.data.root_link_pos_w[:, 0],
        )
        self.target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

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
                self._shock_targets[:, 0].unsqueeze(dim=1),  # Rear right shock
                self._shock_targets[:, 1].unsqueeze(dim=1),  # Rear left shock
                self._shock_targets[:, 2].unsqueeze(dim=1),  # Front right shock
                self._shock_targets[:, 3].unsqueeze(dim=1),  # Front left shock
                self.lidar_min_distance.unsqueeze(dim=1),  # Lidar minimum distance
            ),
            dim=-1,
        )
        
        if torch.any(obs.isnan()):
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
            raise ValueError("Observations cannot be NAN")

        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        position_progress_rew = self._previous_position_error - self._position_error
        target_heading_rew = torch.exp(-torch.abs(self.target_heading_error) / self.heading_coefficient)
        goal_reached = self._position_error < self.position_tolerance
        self._target_index = self._target_index + goal_reached
        self.task_completed = self._target_index > (self._num_goals -1)
        self._target_index = self._target_index % self._num_goals

        # Suspension rewards
        # Shock position penalty (penalize large compression) - use actual joint positions
        shock_positions = self.leatherback.data.joint_pos[:, self._shock_dof_idx]
        shock_compression = torch.abs(shock_positions)
        
        # Shock velocity penalty (penalize high shock velocity/oscillation)
        shock_velocities = torch.abs(self.leatherback.data.joint_vel[:, self._shock_dof_idx])
        
        # Check for NaN in joint data (can happen during multi-env sensor updates)
        if torch.any(torch.isnan(shock_positions)) or torch.any(torch.isnan(shock_velocities)):
            # Use zeros for this step if data is corrupted
            shock_positions = torch.where(torch.isnan(shock_positions), torch.zeros_like(shock_positions), shock_positions)
            shock_velocities = torch.where(torch.isnan(shock_velocities), torch.zeros_like(shock_velocities), shock_velocities)
            shock_compression = torch.abs(shock_positions)
        
        R_shock_pos = self.shock_pos_weight * torch.sum(shock_compression, dim=1)
        R_shock_vel = self.shock_vel_weight * torch.sum(shock_velocities, dim=1)
        
        # Wheel contact bonus (simplified - check if wheels are close to ground level)
        # This is a rough approximation - for proper contact detection, you'd need PhysX contact data
        root_height = self.leatherback.data.root_pos_w[:, 2:3]  # Shape: (num_envs, 1)
        wheel_heights = root_height + shock_positions  # Shape: (num_envs, 4)
        wheels_in_contact = torch.sum(wheel_heights < 0.1, dim=1)  # Within 10cm of ground
        R_wheel_contact = self.wheel_contact_bonus * wheels_in_contact
        
        # Shock action penalty (penalize large actuator commands)
        R_shock_act = self.shock_action_penalty * torch.sum(torch.abs(self._shock_action), dim=1)
        
        # Obstacle collision penalty
        obstacle_collision = self._check_obstacle_collisions()
        R_collision = -20.0 * obstacle_collision.float()  # Penalty for hitting obstacles (2 waypoints worth)
        
        # Lidar-based rewards for obstacle avoidance
        # Use the minimum lidar distance from observations
        min_lidar_distance = self.lidar_min_distance  # Shape: (num_envs,)
        
        # Safe distance reward - encourage maintaining distance from obstacles
        safe_distance_mask = min_lidar_distance > self.lidar_safe_distance
        R_lidar_safe = self.lidar_safe_reward * safe_distance_mask.float()
        
        # Danger zone penalty - penalize getting too close to obstacles
        danger_zone_mask = (min_lidar_distance < self.lidar_danger_distance) & (min_lidar_distance > 0.1)
        R_lidar_danger = self.lidar_danger_penalty * danger_zone_mask.float()
        
        # Collision penalty - large penalty for actual collision
        collision_mask = min_lidar_distance < 0.1  # Very close = collision
        R_lidar_collision = self.lidar_collision_penalty * collision_mask.float()

        composite_reward = (
            position_progress_rew * self.position_progress_weight +
            target_heading_rew * self.heading_progress_weight +
            goal_reached * self.goal_reached_bonus +
            R_shock_pos +
            R_shock_vel +
            R_wheel_contact +
            R_shock_act +
            R_collision +
            R_lidar_safe +
            R_lidar_danger +
            R_lidar_collision
        )

        one_hot_encoded = torch.nn.functional.one_hot(self._target_index.long(), num_classes=self._num_goals)
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)

        if torch.any(composite_reward.isnan()):
            # print("=" * 80)
            # print("NaN DETECTED IN REWARDS!")
            # print("=" * 80)
            # print(f"Position progress NaN: {torch.any((position_progress_rew * self.position_progress_weight).isnan())}")
            # print(f"Target heading NaN: {torch.any((target_heading_rew * self.heading_progress_weight).isnan())}")
            # print(f"Goal reached NaN: {torch.any((goal_reached * self.goal_reached_bonus).isnan())}")
            # print(f"Shock pos NaN: {torch.any(R_shock_pos.isnan())}")
            # print(f"Shock vel NaN: {torch.any(R_shock_vel.isnan())}")
            # print(f"Wheel contact NaN: {torch.any(R_wheel_contact.isnan())}")
            # print(f"Shock act NaN: {torch.any(R_shock_act.isnan())}")
            # print(f"Collision NaN: {torch.any(R_collision.isnan())}")
            # print(f"Lidar safe NaN: {torch.any(R_lidar_safe.isnan())}")
            # print(f"Lidar danger NaN: {torch.any(R_lidar_danger.isnan())}")
            # print(f"Lidar collision NaN: {torch.any(R_lidar_collision.isnan())}")
            # print(f"\nLidar min distance: {self.lidar_min_distance}")
            # print(f"Lidar min distance NaN mask: {self.lidar_min_distance.isnan()}")
            # print("=" * 80)
            raise ValueError("Rewards cannot be NAN")

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
                            
                            # # Debug output
                            # if torch.any(sensor_collision):
                            #     for env_idx in range(self.num_envs):
                            #         if sensor_collision[env_idx]:
                            #             print(f"[COLLISION DETECTED] Env {env_idx}: {sensor_name} hit obstacle! Force: {max_forces[env_idx]:.1f}N")
        
        return collision_detected

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_failed = self.episode_length_buf > self.max_episode_length
        obstacle_collision = self._check_obstacle_collisions()
        return task_failed | obstacle_collision, self.task_completed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.leatherback._ALL_INDICES
        super()._reset_idx(env_ids)

        # At this point env_ids is guaranteed to be Sequence[int]
        assert env_ids is not None
        num_reset = len(env_ids)
        default_state = self.leatherback.data.default_root_state[env_ids]
        leatherback_pose = default_state[:, :7]
        leatherback_velocities = default_state[:, 7:]
        joint_positions = self.leatherback.data.default_joint_pos[env_ids]
        joint_velocities = self.leatherback.data.default_joint_vel[env_ids]

        leatherback_pose[:, :3] += self.scene.env_origins[env_ids]
        leatherback_pose[:, 0] -= self.env_spacing / 2
        leatherback_pose[:, 1] += 2.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device) * self.course_width_coefficient

        angles = torch.pi / 6.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device)
        leatherback_pose[:, 3] = torch.cos(angles * 0.5)
        leatherback_pose[:, 6] = torch.sin(angles * 0.5)

        self.leatherback.write_root_pose_to_sim(leatherback_pose, env_ids)
        self.leatherback.write_root_velocity_to_sim(leatherback_velocities, env_ids)
        self.leatherback.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)
        
        # Create and initialize rigid prim views on first reset (after simulation starts)
        if not hasattr(self, '_prims_initialized'):
            from isaacsim.core.prims import RigidPrim
            # Create views for all obstacles using regex patterns
            self._obstacle_0_view = RigidPrim("/World/envs/env_.*/TestObstacle_0", reset_xform_properties=False)
            self._obstacle_1_view = RigidPrim("/World/envs/env_.*/TestObstacle_1", reset_xform_properties=False)
            # Wall view commented out - no test walls
            # self._wall_view = RigidPrim("/World/envs/env_.*/TestWall", reset_xform_properties=False)
            # Initialize views
            self._obstacle_0_view.initialize()
            self._obstacle_1_view.initialize()
            # self._wall_view.initialize()
            self._prims_initialized = True
            # print(f"[DEBUG] Initialized obstacle views (2 obstacle views, wall view disabled)")
        
        # Reset contact sensors after episode reset
        self._reset_contact_sensors(env_ids)
        
        # # Debug: Check sensors once at startup
        # if len(env_ids) > 0 and env_ids[0] == 0 and not hasattr(self, '_sensors_debugged'):
        #     self._debug_contact_sensors()
        #     self._debug_robot_bodies()
        #     self._sensors_debugged = True

        self._target_positions[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, :] = 0.0

        spacing = 2 / self._num_goals
        target_positions = torch.arange(-0.8, 1.1, spacing, device=self.device) * self.env_spacing / self.course_length_coefficient
        self._target_positions[env_ids, :len(target_positions), 0] = target_positions
        self._target_positions[env_ids, :, 1] = torch.rand((num_reset, self._num_goals), dtype=torch.float32, device=self.device) + self.course_length_coefficient
        self._target_positions[env_ids, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

        self._target_index[env_ids] = 0
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]
        visualize_pos = self._markers_pos.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)
        
        # Reset obstacle positions AFTER prims are initialized
        if hasattr(self, '_prims_initialized'):
            self._reset_obstacle_positions(env_ids)
        
        # Enable lidar visualization after first reset (when sensors are properly initialized)
        if not hasattr(self, '_lidar_vis_enabled'):
            self.lidar.set_debug_vis(True)
            self._lidar_vis_enabled = True

        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions[:, :2] - self.leatherback.data.root_pos_w[:, :2]
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()

        heading = self.leatherback.data.heading_w[:]
        target_heading_w = torch.atan2( 
            self._target_positions[:, 0, 1] - self.leatherback.data.root_pos_w[:, 1],
            self._target_positions[:, 0, 0] - self.leatherback.data.root_pos_w[:, 0],
        )
        self._heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        self._previous_heading_error = self._heading_error.clone()
        
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
        """Create obstacles only for the source environment (env_0) at scene setup time."""
        # print(f"[DEBUG] Creating obstacles for source environment only (will be cloned)...")
        
        if self._obstacle_sizes is None:
            self._obstacle_sizes = torch.zeros((self.num_envs, 2, 3), device=self.device, dtype=torch.float32)
        if self._obstacle_positions is None:
            self._obstacle_positions = torch.zeros((self.num_envs, 2, 3), device=self.device, dtype=torch.float32)
        
        # Only create obstacles for env_0 (source environment)
        env_idx = 0
        for obs_idx in range(2):
            prim_path = f"/World/envs/env_{env_idx}/TestObstacle_{obs_idx}"
            width = torch.rand(1, device=self.device) * (self.cfg.obstacle_width_range[1] - self.cfg.obstacle_width_range[0]) + self.cfg.obstacle_width_range[0]
            height = torch.rand(1, device=self.device) * (self.cfg.obstacle_height_range[1] - self.cfg.obstacle_height_range[0]) + self.cfg.obstacle_height_range[0]
            depth = torch.rand(1, device=self.device) * (self.cfg.obstacle_depth_range[1] - self.cfg.obstacle_depth_range[0]) + self.cfg.obstacle_depth_range[0]
            self._obstacle_sizes[env_idx, obs_idx] = torch.tensor([width.item(), depth.item(), height.item()], device=self.device)
            obstacle_cfg = CuboidCfg(
                size=(width.item(), depth.item(), height.item()),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True, max_linear_velocity=0.0, max_angular_velocity=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.02, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0) if obs_idx == 0 else (0.0, 0.0, 1.0)),
            )
            obstacle_cfg.func(prim_path, obstacle_cfg, translation=(0.0, 0.0, 0.55))
        
        # print(f"[DEBUG] Created 2 obstacles for source environment (test walls disabled)")
    
    def _reset_obstacle_positions(self, env_ids: Sequence[int]):
        """Reset obstacle positions for given environments by moving them using RigidPrimView."""
        # Type guard assertions
        assert self._obstacle_positions is not None
        assert self._obstacle_sizes is not None
        
        if env_ids is None:
            env_ids = self.leatherback._ALL_INDICES
        
        num_reset = len(env_ids)
        
        # Reset obstacle positions to zero for resetting environments
        self._obstacle_positions[env_ids, :, :] = 0.0
        
        # Position obstacles along the course
        for obs_idx in range(2):  # 2 obstacles per environment
            # X position: random position along course
            x_position = torch.rand((num_reset,), device=self.device) * 1.9 - 0.8
            x_position = x_position * self.env_spacing / self.course_length_coefficient
            self._obstacle_positions[env_ids, obs_idx, 0] = x_position
            
            # Y position: random across course width
            y_position = torch.rand((num_reset,), device=self.device) * self.course_width_coefficient * 2 + 1.0
            self._obstacle_positions[env_ids, obs_idx, 1] = y_position
            
            # Z position: at Lidar height (0.55m above ground)
            self._obstacle_positions[env_ids, obs_idx, 2] = 0.55
        
        # Add environment origins
        self._obstacle_positions[env_ids, :, :2] += self.scene.env_origins[env_ids, :2].unsqueeze(1)
        
        # Quaternion for 90 degree rotation around Z (to make obstacles horizontal)
        # Format: (w, x, y, z) - create as float32 tensor directly
        quat_90z = torch.tensor([0.7071, 0.0, 0.0, 0.7071], dtype=torch.float32, device=self.device)
        
        # Convert env_ids to tensor if it's not already
        if not isinstance(env_ids, torch.Tensor):
            env_ids_tensor = torch.tensor(list(env_ids), dtype=torch.int32, device=self.device)
        else:
            env_ids_tensor = env_ids
        
        # Move obstacles using RigidPrimView - set_world_poses expects tensors
        # Obstacle 0
        obs0_positions = self._obstacle_positions[env_ids, 0, :]  # Shape: (num_reset, 3)
        obs0_orientations = quat_90z.unsqueeze(0).repeat(num_reset, 1)  # Shape: (num_reset, 4)
        self._obstacle_0_view.set_world_poses(obs0_positions, obs0_orientations, indices=env_ids_tensor)
        
        # Obstacle 1
        obs1_positions = self._obstacle_positions[env_ids, 1, :]  # Shape: (num_reset, 3)
        obs1_orientations = quat_90z.unsqueeze(0).repeat(num_reset, 1)  # Shape: (num_reset, 4)
        self._obstacle_1_view.set_world_poses(obs1_positions, obs1_orientations, indices=env_ids_tensor)
        
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

    def _reset_contact_sensors(self, env_ids: Sequence[int]):
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




