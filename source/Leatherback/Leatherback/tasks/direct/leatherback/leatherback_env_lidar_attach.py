from __future__ import annotations

import torch
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.utils import configclass
from .waypoint import WAYPOINT_CFG
from .leatherback import LEATHERBACK_CFG
from isaaclab.markers import VisualizationMarkers
from isaacsim.sensors.physx import _range_sensor

@configclass
class LeatherbackEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 20.0
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)
    
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
    
    # Lidar configuration
    lidar_enabled = True
    lidar_min_range = 0.1  # 10cm minimum range
    lidar_max_range = 50.0  # 50m maximum range
    lidar_horizontal_fov = 360.0  # 360 degrees horizontal FOV
    lidar_vertical_fov = 30.0  # 30 degrees vertical FOV
    lidar_horizontal_resolution = 0.4  # degrees
    lidar_vertical_resolution = 4.0  # degrees
    lidar_rotation_rate = 0.0  # Static (no rotation)
    lidar_high_lod = True
    lidar_draw_lines = True
    lidar_draw_points = False

class LeatherbackEnv(DirectRLEnv):
    cfg: LeatherbackEnvCfg

    def __init__(self, cfg: LeatherbackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize obstacle tracking attributes after super().__init__()
        self._obstacle_positions = torch.zeros((self.num_envs, self.cfg.num_obstacles_per_env, 3), device=self.device, dtype=torch.float32)
        self._obstacle_sizes = torch.zeros((self.num_envs, self.cfg.num_obstacles_per_env, 3), device=self.device, dtype=torch.float32)
        
        # Initialize Lidar sensor interface
        self.lidar_interface = None
        if self.cfg.lidar_enabled:
            self.lidar_interface = _range_sensor.acquire_lidar_sensor_interface()
        
        self._throttle_dof_idx, _ = self.leatherback.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.leatherback.find_joints(self.cfg.steering_dof_name)
        self._shock_dof_idx, _ = self.leatherback.find_joints(self.cfg.shock_dof_name)
        self._throttle_state = torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32)
        self._steering_state = torch.zeros((self.num_envs,2), device=self.device, dtype=torch.float32)
        self._shock_targets = torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32)
        self._shock_targets[:, 0:2] = -0.030  # Rear shocks
        self._shock_targets[:, 2:4] = 0.030   # Front shocks
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

    def _setup_scene(self):
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
        self.object_state = []
        
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["leatherback"] = self.leatherback
        
        # Create Lidar sensors for each environment (after scene setup)
        # if self.cfg.lidar_enabled:
        #     self._create_lidar_sensors()

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # Spawn obstacles for each environment
        self._spawn_obstacles()

    def _spawn_obstacles(self):
        """Spawn obstacles for all environments."""
        # Initialize obstacle tensors if not already done
        if not hasattr(self, '_obstacle_sizes'):
            self._obstacle_positions = torch.zeros((self.num_envs, self.cfg.num_obstacles_per_env, 3), device=self.device, dtype=torch.float32)
            self._obstacle_sizes = torch.zeros((self.num_envs, self.cfg.num_obstacles_per_env, 3), device=self.device, dtype=torch.float32)
        
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
                
                # Create obstacle config with randomized size
                obstacle_cfg = CuboidCfg(
                    size=(width.item(), depth.item(), height.item()),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,  # Static obstacles
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.8, 0.2, 0.2),  # Red color
                    ),
                )
                
                # Spawn obstacle (position will be set during reset)
                obstacle_cfg.func(prim_path, obstacle_cfg, translation=(0.0, 0.0, 0.0))

    def _create_lidar_sensors(self):
        """Create PhysX Lidar sensors for each environment."""
        import omni.kit.commands
        import isaacsim.core.utils.prims as prim_utils
        
        print(f"[DEBUG] Creating Lidar sensors for {self.num_envs} environments")
        
        for env_idx in range(self.num_envs):
            # Find the actual robot root prim path
            robot_root_path = f"/World/envs/env_{env_idx}/Robot"
            
            # Check if robot exists and find its root link
            if not prim_utils.is_prim_path_valid(robot_root_path):
                print(f"[DEBUG] Robot path {robot_root_path} does not exist, skipping Lidar creation")
                continue
            
            # Attach Lidar to the Chassis (main moving body of the robot)
            chassis_path = f"/World/envs/env_{env_idx}/Robot/Rigid_Bodies/Chassis"
            
            # Check if chassis exists
            if not prim_utils.is_prim_path_valid(chassis_path):
                print(f"[DEBUG] Chassis path {chassis_path} does not exist, using robot root")
                parent_path = robot_root_path
            else:
                parent_path = chassis_path
                print(f"[DEBUG] Using Chassis as Lidar parent: {parent_path}")
            lidar_name = "Lidar"
            
            print(f"[DEBUG] Creating Lidar at {parent_path}/{lidar_name}")
            
            # Create Lidar sensor using Isaac Sim command
            result, prim = omni.kit.commands.execute(
                "RangeSensorCreateLidar",
                path=lidar_name,
                parent=parent_path,
                min_range=self.cfg.lidar_min_range,
                max_range=self.cfg.lidar_max_range,
                draw_points=self.cfg.lidar_draw_points,
                draw_lines=self.cfg.lidar_draw_lines,
                horizontal_fov=self.cfg.lidar_horizontal_fov,
                vertical_fov=self.cfg.lidar_vertical_fov,
                horizontal_resolution=self.cfg.lidar_horizontal_resolution,
                vertical_resolution=self.cfg.lidar_vertical_resolution,
                rotation_rate=self.cfg.lidar_rotation_rate,
                high_lod=self.cfg.lidar_high_lod,
                yaw_offset=0.0,
                enable_semantics=False
            )
            
            print(f"[DEBUG] Lidar creation result: {result}")
            
            # Position Lidar 1 foot above robot
            if result:
                from pxr import UsdGeom
                
                # The actual path will be parent_path + "/" + lidar_name
                actual_lidar_path = f"{parent_path}/{lidar_name}"
                print(f"[DEBUG] Actual Lidar path: {actual_lidar_path}")
                
                if prim_utils.is_prim_path_valid(actual_lidar_path):
                    prim = prim_utils.get_prim_at_path(actual_lidar_path)
                    if prim.IsValid():
                        print(f"[DEBUG] Setting Lidar position to (0, 0, 0.3048)")
                        xform = UsdGeom.Xform(prim)
                        xform.ClearXformOpOrder()
                        translate_op = xform.AddTranslateOp()
                        translate_op.Set((0.0, 0.0, 0.3048))  # 1 foot = 0.3048m above robot
                    else:
                        print(f"[DEBUG] Prim at {actual_lidar_path} is not valid")
                else:
                    print(f"[DEBUG] Lidar path {actual_lidar_path} is not valid")
            else:
                print(f"[DEBUG] Lidar creation failed for environment {env_idx}")

    def _create_lidar_sensors_attached(self):
        """Create PhysX Lidar sensors properly attached to robot using a different approach."""
        import omni.kit.commands
        import isaacsim.core.utils.prims as prim_utils
        from pxr import UsdGeom, Gf
        
        print(f"[DEBUG] Creating attached Lidar sensors for {self.num_envs} environments")
        
        for env_idx in range(self.num_envs):
            # Get the robot's current world position
            robot_pos = self.leatherback.data.root_pos_w[env_idx]
            robot_quat = self.leatherback.data.root_quat_w[env_idx]
            
            # Create Lidar at robot's world position + offset
            lidar_world_pos = robot_pos + torch.tensor([0.0, 0.0, 0.3048], device=self.device)
            
            # Create Lidar as a standalone prim at world position
            lidar_path = f"/World/envs/env_{env_idx}/Lidar"
            
            print(f"[DEBUG] Creating standalone Lidar at {lidar_path}")
            
            # Create Lidar sensor using Isaac Sim command
            result, prim = omni.kit.commands.execute(
                "RangeSensorCreateLidar",
                path="Lidar",
                parent=f"/World/envs/env_{env_idx}",
                min_range=self.cfg.lidar_min_range,
                max_range=self.cfg.lidar_max_range,
                draw_points=self.cfg.lidar_draw_points,
                draw_lines=self.cfg.lidar_draw_lines,
                horizontal_fov=self.cfg.lidar_horizontal_fov,
                vertical_fov=self.cfg.lidar_vertical_fov,
                horizontal_resolution=self.cfg.lidar_horizontal_resolution,
                vertical_resolution=self.cfg.lidar_vertical_resolution,
                rotation_rate=self.cfg.lidar_rotation_rate,
                high_lod=self.cfg.lidar_high_lod,
                yaw_offset=0.0,
                enable_semantics=False
            )
            
            if result:
                print(f"[DEBUG] Standalone Lidar created successfully")
                # Position the Lidar at robot's world position + offset
                if prim_utils.is_prim_path_valid(lidar_path):
                    prim = prim_utils.get_prim_at_path(lidar_path)
                    if prim.IsValid():
                        xform = UsdGeom.Xform(prim)
                        xform.ClearXformOpOrder()
                        translate_op = xform.AddTranslateOp()
                        translate_op.Set((lidar_world_pos[0].item(), lidar_world_pos[1].item(), lidar_world_pos[2].item()))
                        print(f"[DEBUG] Lidar positioned at robot location: {lidar_world_pos}")
            else:
                print(f"[DEBUG] Standalone Lidar creation failed for environment {env_idx}")

    def _update_lidar_positions(self):
        """Update Lidar positions to follow the robot."""
        if not self.cfg.lidar_enabled or not hasattr(self, '_lidar_created'):
            return
            
        import isaacsim.core.utils.prims as prim_utils
        from pxr import UsdGeom
        
        for env_idx in range(self.num_envs):
            lidar_path = f"/World/envs/env_{env_idx}/Lidar"
            
            if prim_utils.is_prim_path_valid(lidar_path):
                # Get robot's current world position
                robot_pos = self.leatherback.data.root_pos_w[env_idx]
                robot_quat = self.leatherback.data.root_quat_w[env_idx]
                
                # Calculate Lidar position (robot position + 1 foot up)
                lidar_world_pos = robot_pos + torch.tensor([0.0, 0.0, 0.3048], device=self.device)
                
                # Update Lidar position
                prim = prim_utils.get_prim_at_path(lidar_path)
                if prim.IsValid():
                    xform = UsdGeom.Xform(prim)
                    xform.ClearXformOpOrder()
                    translate_op = xform.AddTranslateOp()
                    translate_op.Set((lidar_world_pos[0].item(), lidar_world_pos[1].item(), lidar_world_pos[2].item()))

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
        
        # Update Lidar positions to follow robot
        self._update_lidar_positions()

    def _apply_action(self) -> None:
        self.leatherback.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        self.leatherback.set_joint_position_target(self._steering_state, joint_ids=self._steering_dof_idx)
        self.leatherback.set_joint_position_target(self._shock_targets, joint_ids=self._shock_dof_idx)

    def _get_observations(self) -> dict:
        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions - self.leatherback.data.root_pos_w[:, :2]
        self._previous_position_error = self._position_error.clone()
        self._position_error = torch.norm(self._position_error_vector, dim=-1)

        heading = self.leatherback.data.heading_w
        target_heading_w = torch.atan2(
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 1] - self.leatherback.data.root_link_pos_w[:, 1],
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 0] - self.leatherback.data.root_link_pos_w[:, 0],
        )
        self.target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

        # Get Lidar data (simplified - use dummy data for now)
        if self.cfg.lidar_enabled and self.lidar_interface is not None:
            # For now, use a simple distance approximation
            # In a full implementation, you'd get data from each Lidar sensor
            lidar_min_distance = torch.full((self.num_envs,), 10.0, device=self.device)  # Default 10m
        else:
            lidar_min_distance = torch.full((self.num_envs,), 10.0, device=self.device)  # Default 10m
        
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
                lidar_min_distance.unsqueeze(dim=1),  # Lidar minimum distance
            ),
            dim=-1,
        )
        
        if torch.any(obs.isnan()):
            print("=" * 80)
            print("NaN DETECTED IN OBSERVATIONS!")
            print("=" * 80)
            print(f"Position error NaN: {torch.any(self._position_error.isnan())}")
            print(f"Target heading error NaN: {torch.any(self.target_heading_error.isnan())}")
            print(f"Root lin vel x NaN: {torch.any(self.leatherback.data.root_lin_vel_b[:, 0].isnan())}")
            print(f"Root lin vel y NaN: {torch.any(self.leatherback.data.root_lin_vel_b[:, 1].isnan())}")
            print(f"Root ang vel z NaN: {torch.any(self.leatherback.data.root_ang_vel_w[:, 2].isnan())}")
            print(f"Throttle state NaN: {torch.any(self._throttle_state.isnan())}")
            print(f"Steering state NaN: {torch.any(self._steering_state.isnan())}")
            print(f"\nRoot position: {self.leatherback.data.root_pos_w[obs.isnan().any(dim=1)]}")
            print(f"Root velocity: {self.leatherback.data.root_lin_vel_b[obs.isnan().any(dim=1)]}")
            print(f"Heading: {self.leatherback.data.heading_w[obs.isnan().any(dim=1)]}")
            print("=" * 80)
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
        R_shock_pos = self.shock_pos_weight * torch.sum(shock_compression, dim=1)
        
        # Shock velocity penalty (penalize high shock velocity/oscillation)
        shock_velocities = torch.abs(self.leatherback.data.joint_vel[:, self._shock_dof_idx])
        R_shock_vel = self.shock_vel_weight * torch.sum(shock_velocities, dim=1)
        
        # Wheel contact bonus (simplified - check if wheels are close to ground level)
        # This is a rough approximation - for proper contact detection, you'd need PhysX contact data
        root_height = self.leatherback.data.root_pos_w[:, 2:3]  # Shape: (num_envs, 1)
        wheel_heights = root_height + shock_positions  # Shape: (num_envs, 4)
        wheels_in_contact = torch.sum(wheel_heights < 0.1, dim=1)  # Within 10cm of ground
        R_wheel_contact = self.wheel_contact_bonus * wheels_in_contact
        
        # Shock action penalty (penalize large actuator commands)
        R_shock_act = self.shock_action_penalty * torch.sum(torch.abs(self._shock_action), dim=1)

        composite_reward = (
            position_progress_rew * self.position_progress_weight +
            target_heading_rew * self.heading_progress_weight +
            goal_reached * self.goal_reached_bonus +
            R_shock_pos +
            R_shock_vel +
            R_wheel_contact +
            R_shock_act
        )

        one_hot_encoded = torch.nn.functional.one_hot(self._target_index.long(), num_classes=self._num_goals)
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)

        if torch.any(composite_reward.isnan()):
            raise ValueError("Rewards cannot be NAN")

        return composite_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_failed = self.episode_length_buf > self.max_episode_length
        return task_failed, self.task_completed

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

        # Create Lidar sensors after robot is positioned
        if self.cfg.lidar_enabled and not hasattr(self, '_lidar_created'):
            self._create_lidar_sensors_attached()
            self._lidar_created = True

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
        
        # Randomize obstacle positions
        self._randomize_obstacle_positions(env_ids)

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

    def _randomize_obstacle_positions(self, env_ids: Sequence[int] | None):
        """Randomize obstacle positions for the given environment IDs."""
        import isaacsim.core.utils.prims as prim_utils
        from pxr import UsdGeom
        
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
                waypoint_idx = torch.randint(0, self._num_goals, (1,), device=self.device).item()
                waypoint_pos = self._target_positions[env_idx, waypoint_idx]
                
                # Add random offset around the waypoint
                # X: small random offset along course direction
                obs_x_offset = (torch.rand(1, device=self.device) - 0.5) * 2.0  # ±1m along course
                obs_x = waypoint_pos[0] + obs_x_offset
                
                # Y: random offset across course width
                obs_y_offset = (torch.rand(1, device=self.device) - 0.5) * self.course_width_coefficient * 2
                obs_y = waypoint_pos[1] + obs_y_offset
                
                # Z: half the obstacle height (so bottom touches ground)
                obs_height = self._obstacle_sizes[env_idx, obs_idx, 2]
                obs_z = obs_height / 2
                
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
