from __future__ import annotations

import torch
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
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
    # DEFERRED: Sensors commented out during init to speed up clone_environments()
    # They will be created lazily after first reset
    # contact_chassis = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/Rigid_Bodies/Chassis",  # Chassis sensor
    #     update_period=0.0167,  # 60Hz update rate
    #     history_length=1,  # Store latest contact data
    #     debug_vis=False,  # Disable visualization to reduce overhead
    #     track_pose=False,
    #     track_contact_points=False,
    #     track_air_time=False,
    #     force_threshold=0.1,  # Low threshold to detect all contacts
    #     filter_prim_paths_expr=[],  # Empty = track all contacts (robust approach)
    # )
    # 
    # contact_wheel_front_left = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/Rigid_Bodies/Wheel_Front_Left",  # Front left wheel
    #     update_period=0.0167,  # 60Hz update rate
    #     history_length=1,  # Store latest contact data
    #     debug_vis=False,  # Disable visualization to reduce overhead
    #     track_pose=False,
    #     track_contact_points=False,
    #     track_air_time=False,
    #     force_threshold=0.1,  # Low threshold to detect all contacts
    #     filter_prim_paths_expr=[],  # Empty = track all contacts (robust approach)
    # )
    # 
    # contact_wheel_front_right = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/Rigid_Bodies/Wheel_Front_Right",  # Front right wheel
    #     update_period=0.0167,  # 60Hz update rate
    #     history_length=1,  # Store latest contact data
    #     debug_vis=False,  # Disable visualization to reduce overhead
    #     track_pose=False,
    #     track_contact_points=False,
    #     track_air_time=False,
    #     force_threshold=0.1,  # Low threshold to detect all contacts
    #     filter_prim_paths_expr=[],  # Empty = track all contacts (robust approach)
    # )
    # 
    # contact_wheel_rear_right = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/Rigid_Bodies/Wheel_Rear_Right",  # Rear right wheel
    #     update_period=0.0167,  # 60Hz update rate
    #     history_length=1,  # Store latest contact data
    #     debug_vis=False,  # Disable visualization to reduce overhead
    #     track_pose=False,
    #     track_contact_points=False,
    #     track_air_time=False,
    #     force_threshold=0.1,  # Low threshold to detect all contacts
    #     filter_prim_paths_expr=[],  # Empty = track all contacts (robust approach)
    # )
    # 
    # contact_wheel_rear_left = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/Rigid_Bodies/Wheel_Rear_Left",  # Rear left wheel
    #     update_period=0.0167,  # 60Hz update rate
    #     history_length=1,  # Store latest contact data
    #     debug_vis=False,  # Disable visualization to reduce overhead
    #     track_pose=False,
    #     track_contact_points=False,
    #     track_air_time=False,
    #     force_threshold=0.1,  # Low threshold to detect all contacts
    #     filter_prim_paths_expr=[],  # Empty = track all contacts (robust approach)
    # )
    
    # Lidar sensor attached to chassis using Isaac Lab's MultiMeshRayCaster
    # DEFERRED: LIDAR commented out during init to speed up clone_environments()
    # It will be created lazily after first reset
    # lidar = MultiMeshRayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/Rigid_Bodies/Chassis",
    #     offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.3)),  # 0.3m up from chassis
    #     pattern_cfg=patterns.LidarPatternCfg(
    #         channels=1,  # Single horizontal plane
    #         vertical_fov_range=(0.0, 0.0),  # 0 degrees vertical FOV
    #         horizontal_fov_range=(0.0, 360.0),  # Full 360 degrees
    #         horizontal_res=5.625,  # 5.625 degree resolution = 63 rays (360° excludes duplicate at 0°)
    #     ),
    #     max_distance=20.0,  # 20m maximum range
    #     update_period=0.0167,  # 60Hz update rate
    #     debug_vis=False,  # Disabled initially, enabled after first reset
    #     visualizer_cfg=BLUE_ARROW_X_MARKER_CFG.replace(
    #         prim_path="/Visuals/LidarRayCaster",  # Global visualization path
    #         markers={
    #             "hit": sim_utils.SphereCfg(
    #                 radius=0.15,  # Medium-sized red hit points
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red color
    #             ),
    #             # Only show hit points, no ray lines to avoid stranded rays
    #         },
    #     ),
    #     mesh_prim_paths=[
    #         MultiMeshRayCasterCfg.RaycastTargetCfg(
    #             target_prim_expr="/World/ground",
    #             track_mesh_transforms=True,  # Ground is static
    #         ),
    #         MultiMeshRayCasterCfg.RaycastTargetCfg(
    #             target_prim_expr="{ENV_REGEX_NS}/TestObstacle_.*",
    #             track_mesh_transforms=True,  # Obstacles only move during reset, not during simulation
    #         ),
    #         # Test wall commented out - no walls to detect
    #         # MultiMeshRayCasterCfg.RaycastTargetCfg(
    #         #     target_prim_expr="{ENV_REGEX_NS}/TestWall",
    #         #     track_mesh_transforms=True,  # Wall moves during reset
    #         # ),
    #         # Exclude robot from lidar detection to avoid self-collision
    #         # MultiMeshRayCasterCfg.RaycastTargetCfg(
    #         #     target_prim_expr="{ENV_REGEX_NS}/Robot/.*",
    #         #     track_mesh_transforms=True,
    #         # ),
    #     ],
    # )

@configclass
class LeatherbackEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 90.0
    action_space = 2  # Only throttle + steering (passive shocks)
    observation_space = 79  # 8 base + 8 shock (4 pos + 4 vel) + 63 lidar (360/5.625 excludes duplicate at 360°)
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60, 
        render_interval=decimation,
        device="cuda:0",  # Explicitly use GPU for PhysX simulation
        use_fabric=True,  # Use Fabric for efficient GPU physics data access (required for GPU sim)
        physx=PhysxCfg(
            # Increase GPU buffers for 16K+ environments with many collision objects
            gpu_max_rigid_contact_count=2**24,  # Increased from 2^23 (8M -> 16M contacts)
            gpu_max_rigid_patch_count=2**22,    # Increased to 4.2M patches (2^22) to fix overflow with 4096+ envs
            gpu_found_lost_pairs_capacity=2**22,  # Increased from 2^21 (2M -> 4M pairs)
            gpu_heap_capacity=2**27,  # Increased from 2^26 for more collision data
            gpu_temp_buffer_capacity=2**25,  # Increased from 2^24 for temporary collision data
        ),
    )
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

    env_spacing = 40  # Increased to 40m to prevent environment overlap with walls (4m) + waypoints
    scene: LeatherbackSceneCfg = LeatherbackSceneCfg(
        num_envs=4096, 
        env_spacing=env_spacing, 
        replicate_physics=True,
        lazy_sensor_update=True,  # Only update sensors when accessed (default but explicit)
        # clone_in_fabric=True not used - incompatible with contact sensors in Isaac Sim 5.0
    )
    
    # Two-gap navigation configuration (4 walls total)
    num_obstacles_per_env = 5  # 4 walls for 2 gaps + 1 random wall
    wall_length = 4.0  # Wall length perpendicular to path (m) - longer to prevent going around
    wall_depth = 0.25  # Wall thickness along path (m)
    wall_height = 1.75  # Wall height (m)
    gap_size_range = (1.0, 1.4)  # Gap between walls (m) - robot fits with clearance
    obstacle_cfg: CuboidCfg = CuboidCfg(
        size=(wall_length, wall_depth, wall_height),  # Wall dimensions
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,  # Static walls
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
        
        # Lazy sensor initialization flags for faster startup
        self._contact_sensors_initialized = False
        self._lidar_initialized = False
        
        # Store LIDAR config values since cfg.scene.lidar is commented out
        self._lidar_max_distance = 20.0  # Same as LIDAR config
        
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
        self._goal_reached = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self.task_completed = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
        self._num_goals = 10
        self._target_positions = torch.zeros((self.num_envs, self._num_goals, 2), device=self.device, dtype=torch.float32)
        self._markers_pos = torch.zeros((self.num_envs, self._num_goals, 3), device=self.device, dtype=torch.float32)
        self.env_spacing = self.cfg.env_spacing
        self.course_length_coefficient = 2.5
        self.course_width_coefficient = 2.0
        self.position_tolerance = 0.15
        self.goal_reached_bonus = 15.0  # Increased from 10.0 for better waypoint navigation
        self.position_progress_weight = 2.0  # Increased from 1.0 for stronger progress rewards
        self.heading_coefficient = 0.25
        self.heading_progress_weight = 0.1  # Increased from 0.05 for better heading alignment
        self._target_index = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        
        # Store effective timestep for reward normalization
        # Physics dt * decimation = control timestep
        self.control_dt = self.cfg.sim.dt * self.cfg.decimation
        
        # Lidar-based reward parameters for obstacle avoidance
        self.lidar_danger_distance = 0.5      # Proximity warning threshold
        self.lidar_proximity_penalty = 0.0    # Disabled - waypoints too close, would discourage navigation
        self.collision_penalty = -5.0        # Hard penalty for collision
        
        # Baja passive suspension parameters - calculate spring/damper from robot mass
        # Measure robot mass after first simulation step (mass not available yet)
        self._suspension_configured = False
        self.spring_stiffness_per_wheel = 0.0
        self.damping_per_wheel = 0.0
        
        # Shock detection and recovery tracking for reward system
        self.shock_vertical_accel_threshold = 5.0 * 9.81  # 5g in m/s² - detects aggressive terrain driving (was 3g)
        self.recovery_window = 2.0  # seconds
        self.shock_penalty = -0.2  # Gentle penalty - was too strong at -2.0 (caused -800 reward)
        self.recovery_bonus = 0.1  # Gentle bonus - was too strong at 1.0
        self.shock_detected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.shock_timer = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.recovering = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_vertical_vel = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        
        # Shock activity monitoring
        self._shock_debug_counter = 0

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
                size=(10000.0, 10000.0),  # Large enough for 32K+ envs at 40m spacing (181x181 grid = 7240m + buffer)
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
        # DEFERRED: LIDAR creation commented out to speed up initialization
        # self.lidar = MultiMeshRayCaster(self.cfg.scene.lidar)
        self.lidar = None  # Will be created lazily after first reset
        self.object_state = []
        
        # Create obstacles in source environment (env_0) BEFORE cloning
        self._create_obstacles_for_source_env()
        
        # Clone environments (copy_from_source=True means obstacles ARE copied to all envs)
        self.scene.clone_environments(copy_from_source=True)
        # Don't filter collisions - obstacles are per-environment and need to collide with robot
        # self.scene.filter_collisions(global_prim_paths=[])  # Disabled - prevents obstacle collisions
        self.scene.articulations["leatherback"] = self.leatherback
        
        # DEFERRED: Sensor registration commented out - sensors will be created/registered after first reset
        # Register sensors with scene AFTER cloning to ensure proper multi-env initialization
        # This prevents race conditions and ensures all environments have valid sensor data
        # self.scene.sensors["lidar"] = self.lidar
        
        # Contact sensors and lidar are now configured in the scene configuration
        
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # Initialize rigid prim views for obstacles DURING scene setup (not first reset)
        # This prevents initialization overhead during first environment reset
        from isaacsim.core.prims import RigidPrim
        
        self._obstacle_views = []
        for i in range(5):
            view = RigidPrim(f"/World/envs/env_.*/TestObstacle_{i}", reset_xform_properties=False)
            self._obstacle_views.append(view)
        self._prims_initialized = True
        print(f"[SCENE SETUP] Initialized 5 wall views for obstacle navigation")

    def _initialize_contact_sensors(self):
        """Create and initialize contact sensors after first reset to speed up initialization."""
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        
        print("[LAZY INIT] Creating contact sensors after first reset...")
        
        # Define contact sensor configs
        sensor_configs = {
            'contact_chassis': ContactSensorCfg(
                prim_path=f"{self.scene.env_regex_ns}/Robot/Rigid_Bodies/Chassis",
                update_period=0.0167,
                history_length=1,
                debug_vis=False,
                track_pose=False,
                track_contact_points=False,
                track_air_time=False,
                force_threshold=0.1,
                filter_prim_paths_expr=["/World/ground"],  # Exclude ground - only detect obstacle collisions
            ),
            'contact_wheel_front_left': ContactSensorCfg(
                prim_path=f"{self.scene.env_regex_ns}/Robot/Rigid_Bodies/Wheel_Front_Left",
                update_period=0.0167,
                history_length=1,
                debug_vis=False,
                track_pose=False,
                track_contact_points=False,
                track_air_time=False,
                force_threshold=0.1,
                filter_prim_paths_expr=["/World/ground"],  # Exclude ground - only detect obstacle collisions
            ),
            'contact_wheel_front_right': ContactSensorCfg(
                prim_path=f"{self.scene.env_regex_ns}/Robot/Rigid_Bodies/Wheel_Front_Right",
                update_period=0.0167,
                history_length=1,
                debug_vis=False,
                track_pose=False,
                track_contact_points=False,
                track_air_time=False,
                force_threshold=0.1,
                filter_prim_paths_expr=["/World/ground"],  # Exclude ground - only detect obstacle collisions
            ),
            'contact_wheel_rear_right': ContactSensorCfg(
                prim_path=f"{self.scene.env_regex_ns}/Robot/Rigid_Bodies/Wheel_Rear_Right",
                update_period=0.0167,
                history_length=1,
                debug_vis=False,
                track_pose=False,
                track_contact_points=False,
                track_air_time=False,
                force_threshold=0.1,
                filter_prim_paths_expr=["/World/ground"],  # Exclude ground - only detect obstacle collisions
            ),
            'contact_wheel_rear_left': ContactSensorCfg(
                prim_path=f"{self.scene.env_regex_ns}/Robot/Rigid_Bodies/Wheel_Rear_Left",
                update_period=0.0167,
                history_length=1,
                debug_vis=False,
                track_pose=False,
                track_contact_points=False,
                track_air_time=False,
                force_threshold=0.1,
                filter_prim_paths_expr=["/World/ground"],  # Exclude ground - only detect obstacle collisions
            ),
        }
        
        # Create and register sensors
        for sensor_name, sensor_cfg in sensor_configs.items():
            sensor = ContactSensor(sensor_cfg)
            # Initialize sensor internal state (required for _timestamp, etc.)
            if not sensor.is_initialized:
                sensor._initialize_impl()
            self.scene.sensors[sensor_name] = sensor
        
        self._contact_sensors_initialized = True
        print("[LAZY INIT] Contact sensors created and initialized successfully")

    def _initialize_lidar(self):
        """Create and initialize LIDAR sensor after first reset to speed up initialization."""
        from isaaclab.sensors import MultiMeshRayCaster, MultiMeshRayCasterCfg, patterns
        from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG
        
        print("[LAZY INIT] Creating LIDAR sensor after first reset...")
        
        # Create LIDAR config
        lidar_cfg = MultiMeshRayCasterCfg(
            prim_path=f"{self.scene.env_regex_ns}/Robot/Rigid_Bodies/Chassis",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.3)),
            pattern_cfg=patterns.LidarPatternCfg(
                channels=1,
                vertical_fov_range=(0.0, 0.0),
                horizontal_fov_range=(0.0, 360.0),
                horizontal_res=5.625,
            ),
            max_distance=20.0,
            update_period=0.0167,
            debug_vis=False,
            visualizer_cfg=BLUE_ARROW_X_MARKER_CFG.replace(
                prim_path="/Visuals/LidarRayCaster",
                markers={
                    "hit": sim_utils.SphereCfg(
                        radius=0.15,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            ),
            mesh_prim_paths=[
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    target_prim_expr="/World/ground",
                    track_mesh_transforms=True,
                ),
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    target_prim_expr=f"{self.scene.env_regex_ns}/TestObstacle_.*",
                    track_mesh_transforms=True,
                ),
            ],
        )
        
        # Create and register LIDAR
        self.lidar = MultiMeshRayCaster(lidar_cfg)
        # Initialize LIDAR internal state (required for _timestamp, etc.)
        if not self.lidar.is_initialized:
            self.lidar._initialize_impl()
        self.scene.sensors["lidar"] = self.lidar
        
        self._lidar_initialized = True
        print("[LAZY INIT] LIDAR sensor created and initialized successfully")

    def _configure_suspension_from_mass(self):
        """Calculate and configure Baja-style passive suspension based on measured robot mass.
        
        Raises:
            RuntimeError: If default_mass data is not available yet (simulation not started)
            ValueError: If measured mass is outside reasonable bounds (1-100 kg)
        """
        import math
        
        # Get total robot mass (sum of all body masses) - NO DEFAULTS ALLOWED
        if not hasattr(self.leatherback.data, 'default_mass'):
            raise RuntimeError("[SUSPENSION] Mass data not ready yet - waiting for simulation to start")
        
        total_mass = torch.sum(self.leatherback.data.default_mass[0]).item()
        
        # Sanity check: robot mass should be reasonable (1-100 kg range for small robots)
        if total_mass < 1.0 or total_mass > 100.0:
            raise ValueError(f"[SUSPENSION] FATAL: Measured robot mass {total_mass:.2f} kg is unrealistic! Check USD file.")
        
        # Baja suspension design parameters
        natural_frequency = 2.5  # Hz (2.0-3.0 typical for small robots)
        damping_ratio = 0.3  # dimensionless (0.2-0.5 typical)
        
        # Calculate total spring stiffness: k_total = m * (2π * f_n)²
        omega_n = 2.0 * math.pi * natural_frequency
        k_total = total_mass * (omega_n ** 2)
        
        # Per-wheel stiffness (4 wheels)
        k_per_wheel = k_total / 4.0
        
        # Per-wheel sprung mass
        m_wheel = total_mass / 4.0
        
        # Critical damping per wheel: c_crit = 2 * sqrt(k * m)
        c_crit = 2.0 * math.sqrt(k_per_wheel * m_wheel)
        
        # Actual damping per wheel
        c_per_wheel = damping_ratio * c_crit
        
        # Store values
        self.spring_stiffness_per_wheel = k_per_wheel
        self.damping_per_wheel = c_per_wheel
        self._suspension_configured = True
        
        # Print summary
        print("="*80)
        print("[SUSPENSION] Baja Passive Suspension Configuration")
        print(f"  Total robot mass: {total_mass:.2f} kg (MEASURED from USD)")
        print(f"  Natural frequency: {natural_frequency:.2f} Hz")
        print(f"  Damping ratio: {damping_ratio:.2f}")
        print(f"  Per-wheel spring stiffness: {k_per_wheel:.2f} N/m ({k_per_wheel/1000:.4f} N/mm)")
        print(f"  Per-wheel damping: {c_per_wheel:.2f} N·s/m")
        print(f"  Total spring stiffness: {k_total:.2f} N/m")
        print(f"  Critical damping per wheel: {c_crit:.2f} N·s/m")
        print("="*80)

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
        
        # Configure suspension after first physics step (when default_mass data is available)
        if not self._suspension_configured:
            try:
                self._configure_suspension_from_mass()
            except (RuntimeError, AttributeError, ValueError):
                # Data not ready yet - will retry on next step
                pass

    def _get_observations(self) -> dict:
        # Initialize contact sensors on first call (after first reset)
        if not self._contact_sensors_initialized:
            self._initialize_contact_sensors()
        
        # Initialize LIDAR on first call (after first reset)
        if not self._lidar_initialized:
            self._initialize_lidar()
        
        # Detect environments with corrupted physics data
        pos_nan = torch.isnan(self.leatherback.data.root_pos_w).any(dim=1)
        vel_nan = torch.isnan(self.leatherback.data.root_lin_vel_b).any(dim=1)
        ang_nan = torch.isnan(self.leatherback.data.root_ang_vel_w).any(dim=1)
        
        # Also detect extreme positions (robots that have "teleported" to infinity)
        # For 16K+ envs at 40m spacing, max legitimate distance is ~3600m from origin
        # Set threshold to 10,000m to only catch actual physics explosions
        pos_extreme = torch.any(torch.abs(self.leatherback.data.root_pos_w) > 10000.0, dim=1)
        
        # Detect extreme shock velocities (sign of physics instability)
        shock_velocities_abs = torch.abs(self.leatherback.data.joint_vel[:, self._shock_dof_idx])
        shock_extreme = torch.any(shock_velocities_abs > 100.0, dim=1)  # Shocks shouldn't move > 100 m/s
        
        corrupted_envs = pos_nan | vel_nan | ang_nan | pos_extreme | shock_extreme
        
        if torch.any(corrupted_envs):
            corrupted_env_ids_tensor = torch.where(corrupted_envs)[0]
            # Optimized: Pass tensor directly, avoid CPU sync
            print(f"[PHYSICS RESET] Resetting {corrupted_env_ids_tensor.shape[0]} environments due to corrupted/extreme physics")
            self._reset_idx(corrupted_env_ids_tensor)
        
        # Debug: Print shock activity every 2 seconds (120 steps at 60Hz)
        # DISABLED FOR PERFORMANCE: Causes CPU sync overhead
        # self._shock_debug_counter += 1
        # 
        # if self._shock_debug_counter % 120 == 0:
        #     env_idx = 0  # Monitor first environment
        #     shock_pos = self.leatherback.data.joint_pos[env_idx, self._shock_dof_idx].cpu().numpy()
        #     shock_vel = self.leatherback.data.joint_vel[env_idx, self._shock_dof_idx].cpu().numpy()
        #     speed = torch.norm(self.leatherback.data.root_lin_vel_b[env_idx, :2]).item()
        #     
        #     print(f"[SHOCK] Env 0 @ {speed:.2f}m/s:")
        #     print(f"  Positions: RR={shock_pos[0]:.4f}, RL={shock_pos[1]:.4f}, FR={shock_pos[2]:.4f}, FL={shock_pos[3]:.4f}")
        #     print(f"  Velocities: RR={shock_vel[0]:.4f}, RL={shock_vel[1]:.4f}, FR={shock_vel[2]:.4f}, FL={shock_vel[3]:.4f}")
        
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

        # Get Lidar data from RayCaster sensor (lazily initialized above)
        assert self.lidar is not None, "LIDAR should be initialized by lazy init above"
        lidar_data = self.lidar.data.ray_hits_w  # Shape: (num_envs, num_rays, 3) - hit positions
        lidar_distances = torch.norm(lidar_data - self.lidar.data.pos_w.unsqueeze(1), dim=-1)  # Shape: (num_envs, 63)
        
        # Handle inf values (when ray doesn't hit anything)
        # Replace inf with max_distance to avoid numerical issues
        lidar_distances = torch.where(
            torch.isinf(lidar_distances),
            torch.full_like(lidar_distances, self._lidar_max_distance),
            lidar_distances
        )
        
        # Normalize lidar distances to [0, 1] range for better learning
        lidar_normalized = lidar_distances / self._lidar_max_distance  # Shape: (num_envs, 63)
        
        # Store minimum lidar distance for debugging/rewards
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
        #         print(f"[LIDAR] Env {env_idx}: {num_valid}/63 rays hit obstacles, min distance: {min_dist:.2f}m")
        
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
                # Shock observations (passive suspension proprioception)
                self.leatherback.data.joint_pos[:, self._shock_dof_idx],  # 4 shock positions
                self.leatherback.data.joint_vel[:, self._shock_dof_idx],  # 4 shock velocities
                lidar_normalized,  # All 63 lidar ray distances (normalized to [0,1])
            ),
            dim=-1,
        )
        
        # Replace any remaining NaN values with zeros to prevent training crash
        # This allows training to continue while we identify the root cause
        if torch.any(obs.isnan()):
            nan_mask = obs.isnan()
            # Optimized: Simplified NaN debug, avoid multiple .item() calls
            print(f"[NaN DEBUG] Step {self.common_step_counter}: NaNs detected in observations - replacing with zeros")
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
        position_progress_rew = self._previous_position_error - self._position_error
        target_heading_rew = torch.exp(-torch.abs(self.target_heading_error) / self.heading_coefficient)
        goal_reached = self._position_error < self.position_tolerance
        
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
        
        # Shock detection and recovery rewards (Baja passive suspension)
        shock_events = self._detect_shock_events()
        R_shock = torch.where(shock_events, torch.tensor(self.shock_penalty, device=self.device), torch.tensor(0.0, device=self.device))
        
        # Update shock timers and recovery state
        self.shock_timer = torch.where(shock_events, torch.zeros_like(self.shock_timer), self.shock_timer + self.control_dt)
        self.recovering = shock_events | (self.recovering & (self.shock_timer < self.recovery_window))
        
        # Detect recovery events
        recovery_events = self._detect_recovery()
        R_recovery = torch.where(recovery_events, torch.tensor(self.recovery_bonus, device=self.device), torch.tensor(0.0, device=self.device))
        
        # Clear recovery flag after bonus awarded
        self.recovering = torch.where(recovery_events, torch.zeros_like(self.recovering, dtype=torch.bool), self.recovering)
        
        # Update previous vertical velocity for next step's shock detection
        self.prev_vertical_vel = self.leatherback.data.root_lin_vel_w[:, 2].clone()
        
        self._target_index = self._target_index + goal_reached
        self.task_completed = self._target_index > (self._num_goals - 1)
        self._target_index = self._target_index % self._num_goals

        composite_reward = (
            position_progress_rew * self.position_progress_weight +
            target_heading_rew * self.heading_progress_weight +
            goal_reached * self.goal_reached_bonus +  # Restored: +15 per waypoint
            R_collision +
            R_proximity +
            R_shock +       # Gentle -0.2 penalty for suspension management
            R_recovery      # Gentle +0.1 bonus for recovery
        )

        one_hot_encoded = torch.nn.functional.one_hot(self._target_index.long(), num_classes=self._num_goals)
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)

        if torch.any(composite_reward.isnan()):
            raise ValueError("Rewards cannot be NAN")

        return composite_reward

    def _check_obstacle_collisions(self):
        """Check if the robot has collided with obstacles (walls, objects) - ground excluded.
        
        Contact sensors filter out ground (/World/ground) to prevent false positives from:
        - Rough terrain navigation (potholes, bumps)
        - Normal ground contact force spikes
        
        Uses velocity-adaptive force thresholds for reliable detection:
        - Chassis: Fixed 5N threshold (should never touch anything)
        - Wheels: Base 100N + 10N per m/s (accounts for speed-dependent forces)
        
        This approach detects:
        - Wall collisions (obstacles like TestObstacle_*)
        - Object collisions (future: other robots, debris)
        - Side-swipes and direct impacts
        
        Terrain impacts (potholes) are handled separately by shock detection system.
        """
        collision_detected = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # Get robot forward velocity magnitude for adaptive thresholds
        velocities = self.leatherback.data.root_lin_vel_b[:, :2]  # XY velocity
        speed = torch.norm(velocities, dim=1)  # Speed per environment
        
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
                            
                            # Velocity-adaptive thresholds to distinguish ground contact from collisions
                            if sensor_name == 'contact_chassis':
                                # Chassis: fixed threshold (should never touch anything)
                                threshold = torch.full_like(speed, 5.0)
                            else:
                                # Wheels: adaptive threshold based on speed
                                # Base threshold at rest: 100N (above normal 80-104N range)
                                # Increase by 10N per m/s of speed
                                base_threshold = 100.0
                                velocity_factor = 10.0  # N per m/s
                                threshold = base_threshold + speed * velocity_factor
                            
                            sensor_collision = max_forces > threshold
                            collision_detected = collision_detected | sensor_collision
                            
                            # Debug output - print collisions for first environment only
                            if torch.any(sensor_collision):
                                if sensor_collision[0]:  # Only print env 0
                                    print(f"[COLLISION] Env 0: {sensor_name} collision! Force: {max_forces[0]:.1f}N (threshold: {threshold[0]:.1f}N, speed: {speed[0]:.1f}m/s)")
        
        return collision_detected

    def _detect_shock_events(self) -> torch.Tensor:
        """Detect shock events based on vertical acceleration.
        
        Uses vertical acceleration as signal (reliable for all terrain types):
        - Detects potholes, jumps, rough terrain, landing impacts
        - Physics-based: measures actual chassis impact
        - Terrain-agnostic: works regardless of surface type
        
        Wall collisions are handled separately by collision detection system.
        
        Returns:
            Boolean tensor indicating which environments experienced a shock this step.
        """
        # Vertical acceleration (5g+) detects potholes, jumps, rough terrain
        current_vertical_vel = self.leatherback.data.root_lin_vel_w[:, 2]
        vertical_accel = torch.abs((current_vertical_vel - self.prev_vertical_vel) / self.control_dt)
        shock_detected = vertical_accel > self.shock_vertical_accel_threshold
        
        return shock_detected

    def _debug_shock_detection(self, shock_events: torch.Tensor):
        """Temporary debug method to verify shock detection isn't firing constantly."""
        if not hasattr(self, '_shock_detection_counter'):
            self._shock_detection_counter = 0
        self._shock_detection_counter += 1

        if self._shock_detection_counter % 300 == 0:  # Every 5 seconds at 60Hz
            num_shocks = torch.sum(shock_events).item()
            if num_shocks > 0:
                print(f"[SHOCK DEBUG] Shocks in last 5s: {num_shocks} events (vertical accel > 5g) across {self.num_envs} envs")

    def _detect_recovery(self) -> torch.Tensor:
        """Detect when robot stabilizes after a shock event.
        
        Returns:
            Boolean tensor indicating which environments recovered this step.
        """
        # Only check recovery for environments currently in recovery mode
        if not torch.any(self.recovering):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # Check if robot is within recovery window
        in_recovery_window = self.shock_timer < self.recovery_window
        
        # Check if robot has stabilized (low angular rates + low vertical accel)
        angular_rates = torch.norm(self.leatherback.data.root_ang_vel_w, dim=1)
        current_vertical_vel = self.leatherback.data.root_lin_vel_w[:, 2]
        vertical_accel = torch.abs((current_vertical_vel - self.prev_vertical_vel) / self.control_dt)
        
        stabilized = (angular_rates < 1.0) & (vertical_accel < 1.0 * 9.81)  # 1g threshold
        
        # Recovery occurs when stabilized within window while in recovery mode
        recovery_occurred = in_recovery_window & stabilized & self.recovering
        
        return recovery_occurred

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
        
        # Reset shock detection and recovery state for these environments
        self.shock_detected[env_ids_tensor] = False
        self.shock_timer[env_ids_tensor] = 0.0
        self.recovering[env_ids_tensor] = False
        self.prev_vertical_vel[env_ids_tensor] = 0.0
        
        # Initialize obstacle views if needed (should already be done in _setup_scene)
        if not hasattr(self, '_prims_initialized'):
            print("[WARNING] Obstacle views not initialized in _setup_scene, initializing now...")
            from isaacsim.core.prims import RigidPrim
            
            self._obstacle_views = []
            for i in range(5):
                view = RigidPrim(f"/World/envs/env_.*/TestObstacle_{i}", reset_xform_properties=False)
                view.initialize()
                self._obstacle_views.append(view)
            self._prims_initialized = True
        
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

    def _create_obstacles_for_source_env(self):
        """Create wall templates in source environment (env_0) BEFORE cloning."""
        print(f"[DEBUG] Creating 5 wall templates (2 gaps + 1 random) in source environment (env_0)...")
        
        # Initialize tensors for 5 walls (2 gaps + 1 random)
        if self._obstacle_sizes is None:
            self._obstacle_sizes = torch.zeros((self.num_envs, 5, 3), device=self.device, dtype=torch.float32)
        if self._obstacle_positions is None:
            self._obstacle_positions = torch.zeros((self.num_envs, 5, 3), device=self.device, dtype=torch.float32)
        
        # Create 4 walls ONLY in source environment (env_0)
        env_idx = 0
        
        for obs_idx in range(5):  # 5 walls: 4 for 2 gaps + 1 random wall
            prim_path = f"/World/envs/env_{env_idx}/TestObstacle_{obs_idx}"
            
            # Use fixed wall dimensions from config
            wall_cfg = CuboidCfg(
                size=(self.cfg.wall_length, self.cfg.wall_depth, self.cfg.wall_height),  # 1.5m × 0.25m × 1.75m
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,  # Dynamic but locked
                    disable_gravity=True,
                    max_linear_velocity=0.0,
                    max_angular_velocity=0.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.02,
                    rest_offset=0.0
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),  # Red color
            )
            
            # Place at origin initially - will be repositioned during reset
            initial_pos = (0.0, 0.0, self.cfg.wall_height / 2.0)
            wall_cfg.func(prim_path, wall_cfg, translation=initial_pos)
        
        print(f"[DEBUG] Created 5 wall templates (2 gaps + 1 random) in env_0")
    
    def _reset_obstacle_positions(self, env_ids: torch.Tensor | Sequence[int]):
        """Place two walls perpendicular to path between waypoints to create a gap the robot must navigate through."""
        # Type guard assertions
        assert self._obstacle_positions is not None
        assert self._obstacle_sizes is not None
        
        if env_ids is None:
            env_ids = self.leatherback._ALL_INDICES
        
        # Convert env_ids to tensor if needed
        if not isinstance(env_ids, torch.Tensor):
            env_ids_tensor = torch.tensor(list(env_ids), dtype=torch.int32, device=self.device)
        else:
            env_ids_tensor = env_ids
        
        num_reset = len(env_ids_tensor)
        
        # Reset obstacle positions to zero
        self._obstacle_positions[env_ids_tensor, :, :] = 0.0
        
        # BATCHED OPERATIONS: Process all environments at once
        
        # 1. Pick TWO random waypoints for all environments (2 gaps per env)
        # Exclude first (0) and last (9) waypoints, choose from waypoints 1-8
        # Gap 1: Random waypoint from 1-8
        waypoint_1_indices = torch.randint(1, self._num_goals - 1, (num_reset,), device=self.device)
        # Gap 2: Different random waypoint from 1-8 (ensure they're different)
        waypoint_2_indices = torch.randint(1, self._num_goals - 1, (num_reset,), device=self.device)
        
        # Ensure gap 2 is different from gap 1
        same_waypoint_mask = waypoint_2_indices == waypoint_1_indices
        waypoint_2_indices = torch.where(
            same_waypoint_mask,
            torch.randint(1, self._num_goals - 1, (num_reset,), device=self.device),
            waypoint_2_indices
        )
        
        # 2. Extract waypoints for GAP 1 (shape: [num_reset, 2])
        waypoint_1_world = self._target_positions[env_ids_tensor, waypoint_1_indices, :2]
        prev_waypoint_1_world = self._target_positions[env_ids_tensor, waypoint_1_indices - 1, :2]
        
        # 3. Extract waypoints for GAP 2 (shape: [num_reset, 2])
        waypoint_2_world = self._target_positions[env_ids_tensor, waypoint_2_indices, :2]
        prev_waypoint_2_world = self._target_positions[env_ids_tensor, waypoint_2_indices - 1, :2]
        
        # 4. Convert to LOCAL coordinates (subtract env origins)
        env_origins = self.scene.env_origins[env_ids_tensor, :2]
        waypoint_1 = waypoint_1_world - env_origins
        prev_waypoint_1 = prev_waypoint_1_world - env_origins
        waypoint_2 = waypoint_2_world - env_origins
        prev_waypoint_2 = prev_waypoint_2_world - env_origins
        
        # 5. Use waypoints as gap centers (harder - robot must navigate gap AND reach waypoint)
        midpoints1 = waypoint_1
        midpoints2 = waypoint_2
        
        # 6. Calculate direction vectors for both gaps (from previous waypoint to target waypoint)
        directions1 = waypoint_1 - prev_waypoint_1
        directions2 = waypoint_2 - prev_waypoint_2
        
        # Normalize directions and handle zero-length vectors
        for directions in [directions1, directions2]:
            direction_lengths = torch.norm(directions, dim=1, keepdim=True)
            zero_length_mask = direction_lengths.squeeze() < 1e-6
            direction_lengths = torch.where(direction_lengths < 1e-6, torch.ones_like(direction_lengths), direction_lengths)
            directions_normalized = directions / direction_lengths
            directions_normalized[zero_length_mask] = torch.tensor([1.0, 0.0], device=self.device)
            directions[:] = directions_normalized
        
        # 7. Calculate perpendicular vectors (90 degree rotation) for both gaps
        perpendiculars1 = torch.stack([-directions1[:, 1], directions1[:, 0]], dim=1)
        perpendiculars2 = torch.stack([-directions2[:, 1], directions2[:, 0]], dim=1)
        
        # 8. Calculate rotation angles for quaternions (shape: [num_reset])
        angles1 = torch.atan2(perpendiculars1[:, 1], perpendiculars1[:, 0])
        angles2 = torch.atan2(perpendiculars2[:, 1], perpendiculars2[:, 0])
        half_angles1 = angles1 / 2.0
        half_angles2 = angles2 / 2.0
        
        # 9. Create quaternions for both gaps (shape: [num_reset, 4])
        wall_orientations1 = torch.zeros((num_reset, 4), device=self.device, dtype=torch.float32)
        wall_orientations1[:, 0] = torch.cos(half_angles1)  # w
        wall_orientations1[:, 3] = torch.sin(half_angles1)  # z
        
        wall_orientations2 = torch.zeros((num_reset, 4), device=self.device, dtype=torch.float32)
        wall_orientations2[:, 0] = torch.cos(half_angles2)  # w
        wall_orientations2[:, 3] = torch.sin(half_angles2)  # z
        
        # 10. Randomize gap sizes for both gaps (shape: [num_reset, 1])
        gap_sizes = torch.rand((num_reset, 1), device=self.device) * (self.cfg.gap_size_range[1] - self.cfg.gap_size_range[0]) + self.cfg.gap_size_range[0]
        half_gaps = gap_sizes / 2.0
        
        # 11. Calculate wall offsets (shape: [num_reset, 1])
        wall_half_length = self.cfg.wall_length / 2.0
        wall_offsets = half_gaps + wall_half_length
        
        # 12. Calculate wall positions for GAP 1 (walls 0,1)
        left_wall1_pos = midpoints1 - perpendiculars1 * wall_offsets
        right_wall1_pos = midpoints1 + perpendiculars1 * wall_offsets
        
        # 13. Calculate wall positions for GAP 2 (walls 2,3)
        left_wall2_pos = midpoints2 - perpendiculars2 * wall_offsets
        right_wall2_pos = midpoints2 + perpendiculars2 * wall_offsets
        
        # 14. Assign wall positions (LOCAL coordinates)
        # Gap 1: walls 0,1
        self._obstacle_positions[env_ids_tensor, 0, :2] = left_wall1_pos
        self._obstacle_positions[env_ids_tensor, 0, 2] = self.cfg.wall_height / 2.0
        self._obstacle_positions[env_ids_tensor, 1, :2] = right_wall1_pos
        self._obstacle_positions[env_ids_tensor, 1, 2] = self.cfg.wall_height / 2.0
        
        # Gap 2: walls 2,3
        self._obstacle_positions[env_ids_tensor, 2, :2] = left_wall2_pos
        self._obstacle_positions[env_ids_tensor, 2, 2] = self.cfg.wall_height / 2.0
        self._obstacle_positions[env_ids_tensor, 3, :2] = right_wall2_pos
        self._obstacle_positions[env_ids_tensor, 3, 2] = self.cfg.wall_height / 2.0
        
        # Store orientations for all 4 walls
        wall_orientations = torch.stack([wall_orientations1, wall_orientations1, wall_orientations2, wall_orientations2], dim=1)
        
        # 13. Convert LOCAL to WORLD coordinates
        self._obstacle_positions[env_ids_tensor, :, :2] += self.scene.env_origins[env_ids_tensor, :2].unsqueeze(1)
        
        # 14. Add exactly 1 random solid wall with collision avoidance
        min_wall_spacing = 4.0  # Minimum 4m between any walls
        min_start_distance = 6.0  # Minimum 6m from robot start
        
        for env_offset in range(num_reset):
            env_id = env_ids_tensor[env_offset]
            
            # Get existing gap wall positions for this environment (indices 0-3)
            existing_positions = self._obstacle_positions[env_id, 0:4, :2].clone()
            robot_start = self.leatherback.data.root_pos_w[env_id, :2]
            
            max_attempts = 20  # Try up to 20 times to find valid position
            placed = False
            
            for attempt in range(max_attempts):
                # Pick random waypoint segment (exclude first segment 0->1)
                segment_idx = torch.randint(2, self._num_goals, (1,), device=self.device).item()
                
                # Get waypoint positions in world coords
                wp_start = self._target_positions[env_id, segment_idx - 1, :2]
                wp_end = self._target_positions[env_id, segment_idx, :2]
                
                # Place wall at random position along segment (20%-80%)
                t = torch.rand(1, device=self.device).item() * 0.6 + 0.2
                wall_center = wp_start + t * (wp_end - wp_start)
                
                # Check collision with existing gap walls
                distances_to_gaps = torch.norm(existing_positions - wall_center.unsqueeze(0), dim=1)
                min_gap_dist = torch.min(distances_to_gaps).item()
                
                # Check distance from robot start
                dist_from_start = torch.norm(wall_center - robot_start).item()
                
                # Accept position if it's far enough from everything
                if min_gap_dist >= min_wall_spacing and dist_from_start >= min_start_distance:
                    self._obstacle_positions[env_id, 4, :2] = wall_center
                    self._obstacle_positions[env_id, 4, 2] = self.cfg.wall_height / 2.0
                    placed = True
                    break
            
            # If couldn't place after max attempts, put far away
            if not placed:
                self._obstacle_positions[env_id, 4, :] = torch.tensor([10000.0, 10000.0, 0.0], device=self.device)
        
        # 15. Move all 5 walls using BATCHED view updates (critical for 2048+ envs)
        # Process environments in batches to avoid overwhelming USD stage updates
        batch_size = 128  # Larger batches for faster initialization (optimized for modern GPUs)
        num_batches = (num_reset + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_reset)
            batch_env_ids = env_ids_tensor[start_idx:end_idx]
            
            for obs_idx in range(5):  # 4 gap walls + 1 random wall
                if obs_idx < len(self._obstacle_views):
                    view = self._obstacle_views[obs_idx]
                    # Safety check: ensure view has instances for all batch environments
                    max_env_id = torch.max(batch_env_ids).item()
                    if view.count <= max_env_id:
                        print(f"[ERROR] Wall {obs_idx} view has only {view.count} instances but trying to access env {max_env_id}")
                        continue
                    
                    wall_positions = self._obstacle_positions[batch_env_ids, obs_idx, :]
                    
                    if obs_idx < 4:
                        # Gap walls use calculated orientations
                        wall_orientations_batch = wall_orientations[start_idx:end_idx, obs_idx, :]
                    else:
                        # Random wall gets random rotation (0-120 degrees)
                        wall_orientations_batch = torch.zeros((len(batch_env_ids), 4), device=self.device, dtype=torch.float32)
                        random_angles = torch.rand(len(batch_env_ids), device=self.device) * (2 * torch.pi / 3)  # 0 to 120 degrees
                        half_angles = random_angles / 2.0
                        wall_orientations_batch[:, 0] = torch.cos(half_angles)
                        wall_orientations_batch[:, 3] = torch.sin(half_angles)
                    
                    view.set_world_poses(
                        wall_positions, 
                        wall_orientations_batch, 
                        indices=batch_env_ids
                    )
        
        print(f"[TWO-GAP + WALL] Reset {num_reset} environments in {num_batches} batches: 2 gaps + 1 random wall")
        # Debug first environment - DISABLED FOR PERFORMANCE
        # if 0 in env_ids_tensor.cpu():
        #     print(f"  Env 0 Gap 1 walls: {self._obstacle_positions[0, 0, :].cpu().numpy()}, {self._obstacle_positions[0, 1, :].cpu().numpy()}")
        #     print(f"  Env 0 Gap 2 walls: {self._obstacle_positions[0, 2, :].cpu().numpy()}, {self._obstacle_positions[0, 3, :].cpu().numpy()}")
        #     print(f"  Env 0 Random wall: {self._obstacle_positions[0, 4, :2].cpu().numpy()}")

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

