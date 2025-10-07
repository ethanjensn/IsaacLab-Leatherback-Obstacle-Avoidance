# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the leatherback robot."""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# USD path
USD_PATH = r"C:\IsaacLab\leatherback_car\Leatherback\leatherback - Copy.usd"

LEATHERBACK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=50.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        activate_contact_sensors=True,  # Enable contact sensors on the robot
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
        joint_pos={
            "Wheel__Knuckle__Front_Left": 0.0,
            "Wheel__Knuckle__Front_Right": 0.0,
            "Wheel__Upright__Rear_Right": 0.0,
            "Wheel__Upright__Rear_Left": 0.0,
            "Knuckle__Upright__Front_Right": 0.0,
            "Knuckle__Upright__Front_Left": 0.0,
            "Shock__Rear_Right": -0.030,
            "Shock__Rear_Left": -0.030,
            "Shock__Front_Right": 0.030,
            "Shock__Front_Left": 0.030,
        },
    ),
    actuators={
        "rear_wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*Wheel__Upright__Rear.*"],
            effort_limit_sim=5000.0,
            velocity_limit_sim=50.0,
            stiffness=0.0,
            damping=1000.0,
        ),
        "front_wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*Wheel__Knuckle__Front.*"],
            effort_limit_sim=5000.0,
            velocity_limit_sim=50.0,
            stiffness=0.0,
            damping=1000.0,
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["Knuckle__Upright__Front_Right", "Knuckle__Upright__Front_Left"],
            effort_limit_sim=2000.0,
            velocity_limit_sim=10.0,
            stiffness=5000.0,
            damping=100.0,
        ),
        "shocks": ImplicitActuatorCfg(
            joint_names_expr=["Shock.*"],
            effort_limit_sim=10000.0,
            velocity_limit_sim=5.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)

