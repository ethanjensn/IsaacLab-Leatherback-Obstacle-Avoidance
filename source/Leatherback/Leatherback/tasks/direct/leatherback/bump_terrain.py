# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom terrain generator for random spherical bumps."""

from __future__ import annotations

import numpy as np
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh


@configclass
class HfSphericalBumpsTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a terrain with random spherical bumps."""

    bump_height_range: tuple[float, float] = (0.01, 0.02)
    """The minimum and maximum height of the bumps (in m). Default: 1-2cm (very subtle)."""

    bump_radius_range: tuple[float, float] = (0.20, 0.35)
    """The minimum and maximum radius of the bumps (in m). Default: 40-70cm diameter."""

    num_bumps_per_env: int = 100
    """The number of bumps to generate per environment. Default: 100."""


@height_field_to_mesh
def spherical_bumps_terrain(difficulty: float, cfg: HfSphericalBumpsTerrainCfg) -> np.ndarray:
    """Generate a terrain with random spherical bumps.

    The terrain consists of randomly placed spherical bumps with varying heights and radii.
    The bumps are distributed across the entire terrain to create challenging off-road conditions
    that trigger the robot's shock detection and recovery system.

    Args:
        difficulty: The difficulty of the terrain (0.0 to 1.0). Higher difficulty produces taller bumps.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        Shape is (width_pixels, length_pixels).
    """
    # Resolve terrain configuration based on difficulty
    bump_height = cfg.bump_height_range[0] + difficulty * (
        cfg.bump_height_range[1] - cfg.bump_height_range[0]
    )

    # Convert to discrete units
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    # Initialize flat height field
    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.float32)

    # Create coordinate grids for distance calculations
    x_coords = np.arange(width_pixels)
    y_coords = np.arange(length_pixels)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')

    # Generate random bumps
    for _ in range(cfg.num_bumps_per_env):
        # Random position (in pixels)
        center_x = np.random.randint(0, width_pixels)
        center_y = np.random.randint(0, length_pixels)

        # Random radius (in meters, then convert to pixels)
        radius_m = np.random.uniform(cfg.bump_radius_range[0], cfg.bump_radius_range[1])
        radius_pixels = radius_m / cfg.horizontal_scale

        # Random height variation (±20% from difficulty-based height)
        height_variation = np.random.uniform(0.8, 1.2)
        bump_height_actual = bump_height * height_variation

        # Calculate distance from bump center to all points
        distances = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)

        # Create spherical bump using smooth Gaussian-like falloff
        # Points within radius get positive height, falloff smoothly to zero
        mask = distances <= radius_pixels
        normalized_dist = np.clip(distances / radius_pixels, 0.0, 1.0)
        
        # Smoother bump shape: h = h_max * (1 - d/r)^6
        # This creates an extremely flat, smooth bump profile (power 6 falloff)
        bump_shape = (1.0 - normalized_dist) ** 6
        bump_heights = bump_height_actual * bump_shape * mask

        # Add bump to terrain (use max to handle overlapping bumps)
        hf_raw = np.maximum(hf_raw, bump_heights)

    # Convert to discrete integer heights
    hf_discrete = np.rint(hf_raw / cfg.vertical_scale).astype(np.int16)

    return hf_discrete

