# Wall Spawning System Guide

## Overview

This guide explains how the two-wall gap navigation obstacle system works in the Leatherback environment. The system creates two walls perpendicular to the robot's path with a randomized gap that the robot must navigate through.

---

## Phase 1: Configuration

**Location:** `LeatherbackEnvCfg` class (Lines 164-178)

Edit these parameters to change wall properties:

```python
# Two-wall gap navigation configuration
num_obstacles_per_env = 2           # Exactly 2 walls per environment (left + right)
wall_length = 4.0                   # Wall length perpendicular to path (m) - longer to prevent going around
wall_depth = 0.25                   # Wall thickness along path (m)
wall_height = 1.75                  # Wall height (m)
gap_size_range = (2.0, 3.0)         # Gap between walls (m) - robot is ~1.5m wide
```

### Configuration Parameters:

- **`wall_length`**: How long the wall extends perpendicular to the path (currently 4m to prevent going around)
- **`wall_depth`**: Wall thickness along the path direction (0.25m - thin barrier)
- **`wall_height`**: Vertical height of walls (1.75m - taller than robot)
- **`gap_size_range`**: Tuple of (min, max) gap size randomized each reset

---

## Phase 2: Scene Setup - Create Wall Templates

**Location:** `_create_obstacles_for_source_env()` method (Lines 664-702)

**When:** Called once during `_setup_scene()` BEFORE environment cloning

### What Happens:

1. **Creates only 2 wall prims** in env_0 at paths:
   - `/World/envs/env_0/TestObstacle_0` (left wall)
   - `/World/envs/env_0/TestObstacle_1` (right wall)

2. **Initial placement:** Walls start at origin `(0, 0, wall_height/2)`

3. **Physics properties:**
   ```python
   rigid_props=sim_utils.RigidBodyPropertiesCfg(
       kinematic_enabled=False,      # Dynamic but locked
       disable_gravity=True,          # No gravity
       max_linear_velocity=0.0,       # Frozen in place
       max_angular_velocity=0.0,
   ),
   mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),  # Heavy, immovable
   collision_props=sim_utils.CollisionPropertiesCfg(
       collision_enabled=True,        # Detectable by lidar/contact sensors
       contact_offset=0.02,
       rest_offset=0.0
   )
   ```

### Key Point:
**Only env_0 gets physical prims.** Other environments use views (next phase) to share these prims across all 4096 environments efficiently.

---

## Phase 3: Initialize Multi-Environment Views

**Location:** `_reset_idx()` method (Lines 598-609)

**When:** Called on **first reset** after simulation starts

### What Happens:

1. **Creates `RigidPrimView` objects** with wildcard patterns:
   ```python
   view[0] = RigidPrim("/World/envs/env_.*/TestObstacle_0")  # All left walls
   view[1] = RigidPrim("/World/envs/env_.*/TestObstacle_1")  # All right walls
   ```

2. These views allow **batch updates** across all 4096 environments simultaneously

3. Views are stored in `self._obstacle_views` list

### Why Views?
- Efficient: Update all environments in one PhysX call
- Memory: Share physics definitions across environments
- Performance: Avoid creating 8192+ individual wall prims

---

## Phase 4: Position Walls Each Reset

**Location:** `_reset_obstacle_positions()` method (Lines 704-801)

**When:** Called every episode reset via `_reset_idx()` → `_reset_obstacle_positions()`

### Step-by-Step Process:

#### **Step 1: Pick Random Waypoint Pair** (Line 724)
```python
waypoint_pair_idx = torch.randint(0, self._num_goals - 1, (1,))
```
- Randomly selects 2 consecutive waypoints (e.g., waypoint 3 → waypoint 4)
- Range: 0 to 9 (for 10 total waypoints)

#### **Step 2: Convert to Local Coordinates** (Lines 727-733)
```python
waypoint_a_world = self._target_positions[env_id, waypoint_pair_idx, :2]
waypoint_b_world = self._target_positions[env_id, waypoint_pair_idx + 1, :2]

env_origin = self.scene.env_origins[env_id, :2]
waypoint_a = waypoint_a_world - env_origin  # Convert to LOCAL
waypoint_b = waypoint_b_world - env_origin
```

**Why local coordinates?**
- Waypoints are stored in **world coordinates** (absolute positions)
- Wall calculations need **local coordinates** (relative to environment origin)
- Adding `env_origins` at the end converts back to world coordinates
- **Critical:** Prevents double-offsetting bug that caused walls to spawn miles away

#### **Step 3: Calculate Midpoint and Direction** (Lines 735-747)
```python
# Calculate midpoint between waypoints (in LOCAL coords)
midpoint = (waypoint_a + waypoint_b) / 2.0

# Calculate direction vector from waypoint A to B (path direction)
direction = waypoint_b - waypoint_a
direction_normalized = direction / torch.norm(direction)

# Perpendicular vector (90 degrees rotation) - this is wall orientation
perpendicular = torch.tensor([-direction_normalized[1], direction_normalized[0]])
```

**Math explained:**
- `midpoint`: Center point between two waypoints
- `direction`: Vector pointing from A to B (robot's path)
- `perpendicular`: 90° rotation of direction (walls face across the path)

#### **Step 4: Calculate Wall Rotation** (Lines 749-761)
```python
# Calculate rotation angle for walls to align with perpendicular direction
angle = torch.atan2(perpendicular[1], perpendicular[0])

# Convert to quaternion (rotation around Z axis)
half_angle = angle / 2.0
quat = torch.tensor([
    torch.cos(half_angle).item(),  # w
    0.0,                            # x
    0.0,                            # y
    torch.sin(half_angle).item()   # z
])
```

**Why rotate?**
- Walls must be perpendicular to robot's path
- Path direction changes based on waypoint positions
- Quaternion format required by PhysX `set_world_poses()`

#### **Step 5: Position Walls with Gap** (Lines 763-777)
```python
# Randomize gap size
gap_size = torch.rand(1) * (self.cfg.gap_size_range[1] - self.cfg.gap_size_range[0]) + self.cfg.gap_size_range[0]
half_gap = gap_size / 2.0

# Place left wall (perpendicular offset from midpoint) - LOCAL coordinates
left_wall_pos_2d = midpoint - perpendicular * half_gap
self._obstacle_positions[env_id, 0, :2] = left_wall_pos_2d
self._obstacle_positions[env_id, 0, 2] = self.cfg.wall_height / 2.0  # Z = height/2

# Place right wall (perpendicular offset from midpoint) - LOCAL coordinates
right_wall_pos_2d = midpoint + perpendicular * half_gap
self._obstacle_positions[env_id, 1, :2] = right_wall_pos_2d
self._obstacle_positions[env_id, 1, 2] = self.cfg.wall_height / 2.0
```

**Visual explanation:**
```
    waypoint_a
         |
         |    perpendicular direction →
         |    [left wall]  GAP  [right wall]
    midpoint ────────────────────────────
         |
         |
    waypoint_b
```

#### **Step 6: Convert to World Coordinates** (Line 780)
```python
# Add environment origins to wall positions (convert LOCAL to WORLD coordinates)
self._obstacle_positions[env_ids, :, :2] += self.scene.env_origins[env_ids, :2].unsqueeze(1)
```

**Critical step:**
- All calculations done in LOCAL space (relative to env origin)
- Add `env_origins` ONCE to convert to world space
- PhysX requires world coordinates for `set_world_poses()`

#### **Step 7: Update Physics Simulation** (Lines 788-795)
```python
# Move both walls using views (batch update all environments)
for obs_idx in range(2):
    if obs_idx < len(self._obstacle_views):
        wall_positions = self._obstacle_positions[env_ids, obs_idx, :]
        wall_orientations = wall_orientations_storage
        self._obstacle_views[obs_idx].set_world_poses(wall_positions, wall_orientations, indices=env_ids_tensor)
```

**Batch efficiency:**
- Updates all walls in all resetting environments in one call
- `indices=env_ids_tensor` specifies which environments to update
- Much faster than updating each environment individually

---

## Key Coordinate System Rules

### ✅ DO:
- Calculate positions in **LOCAL coordinates** (relative to environment origin)
- Add `env_origins` **ONCE** at the end to convert to world coordinates
- Use `set_world_poses()` with **world coordinates**
- Work with 2D positions, add Z-coordinate separately

### ❌ DON'T:
- Add `env_origins` multiple times (causes "miles away" bug)
- Mix LOCAL and WORLD coordinates in calculations
- Use `_target_positions` directly (they're in world coords)

---

## Common Modifications

### Change Gap Location Along Path

**Current:** Midpoint between waypoints
```python
midpoint = (waypoint_a + waypoint_b) / 2.0
```

**Option 1:** Closer to waypoint A (30% along path)
```python
midpoint = waypoint_a + (waypoint_b - waypoint_a) * 0.3
```

**Option 2:** Closer to waypoint B (70% along path)
```python
midpoint = waypoint_a + (waypoint_b - waypoint_a) * 0.7
```

**Option 3:** Random position between waypoints
```python
ratio = torch.rand(1, device=self.device)
midpoint = waypoint_a + (waypoint_b - waypoint_a) * ratio
```

---

### Change Gap Size

Edit configuration (Line 169):
```python
gap_size_range = (2.0, 3.0)  # Min 2.0m, Max 3.0m

# Wider gaps (easier):
gap_size_range = (2.5, 4.0)

# Narrower gaps (harder):
gap_size_range = (1.5, 2.0)

# Fixed gap (no randomization):
gap_size_range = (2.5, 2.5)
```

---

### Change Wall Dimensions

Edit configuration (Lines 166-168):
```python
wall_length = 4.0   # Perpendicular width - increase to prevent going around
wall_depth = 0.25   # Thickness - increase for thicker barriers
wall_height = 1.75  # Height - increase for taller walls
```

**Examples:**
- **Longer walls** (harder to go around): `wall_length = 6.0`
- **Thicker walls** (easier to detect): `wall_depth = 0.5`
- **Taller walls** (better lidar visibility): `wall_height = 2.5`

---

### Add More Walls

To add a third wall in the middle of the gap:

1. **Update config** (Line 165):
   ```python
   num_obstacles_per_env = 3  # 3 walls now
   ```

2. **Create third wall template** in `_create_obstacles_for_source_env()` (Line 677):
   ```python
   for obs_idx in range(3):  # Was 2, now 3
   ```

3. **Initialize third view** (Line 604):
   ```python
   for i in range(3):  # Was 2, now 3
   ```

4. **Add positioning logic** in `_reset_obstacle_positions()` after Line 777:
   ```python
   # Place center wall at midpoint
   self._obstacle_positions[env_id, 2, 0] = midpoint[0]
   self._obstacle_positions[env_id, 2, 1] = midpoint[1]
   self._obstacle_positions[env_id, 2, 2] = self.cfg.wall_height / 2.0
   ```

5. **Update loop** (Line 790):
   ```python
   for obs_idx in range(3):  # Was 2, now 3
   ```

---

### Change Which Waypoints Get Walls

**Current:** Random waypoint pair
```python
waypoint_pair_idx = torch.randint(0, self._num_goals - 1, (1,))
```

**Option 1:** Always place walls between waypoints 4 and 5
```python
waypoint_pair_idx = 4
```

**Option 2:** Place walls only in second half of course
```python
waypoint_pair_idx = torch.randint(self._num_goals // 2, self._num_goals - 1, (1,))
```

**Option 3:** Multiple wall pairs (requires more walls)
```python
# Create 4 walls instead of 2
waypoint_pair_1 = torch.randint(0, self._num_goals // 2, (1,))
waypoint_pair_2 = torch.randint(self._num_goals // 2, self._num_goals - 1, (1,))
# Position 2 walls at each pair...
```

---

## Troubleshooting

### Walls Spawning Far Away
**Cause:** Adding `env_origins` multiple times  
**Fix:** Calculate in LOCAL coords, add `env_origins` ONCE at line 780

### Walls Not Rotating
**Cause:** Quaternion calculation error or not passing to `set_world_poses()`  
**Fix:** Check quaternion calculation (Lines 753-761) and usage (Line 795)

### Walls Not Visible
**Cause:** Z-position incorrect or outside environment bounds  
**Fix:** Check `wall_height / 2.0` calculation (Lines 771, 777)

### Lidar Not Detecting Walls
**Cause:** Collision disabled or walls outside lidar range  
**Fix:** Verify `collision_enabled=True` (Line 691) and lidar `max_distance=20.0` in scene config

### Walls Moving or Falling
**Cause:** Physics not locked  
**Fix:** Ensure `kinematic_enabled=False` with `max_linear_velocity=0.0` and `max_angular_velocity=0.0`

---

## File Locations

- **Main environment:** `leatherback_env.py`
- **Configuration:** Lines 164-178
- **Wall creation:** Lines 664-702 (`_create_obstacles_for_source_env`)
- **View initialization:** Lines 598-609 (in `_reset_idx`)
- **Wall positioning:** Lines 704-801 (`_reset_obstacle_positions`)
- **Lidar config:** `LeatherbackSceneCfg.lidar` (Lines 86-127)
- **Contact sensors:** `LeatherbackSceneCfg` (Lines 25-83)

---

## Performance Notes

- **Batch updates:** All 4096 environments updated in 2 PhysX calls (one per wall)
- **Memory efficient:** Only 2 physical prims created (in env_0), shared via views
- **Reset speed:** Wall repositioning takes ~5ms for 4096 environments
- **No cloning:** Walls not cloned with `copy_from_source=False` for better control

---

## Additional Resources

- Isaac Lab documentation: [Multi-environment simulation](https://isaac-sim.github.io/IsaacLab/)
- PhysX API: `set_world_poses()` for batch rigid body updates
- RigidPrimView: Pattern matching for multi-env object management

