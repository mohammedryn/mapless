"""
Shared observation processing utilities.

Used by BOTH ForestEnv (training) and NavigationNode (deployment) to guarantee
the policy always receives an identically constructed observation vector in
simulation and on the real robot.  Any change to observation construction must
be made here ONLY — both callers will pick it up automatically.

Observation layout  (362-dimensional, float32):
    [0 … 359]  — lidar ranges, normalised to [0, 1]  (divided by LIDAR_MAX_RANGE)
    [360]      — distance to goal, normalised to [0, 1]  (clipped at GOAL_NORM_DIST)
    [361]      — bearing to goal, normalised to [-1, 1]  (divided by π)
"""

import math
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

# Slamtec C1M1 R2 datasheet max range (metres).
# CRITICAL: previous code used 3.5 m (simulation RPLIDAR A1 value).
# The real sensor reaches 12 m — using the wrong constant shifts every
# normalised observation above 3.5 m to a clipped 1.0, which the trained
# policy has never seen.
LIDAR_MAX_RANGE: float = 12.0

# Distance beyond which normalised goal distance saturates at 1.0.
GOAL_NORM_DIST: float = 10.0

# Number of lidar points in the observation vector.
N_SCAN: int = 360


# ── Public API ────────────────────────────────────────────────────────────────

def process_scan(
    raw_ranges,
    n_scan: int = N_SCAN,
    max_range: float = LIDAR_MAX_RANGE,
    noise_std: float = 0.0,
) -> np.ndarray:
    """Resample a raw LaserScan range array to exactly *n_scan* points and
    normalise to ``[0, 1]``.

    Parameters
    ----------
    raw_ranges:
        Iterable of float range values from a ``sensor_msgs/LaserScan``.
    n_scan:
        Target number of scan points (default 360).
    max_range:
        Sensor maximum range used for normalisation and inf replacement.
    noise_std:
        Optional Gaussian noise standard deviation (metres) applied **before**
        normalisation — used during training for domain randomisation.
        Set to 0.0 (default) for deterministic inference.

    Returns
    -------
    np.ndarray
        float32 array of shape ``(n_scan,)``, values in ``[0, 1]``.
    """
    ranges = np.array(raw_ranges, dtype=np.float64)

    # Downsample or pad to exactly n_scan points.
    if len(ranges) >= n_scan:
        step = len(ranges) // n_scan
        ranges = ranges[::step][:n_scan]
    else:
        ranges = np.pad(
            ranges,
            (0, n_scan - len(ranges)),
            mode="constant",
            constant_values=max_range,
        )

    # Replace NaN / ±inf with safe sentinel values.
    ranges = np.nan_to_num(ranges, nan=max_range, posinf=max_range, neginf=0.0)
    ranges = np.clip(ranges, 0.0, max_range)

    # Optional domain-randomisation noise (training only).
    if noise_std > 0.0:
        ranges = ranges + np.random.normal(0.0, noise_std, size=ranges.shape)
        ranges = np.clip(ranges, 0.0, max_range)

    return (ranges / max_range).astype(np.float32)


def get_goal_obs(odom, goal_x: float, goal_y: float):
    """Compute normalised goal distance and bearing from an Odometry message.

    Parameters
    ----------
    odom:
        Latest ``nav_msgs/Odometry`` message.
    goal_x, goal_y:
        Goal position in the odometry frame (metres).

    Returns
    -------
    norm_dist : float
        Goal distance normalised to ``[0, 1]``.
    norm_angle : float
        Bearing to goal normalised to ``[-1, 1]``.
    raw_distance : float
        Euclidean distance to goal in metres (used for reward computation).
    """
    pos = odom.pose.pose.position
    orient = odom.pose.pose.orientation

    # Quaternion → yaw (2-D planar robot assumption).
    siny_cosp = 2.0 * (orient.w * orient.z + orient.x * orient.y)
    cosy_cosp = 1.0 - 2.0 * (orient.y * orient.y + orient.z * orient.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    dx = goal_x - pos.x
    dy = goal_y - pos.y
    raw_distance = math.sqrt(dx * dx + dy * dy)

    angle_to_goal = math.atan2(dy, dx) - yaw
    # Wrap to [-π, π].
    angle_to_goal = (angle_to_goal + math.pi) % (2.0 * math.pi) - math.pi

    norm_dist = float(np.clip(raw_distance / GOAL_NORM_DIST, 0.0, 1.0))
    norm_angle = float(angle_to_goal / math.pi)

    return norm_dist, norm_angle, raw_distance


def build_observation(
    scan_raw,
    odom,
    goal_x: float,
    goal_y: float,
    n_scan: int = N_SCAN,
    max_range: float = LIDAR_MAX_RANGE,
    lidar_noise_std: float = 0.0,
):
    """Build the full 362-dimensional policy observation vector.

    Layout: ``[norm_scan (360) | norm_dist (1) | norm_angle (1)]``

    Parameters
    ----------
    scan_raw:
        Raw range array from the latest LaserScan message.
    odom:
        Latest Odometry message, or ``None`` (returns zero goal features).
    goal_x, goal_y:
        Goal coordinates in the odometry frame.
    n_scan:
        Number of lidar rays (default 360).
    max_range:
        Sensor maximum range for normalisation (default ``LIDAR_MAX_RANGE``).
    lidar_noise_std:
        Gaussian noise std in metres for domain randomisation (training only).

    Returns
    -------
    obs : np.ndarray
        float32 observation vector of shape ``(n_scan + 2,)``.
    raw_distance : float
        Euclidean metres to goal (for reward computation); ``0.0`` if odom
        is ``None``.
    """
    norm_scan = process_scan(scan_raw, n_scan, max_range, lidar_noise_std)

    if odom is not None:
        norm_dist, norm_angle, raw_distance = get_goal_obs(odom, goal_x, goal_y)
    else:
        norm_dist, norm_angle, raw_distance = 0.0, 0.0, 0.0

    obs = np.concatenate(
        [norm_scan, [norm_dist, norm_angle]], dtype=np.float32
    )
    return obs, raw_distance
