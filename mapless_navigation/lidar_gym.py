"""
LidarGym — fast pure-Python 2D lidar simulation environment.

A zero-ROS, zero-Gazebo drop-in training replacement for ForestEnv.
Uses vectorised NumPy raycasting against circular obstacles in a 2-D arena.

Throughput:  ~10,000–50,000 steps/sec per process (vs 20 steps/sec with Gazebo).
With SubprocVecEnv(n_envs=8): 80,000–400,000 effective steps/sec.

Observation space : 362-dimensional float32 — identical to ForestEnv.
Action space      : 2-dimensional float32 in [-1, 1] — identical to ForestEnv.
Reward function   : identical to ForestEnv (progress + proximity + angular + time).
Domain rand       : per-episode lidar noise-std and speed-scale — identical to ForestEnv.
"""

import math
import os

import numpy as np
import gymnasium
from gymnasium import spaces

try:
    import yaml as _yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

# ── Arena geometry ────────────────────────────────────────────────────────────
# Robot spawns at (0, 0) — the "odom origin".
# Default goal ranges from training.yaml span x ∈ [2, 8], y ∈ [-3, 3].
# Arena is large enough that goals and obstacles fit comfortably.

_X_MIN, _X_MAX = -4.0, 13.0   # 17 m wide
_Y_MIN, _Y_MAX = -8.0,  8.0   # 16 m tall
_SPAWN = (0.0, 0.0, 0.0)       # (x, y, theta) robot initial pose

# Obstacle generation
_OBS_R_MIN = 0.20   # metres
_OBS_R_MAX = 1.00
_OBS_N_MIN = 8
_OBS_N_MAX = 20

_DT     = 0.05   # seconds per step (matches ForestEnv 20 Hz loop)
_N_SCAN = 360


# ── Config helper ─────────────────────────────────────────────────────────────

def _load_training_yaml() -> dict:
    """Load config/training.yaml, returning {} on any failure."""
    if not _YAML_OK:
        return {}
    for path in _candidate_config_paths():
        try:
            with open(path) as f:
                data = _yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            pass
    return {}


def _candidate_config_paths():
    # 1. colcon-installed package share
    try:
        from ament_index_python.packages import get_package_share_directory
        yield os.path.join(
            get_package_share_directory("mapless_navigation"),
            "config", "training.yaml")
    except Exception:
        pass
    # 2. source-tree relative path (repo root / config / training.yaml)
    here = os.path.dirname(os.path.abspath(__file__))
    yield os.path.normpath(os.path.join(here, "..", "config", "training.yaml"))


# ── Main class ────────────────────────────────────────────────────────────────

class LidarGym(gymnasium.Env):
    """Fast 2-D lidar navigation environment — no ROS, no Gazebo.

    Parameters
    ----------
    env_config:
        Optional dict with the same keys as the ``env_config`` block of
        ``config/training.yaml``.  Missing keys fall back to defaults.
        If *None* (default), the YAML is loaded automatically.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: dict = None):
        super().__init__()

        if env_config is None:
            full = _load_training_yaml()
            env_config = full.get("env_config", {})

        cfg = env_config

        # Reward / termination parameters
        self._max_range      = float(cfg.get("max_range",                12.0))
        self._col_dist       = float(cfg.get("collision_dist",            0.20))
        self._col_pen        = float(cfg.get("collision_penalty",        100.0))
        self._goal_dist      = float(cfg.get("goal_reached_dist",         0.50))
        self._goal_rew       = float(cfg.get("goal_reached_reward",      100.0))
        self._prox_dist      = float(cfg.get("proximity_penalty_dist",    0.50))
        self._prox_w         = float(cfg.get("proximity_penalty_weight",   5.0))
        self._ang_w          = float(cfg.get("angular_penalty_weight",     0.05))
        self._max_steps      = int(  cfg.get("max_steps",                500))

        # Goal randomization
        self._goal_x_range   = cfg.get("goal_x_range", [2.0, 8.0])
        self._goal_y_range   = cfg.get("goal_y_range", [-3.0, 3.0])

        # Domain randomization
        dr = cfg.get("domain_rand", {})
        self._noise_max      = float(dr.get("lidar_noise_std",   0.03))
        self._speed_min      = float(dr.get("speed_scale_min",   0.85))
        self._speed_max      = float(dr.get("speed_scale_max",   1.15))

        # Robot physical limits (from rover.yaml / env_config)
        self._v_max   = float(cfg.get("max_linear_vel",  0.26))
        self._w_max   = float(cfg.get("max_angular_vel", 1.82))

        # Spaces — identical to ForestEnv
        n_obs     = _N_SCAN + 2
        obs_low   = np.zeros(n_obs, dtype=np.float32)
        obs_low[-1] = -1.0  # bearing can be negative
        self.observation_space = spaces.Box(
            low=obs_low, high=1.0, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Episode state (populated in reset)
        self._x          = 0.0
        self._y          = 0.0
        self._theta      = 0.0
        self._goal_x     = 5.0
        self._goal_y     = 0.0
        self._obstacles  = np.zeros((0, 3), dtype=np.float64)  # (N, [cx, cy, r])
        self._step_n     = 0
        self._prev_dist  = 0.0
        self._noise_std  = 0.0
        self._speed      = 1.0
        self._rng        = np.random.default_rng()
        # Cached base ray angles — offsets from heading, shape (360,)
        self._base_angles = np.linspace(0.0, 2.0 * math.pi, _N_SCAN, endpoint=False)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_n = 0

        # Per-episode domain randomisation
        self._noise_std = float(self._rng.uniform(0.0, self._noise_max))
        self._speed     = float(self._rng.uniform(self._speed_min, self._speed_max))

        # Randomise goal
        self._goal_x = float(self._rng.uniform(*self._goal_x_range))
        self._goal_y = float(self._rng.uniform(*self._goal_y_range))

        # Reset robot pose
        self._x, self._y, self._theta = _SPAWN

        # Place obstacles with clearance around spawn and goal
        self._obstacles = self._sample_obstacles()

        raw = self._cast_rays()
        obs = self._build_obs(raw)
        self._prev_dist = self._goal_range()
        return obs, {}

    def step(self, action):
        self._step_n += 1

        # Unicycle kinematics
        v = (float(action[0]) + 1.0) * 0.5 * self._v_max * self._speed
        w = float(action[1]) * self._w_max

        self._theta = (self._theta + w * _DT + math.pi) % (2.0 * math.pi) - math.pi
        self._x    += v * math.cos(self._theta) * _DT
        self._y    += v * math.sin(self._theta) * _DT

        # Keep robot inside arena (walls will register as near obstacles anyway)
        m = 0.01
        self._x = float(np.clip(self._x, _X_MIN + m, _X_MAX - m))
        self._y = float(np.clip(self._y, _Y_MIN + m, _Y_MAX - m))

        raw      = self._cast_rays()
        obs      = self._build_obs(raw)
        dist     = self._goal_range()
        rew, done = self._calc_reward(raw, dist, action)
        self._prev_dist = dist
        truncated = self._step_n >= self._max_steps
        return obs, rew, done, truncated, {}

    def close(self):
        pass

    # ── Core helpers ──────────────────────────────────────────────────────────

    def _cast_rays(self) -> np.ndarray:
        """Vectorised 360-ray cast.  Returns float64 distances in [0, max_range]."""
        angles = self._theta + self._base_angles
        cos_a  = np.cos(angles)   # (360,)
        sin_a  = np.sin(angles)
        ox, oy = self._x, self._y
        INF    = self._max_range * 2.0

        ranges = np.full(_N_SCAN, INF, dtype=np.float64)

        # ── Wall (AABB) intersections ─────────────────────────────────────────
        with np.errstate(divide="ignore", invalid="ignore"):
            for wall_t, check, lo, hi in (
                ((_X_MIN - ox) / cos_a,  oy + ((_X_MIN - ox) / cos_a) * sin_a, _Y_MIN, _Y_MAX),
                ((_X_MAX - ox) / cos_a,  oy + ((_X_MAX - ox) / cos_a) * sin_a, _Y_MIN, _Y_MAX),
                ((_Y_MIN - oy) / sin_a,  ox + ((_Y_MIN - oy) / sin_a) * cos_a, _X_MIN, _X_MAX),
                ((_Y_MAX - oy) / sin_a,  ox + ((_Y_MAX - oy) / sin_a) * cos_a, _X_MIN, _X_MAX),
            ):
                ok = (wall_t > 1e-6) & np.isfinite(wall_t) & (check >= lo) & (check <= hi)
                ranges = np.minimum(ranges, np.where(ok, wall_t, INF))

        # ── Circular obstacle intersections ───────────────────────────────────
        if len(self._obstacles):
            cx = self._obstacles[:, 0]   # (N,)
            cy = self._obstacles[:, 1]
            r  = self._obstacles[:, 2]

            ocx = ox - cx   # (N,)
            ocy = oy - cy

            # b[ray, obs] = 2*(dir·oc);  c[obs] = |oc|^2 - r^2
            b    = 2.0 * (cos_a[:, None] * ocx + sin_a[:, None] * ocy)  # (360, N)
            c    = ocx**2 + ocy**2 - r**2                                 # (N,)
            disc = b**2 - 4.0 * c                                         # (360, N)

            with np.errstate(invalid="ignore"):
                t_hit = (-b - np.sqrt(np.maximum(disc, 0.0))) * 0.5  # (360, N)

            valid    = (disc > 0.0) & (t_hit > 1e-6)
            t_min_ob = np.where(valid, t_hit, INF).min(axis=1)        # (360,)
            ranges   = np.minimum(ranges, t_min_ob)

        return np.clip(ranges, 0.0, self._max_range)

    def _build_obs(self, raw_ranges: np.ndarray) -> np.ndarray:
        """Construct 362-dim observation from raw (unnoised) lidar distances."""
        # Add domain-rand noise (identical in purpose to forest_env + obs_utils)
        if self._noise_std > 0.0:
            noisy = raw_ranges + self._rng.standard_normal(_N_SCAN) * self._noise_std
            noisy = np.clip(noisy, 0.0, self._max_range)
        else:
            noisy = raw_ranges

        norm_scan = (noisy / self._max_range).astype(np.float32)

        # Goal bearing and distance (pure geometry — no Odometry message needed)
        dx      = self._goal_x - self._x
        dy      = self._goal_y - self._y
        dist    = math.sqrt(dx * dx + dy * dy)
        bearing = (math.atan2(dy, dx) - self._theta + math.pi) % (2.0 * math.pi) - math.pi

        norm_d = float(np.clip(dist / 10.0, 0.0, 1.0))
        norm_b = float(bearing / math.pi)

        return np.concatenate([norm_scan, [norm_d, norm_b]], dtype=np.float32)

    def _calc_reward(self, raw_ranges: np.ndarray, curr_dist: float, action) -> tuple:
        """Reward identical to ForestEnv._calculate_reward (collision uses unnoised ranges)."""
        min_laser = float(raw_ranges.min())

        if min_laser < self._col_dist:
            return -self._col_pen, True

        if curr_dist < self._goal_dist:
            return self._goal_rew, True

        reward  = (self._prev_dist - curr_dist) * 10.0
        if min_laser < self._prox_dist:
            reward -= self._prox_w * (self._prox_dist - min_laser)
        reward -= self._ang_w * abs(float(action[1]))
        reward -= 0.1
        return reward, False

    def _goal_range(self) -> float:
        dx = self._goal_x - self._x
        dy = self._goal_y - self._y
        return math.sqrt(dx * dx + dy * dy)

    def _sample_obstacles(self) -> np.ndarray:
        """Rejection-sample circular obstacles with clearance around spawn and goal."""
        n_target = int(self._rng.integers(_OBS_N_MIN, _OBS_N_MAX + 1))
        obstacles = []
        spawn_clear = 2.0
        goal_clear  = 1.0

        for _ in range(600):
            if len(obstacles) >= n_target:
                break
            r  = float(self._rng.uniform(_OBS_R_MIN, _OBS_R_MAX))
            cx = float(self._rng.uniform(_X_MIN + r + 0.1, _X_MAX - r - 0.1))
            cy = float(self._rng.uniform(_Y_MIN + r + 0.1, _Y_MAX - r - 0.1))

            sx, sy, _ = _SPAWN
            if math.sqrt((cx - sx)**2 + (cy - sy)**2) < r + spawn_clear:
                continue
            if math.sqrt((cx - self._goal_x)**2 + (cy - self._goal_y)**2) < r + goal_clear:
                continue
            obstacles.append((cx, cy, r))

        if obstacles:
            return np.array(obstacles, dtype=np.float64)
        return np.zeros((0, 3), dtype=np.float64)
