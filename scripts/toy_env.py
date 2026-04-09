import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ToyMedicalEnv(gym.Env):
    """
    Dummy 3D Environment for hardware validation. 
    State is a 3D underlying array of 128^3, but outputs 64^3 flat observations.
    Reward is explicitly mathematical distance gradient reduction.
    """
    def __init__(self):
        super().__init__()
        # Flattened 3D Tensor array into Dreamer Channel space (64 Layers)
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 64), dtype=np.uint8)
        
        # 3D Valid continuous navigation
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        self.volume_shape = (128, 128, 128)
        self.target_state = np.zeros(3, dtype=np.float32)
        self.state = np.zeros(3, dtype=np.float32)
        self.volume = np.zeros(self.volume_shape, dtype=np.uint8)
        
    def _draw_sphere(self, center, radius=5):
        self.volume.fill(0)
        c = np.round(center).astype(int)
        
        rmin = np.maximum(0, c - radius)
        rmax = np.minimum(np.array(self.volume_shape), c + radius + 1)
        
        x, y, z = np.ogrid[rmin[0]:rmax[0], rmin[1]:rmax[1], rmin[2]:rmax[2]]
        mask = (x - c[0])**2 + (y - c[1])**2 + (z - c[2])**2 <= radius**2
        
        # Stamp pure 255 (Bright Tumor proxy) structure mathematically
        self.volume[rmin[0]:rmax[0], rmin[1]:rmax[1], rmin[2]:rmax[2]][mask] = 255

    def _get_3d_crop(self, center, shape=64):
        """ Native centered slice extraction mathematically safe bound rules """
        half = shape // 2
        out = np.zeros((shape, shape, shape), dtype=np.uint8)
        c = np.round(center).astype(int)
        
        s_glob = np.maximum(0, c - half)
        e_glob = np.minimum(np.array(self.volume.shape), c + half)
        
        s_loc = np.maximum(0, half - c)
        e_loc = s_loc + (e_glob - s_glob)
        
        valid = (s_glob < e_glob).all()
        if valid:
            out[
                s_loc[0]:e_loc[0],
                s_loc[1]:e_loc[1],
                s_loc[2]:e_loc[2]
            ] = self.volume[
                s_glob[0]:e_glob[0],
                s_glob[1]:e_glob[1],
                s_glob[2]:e_glob[2]
            ]
        return out

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        # Place target randomly off-center
        self.target_state = np.random.uniform(32, 96, size=3).astype(np.float32)
        self._draw_sphere(self.target_state, radius=5)
        
        # Exact Center Initialization
        self.state = np.array([64.0, 64.0, 64.0], dtype=np.float32)
        return self._get_3d_crop(self.state), {}

    def step(self, action):
        old_dist = np.linalg.norm(self.state - self.target_state)
        
        # Step scaling explicitly tuned for swift traversal of the 128 bounds
        self.state += action * 5.0
        self.state = np.clip(self.state, 0, np.array(self.volume.shape) - 1)
        
        new_dist = np.linalg.norm(self.state - self.target_state)
        
        # Positive reward if it moves closer natively
        reward = float(old_dist - new_dist)
        
        done = False
        if new_dist < 5.0:
            reward += 10.0 # Huge native bump to solidify termination loop tracking
            done = True
            
        return self._get_3d_crop(self.state), reward, done, False, {"distance": new_dist}


class DreamerV3Wrapper:
    """ Maps to official pure Gym outputs explicit dict translation bindings """
    def __init__(self, env):
        self._env = env
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _process_obs(self, obs, reward=0.0, is_first=False, is_terminal=False, is_last=False):
        return {
            'image': obs,
            'reward': np.array(reward, dtype=np.float32),
            'is_first': np.array(is_first, dtype=bool),
            'is_terminal': np.array(is_terminal, dtype=bool),
            'is_last': np.array(is_last, dtype=bool)
        }

    def reset(self):
        obs, _ = self._env.reset()
        return self._process_obs(obs, reward=0.0, is_first=True, is_terminal=False, is_last=False)

    def step(self, action):
        if isinstance(action, dict):
            action_arr = action['action']
        else:
            action_arr = action
            
        action_arr = np.clip(np.array(action_arr), -1.0, 1.0)
        obs, reward, terminated, truncated, _ = self._env.step(action_arr)
        
        done = terminated or truncated
        return self._process_obs(obs, reward=reward, is_first=False, is_terminal=terminated, is_last=done)
