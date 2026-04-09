import gymnasium as gym
from gymnasium import spaces
import dm_env
import numpy as np

class MedicalEnv(gym.Env):
    """
    True 3D Medical Environment for Continuous Navigational RL.
    """
    def __init__(self, volume=None, spacing=(1.0, 1.0, 1.0), target_state=None):
        super().__init__()
        # Flattened 3D Tensor interpreted as 64 pseudo-channels 
        self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(64, 64, 64), dtype=np.float32)
        
        # Continuous action space: translations in X, Y, Z tightly bounded in [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        self.volume = volume if volume is not None else np.zeros((128, 128, 128), dtype=np.float32)
        
        # State represents standard Euclidean coordinate tracking
        self.state = np.array([64.0, 64.0, 64.0], dtype=np.float32)
        self.target_state = target_state if target_state is not None else np.array([32.0, 32.0, 32.0], dtype=np.float32)

    def _get_3d_crop(self, center, shape=64):
        """ Extract a continuous 3D block dynamically cropped and zero-padded. """
        half = shape // 2
        out = np.zeros((shape, shape, shape), dtype=np.float32)
        
        c = np.round(center).astype(int)
        
        # Determine global vs localized slice boundaries
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

    def get_observation(self):
        crop_3d = self._get_3d_crop(self.state, shape=64)
        return crop_3d

    def step(self, action):
        old_dist = np.linalg.norm(self.state - self.target_state)
        
        # Action governs translation shift velocities implicitly bounded by [-1, 1]
        self.state += action * 2.0  
        self.state = np.clip(self.state, 0, np.array(self.volume.shape) - 1)
        
        new_dist = np.linalg.norm(self.state - self.target_state)
        reward = float(old_dist - new_dist) 
        
        done = bool(new_dist < 1.0)
        
        return self.get_observation(), reward, done, False, {"distance": new_dist}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(16, np.array(self.volume.shape) - 16, size=3)
        return self.get_observation(), {}

class DreamerV3Wrapper:
    """
    Adapter bridging MedicalEnv's true 3D arrays into DreamerV3's explicit dictionary formats.
    """
    def __init__(self, env: MedicalEnv):
        self._env = env
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def __getattr__(self, name):
        """ Redirect generic calls to the underlying Gym environment. """
        return getattr(self._env, name)

    def _process_obs(self, obs, reward=0.0, is_first=False, is_terminal=False, is_last=False):
        """ Transforms the raw gym step output into the strict DreamerV3 unified dictionary. """
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
            # Intercept action dictionaries strictly mapped from DreamerV3
            action_arr = action['action']
        else:
            action_arr = action
            
        action_arr = np.clip(np.array(action_arr), -1.0, 1.0)
        
        obs, reward, terminated, truncated, _ = self._env.step(action_arr)
        
        done = terminated or truncated
        return self._process_obs(
            obs, 
            reward=reward, 
            is_first=False, 
            is_terminal=terminated, 
            is_last=done
        )
