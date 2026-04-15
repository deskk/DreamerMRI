import os
import json
import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import nibabel as nib
import embodied
import elements

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

GLOBAL_VOLUME_CACHE = {}
GLOBAL_TARGET_CACHE = {}

class RealMedicalEnv(gym.Env):
    def __init__(self, data_dir: str, debug_patient_id=None):
        super().__init__()
        
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 64), dtype=np.uint8)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        self.data_dir = data_dir
        self.debug_patient_id = debug_patient_id
        
        self.patients = [d for d in os.listdir(data_dir) if d.startswith("Breast_MRI_")]
        
        if len(self.patients) == 0:
            raise FileNotFoundError(f"Empty Dataset Volume at {data_dir}")
            
        if len(GLOBAL_VOLUME_CACHE) == 0:
            logging.info("Hydrating Core Dataset Caches statically bounding GPU speeds...")
            for pid in self.patients[:50]: 
                vol_path = os.path.join(data_dir, pid, "subtraction_1mm.nii.gz")
                target_path = os.path.join(data_dir, pid, "target.json")
                
                if os.path.exists(vol_path) and os.path.exists(target_path):
                    vol_data = nib.load(vol_path).get_fdata()
                    # Secure integer casting boundary scaling
                    vol_min, vol_max = vol_data.min(), vol_data.max()
                    if vol_max > vol_min:
                        vol_data = ((vol_data - vol_min) / (vol_max - vol_min) * 255.0)
                    vol_data = vol_data.astype(np.uint8)
                    
                    GLOBAL_VOLUME_CACHE[pid] = vol_data
                    with open(target_path, "r") as f:
                        meta = json.load(f)
                        GLOBAL_TARGET_CACHE[pid] = np.array(meta["target_voxel"], dtype=np.float32)

        self.current_pid = None
        self.volume = None
        self.target_state = np.zeros(3, dtype=np.float32)
        self.state = np.zeros(3, dtype=np.float32)
        
    def _get_3d_crop(self, center, shape=64):
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
            
        valid_keys = list(GLOBAL_VOLUME_CACHE.keys())
        if self.debug_patient_id and self.debug_patient_id in valid_keys:
            self.current_pid = self.debug_patient_id
        else:
            self.current_pid = np.random.choice(valid_keys)
        
        self.volume = GLOBAL_VOLUME_CACHE[self.current_pid]
        self.target_state = GLOBAL_TARGET_CACHE[self.current_pid]
        
        self.state = np.array([self.volume.shape[0]//2, self.volume.shape[1]//2, self.volume.shape[2]//2], dtype=np.float32)
        return self._get_3d_crop(self.state), {}

    def step(self, action):
        old_dist = np.linalg.norm(self.state - self.target_state)
        
        self.state += action * 10.0
        self.state = np.clip(self.state, 0, np.array(self.volume.shape) - 1)
        
        new_dist = np.linalg.norm(self.state - self.target_state)
        reward = float(old_dist - new_dist)
        
        done = False
        if new_dist < 5.0:
            reward += 10.0
            done = True
            
        return self._get_3d_crop(self.state), reward, done, False, {"distance": new_dist}


class DreamerV3Wrapper:
    def __init__(self, env):
        self._env = env
        self.obs_space = {
            'obs_3d': elements.Space(np.uint8, (64, 64, 64)),
            'image': elements.Space(np.uint8, (64, 64, 3)),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool)
        }
        self.act_space = {
            'action': elements.Space(np.float32, (3,), -1.0, 1.0),
            'reset': elements.Space(bool)
        }

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _process_obs(self, obs, reward=0.0, is_first=False, is_terminal=False, is_last=False):
        # Native RGB Center-Slice interception strictly avoiding tensor mismatch for W&B visuals
        center_slice = obs[:, :, obs.shape[2]//2]
        rgb_image = np.stack([center_slice, center_slice, center_slice], axis=-1)
        
        return {
            'obs_3d': obs,
            'image': rgb_image,
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
            if action.get('reset', False):
                return self.reset()
            action_arr = action['action']
        else:
            action_arr = action
            
        action_arr = np.clip(np.array(action_arr), -1.0, 1.0)
        obs, reward, terminated, truncated, _ = self._env.step(action_arr)
        
        done = terminated or truncated
        return self._process_obs(obs, reward=reward, is_first=False, is_terminal=terminated, is_last=done)
