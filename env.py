import gymnasium as gym
from gymnasium import spaces
import dm_env
from dm_env import specs
import numpy as np

class MedicalEnv(gym.Env):
    """
    Medical Environment for 2.5D MRI navigation.
    """
    def __init__(self, volume=None, spacing=(1.0, 1.0, 1.0), affine=None, target_state=None):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3, 64, 64), dtype=np.float32)
        self.action_space = spaces.Discrete(12)
        
        self.volume = volume if volume is not None else np.zeros((128, 128, 128))
        self.affine = affine if affine is not None else np.eye(4)
        
        self.state = np.array([64.0, 64.0, 64.0, 10.0, 10.0, 10.0])
        self.target_state = target_state if target_state is not None else np.array([64.0, 64.0, 64.0, 10.0, 10.0, 10.0])
        self.alpha = 0.0 # Curriculum parameter t
        
        import nibabel as nib
        self.ornt = nib.io_orientation(self.affine)
        
        self.axis_sagittal = np.where(self.ornt[:, 0] == 0)[0][0]
        self.axis_coronal  = np.where(self.ornt[:, 0] == 1)[0][0]
        self.axis_axial    = np.where(self.ornt[:, 0] == 2)[0][0]

    def set_alpha(self, alpha):
        self.alpha = np.clip(alpha, 0.0, 1.0)

    def calculate_iou(self, boxA, boxB):
        minA = boxA[:3] - boxA[3:] / 2.0
        maxA = boxA[:3] + boxA[3:] / 2.0
        minB = boxB[:3] - boxB[3:] / 2.0
        maxB = boxB[:3] + boxB[3:] / 2.0
        
        inter_min = np.maximum(minA, minB)
        inter_max = np.minimum(maxA, maxB)
        
        inter_dims = np.maximum(inter_max - inter_min, 0.0)
        inter_vol = inter_dims[0] * inter_dims[1] * inter_dims[2]
        
        volA = np.prod(boxA[3:])
        volB = np.prod(boxB[3:])
        
        iou = inter_vol / (volA + volB - inter_vol + 1e-6)
        return iou

    def _get_slice(self, axis, coord, shape=64):
        idx = int(np.round(coord))
        idx = max(0, min(self.volume.shape[axis] - 1, idx))
        
        slices = [slice(None)] * 3
        slices[axis] = idx
        plane2d = self.volume[tuple(slices)]
        
        rem_axes = [i for i in range(3) if i != axis]
        c1 = int(np.round(self.state[rem_axes[0]]))
        c2 = int(np.round(self.state[rem_axes[1]]))
        
        half = shape // 2
        out = np.zeros((shape, shape), dtype=np.float32)
        
        start1 = c1 - half
        end1 = c1 + half
        start2 = c2 - half
        end2 = c2 + half
        
        v_start1 = max(0, start1)
        v_end1 = min(plane2d.shape[0], end1)
        v_start2 = max(0, start2)
        v_end2 = min(plane2d.shape[1], end2)
        
        o_start1 = v_start1 - start1
        o_end1 = shape - (end1 - v_end1)
        o_start2 = v_start2 - start2
        o_end2 = shape - (end2 - v_end2)
        
        if v_start1 < v_end1 and v_start2 < v_end2:
            out[o_start1:o_end1, o_start2:o_end2] = plane2d[v_start1:v_end1, v_start2:v_end2]
            
        return out

    def get_observation(self):
        ax_plane = self._get_slice(self.axis_axial, self.state[self.axis_axial], 64)
        cor_plane = self._get_slice(self.axis_coronal, self.state[self.axis_coronal], 64)
        sag_plane = self._get_slice(self.axis_sagittal, self.state[self.axis_sagittal], 64)
        
        obs = np.stack([ax_plane, cor_plane, sag_plane], axis=0)
        return obs

    def step(self, action):
        old_dist = np.linalg.norm(self.state[:3] - self.target_state[:3])
        old_iou = self.calculate_iou(self.state, self.target_state)

        step_size = 1.0 
        scale_step = 1.0
        
        if action == 0: self.state[0] += step_size
        elif action == 1: self.state[0] -= step_size
        elif action == 2: self.state[1] += step_size
        elif action == 3: self.state[1] -= step_size
        elif action == 4: self.state[2] += step_size
        elif action == 5: self.state[2] -= step_size
        elif action == 6: self.state[3] += scale_step
        elif action == 7: self.state[3] -= scale_step
        elif action == 8: self.state[4] += scale_step
        elif action == 9: self.state[4] -= scale_step
        elif action == 10: self.state[5] += scale_step
        elif action == 11: self.state[5] -= scale_step
        
        self.state[:3] = np.clip(self.state[:3], 0, np.array(self.volume.shape) - 1)
        self.state[3:] = np.clip(self.state[3:], 1, 100) 
        
        new_dist = np.linalg.norm(self.state[:3] - self.target_state[:3])
        new_iou = self.calculate_iou(self.state, self.target_state)
        
        delta_dist = old_dist - new_dist  
        delta_iou = new_iou - old_iou     
        
        reward = (1.0 - self.alpha) * float(delta_dist) + self.alpha * float(delta_iou)
        
        obs = self.get_observation()
        done = False
        if new_dist < 1.0 and new_iou > 0.90:
            done = True
            
        info = {'state': self.state.copy(), 'distance': new_dist, 'iou': new_iou}
        
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state[:3] = np.random.uniform(10, np.array(self.volume.shape) - 10)
        self.state[3:] = np.array([10.0, 10.0, 10.0])
        return self.get_observation(), {}

class DreamerWrapper(dm_env.Environment):
    """
    Wraps MedicalEnv to map observation and discrete action spaces
    exactly as external/dreamer4 expects.
    Dreamer typically expects dictionary observations.
    """
    def __init__(self, env: gym.Env):
        self._env = env

    def reset(self) -> dm_env.TimeStep:
        obs, info = self._env.reset()
        obs_dict = {'image': obs}
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=obs_dict
        )

    def step(self, action) -> dm_env.TimeStep:
        # Action from Dreamer will likely be a 1D vector like [0, 0, 1, 0...] or scalar depending on discrete implementation
        if isinstance(action, dict):
            act = action.get('action') 
        elif isinstance(action, np.ndarray) and action.size > 1:
            act = np.argmax(action)
        else:
            act = int(action)
            
        obs, reward, done, truncated, info = self._env.step(act)
        obs_dict = {'image': obs}
        
        if done or truncated:
            step_type = dm_env.StepType.LAST
            discount = 0.0
        else:
            step_type = dm_env.StepType.MID
            discount = 1.0
            
        return dm_env.TimeStep(
            step_type=step_type,
            reward=float(reward),
            discount=discount,
            observation=obs_dict
        )

    def observation_spec(self):
        shape = self._env.observation_space.shape
        return {'image': specs.BoundedArray(shape=shape, dtype=np.float32, 
                                            minimum=self._env.observation_space.low.min(), 
                                            maximum=self._env.observation_space.high.max(), 
                                            name='image')}

    def action_spec(self):
        # dm_env format for discrete 12 actions (typically BoundedArray of format scalar integer)
        # However Dreamer specifically expects actions as categorical
        return specs.DiscreteArray(num_values=12, name='action')
