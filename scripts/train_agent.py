import os
import sys
import json
import numpy as np
import nibabel as nib

# Add repo to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import MedicalEnv, DreamerWrapper

class MultiPatientMedicalEnv(MedicalEnv):
    """
    Multi-Patient Environment that dynamically hot-swaps patient MRI volumes
    at each reset. It implicitly controls the Alpha curriculum parameter globally.
    """
    def __init__(self, dataset_dir, max_steps=500000):
        # Initialize generic empty env; will be populated dynamically on first reset
        super().__init__()
        self.dataset_dir = dataset_dir
        self.preprocessed_dir = os.path.join(dataset_dir, "preprocessed")
        self.patients = [d for d in os.listdir(self.preprocessed_dir) if os.path.isdir(os.path.join(self.preprocessed_dir, d))]
        if not self.patients:
            raise ValueError("No preprocessed patients found.")
            
        self.max_training_steps = max_steps
        self.global_step = 0
        self.current_patient = None
        
    def reset(self, seed=None, options=None):
        # 1. Pick a random patient across the preprocessed Duke dataset
        self.current_patient = np.random.choice(self.patients)
        
        # 2. Load the patient data dynamically to save RAM
        subj_dir = os.path.join(self.preprocessed_dir, self.current_patient)
        nifti_path = os.path.join(subj_dir, "subtraction_1mm.nii.gz")
        json_path = os.path.join(subj_dir, "dataset.json")
        
        vol = nib.load(nifti_path)
        self.volume = vol.get_fdata(dtype=np.float32)
        self.affine = vol.affine
        
        # Re-calc orientation parameters
        self.ornt = nib.io_orientation(self.affine)
        self.axis_sagittal = np.where(self.ornt[:, 0] == 0)[0][0]
        self.axis_coronal  = np.where(self.ornt[:, 0] == 1)[0][0]
        self.axis_axial    = np.where(self.ornt[:, 0] == 2)[0][0]
        
        with open(json_path, "r") as f:
            meta = json.load(f)
            
        V_new = np.array(meta["V_new"])
        
        # Simplified physical scale target (approx 10mm for bounding boxes if not explicitly mapped previously)
        target_scale = np.array([10.0, 10.0, 10.0]) 
        self.target_state = np.concatenate([V_new, target_scale])
        
        return super().reset(seed=seed, options=options)
        
    def step(self, action):
        self.global_step += 1
        
        # --- Curriculum Logic (Alpha mathematically tracks global execution progress) ---
        alpha_t = min(1.0, self.global_step / self.max_training_steps)
        self.set_alpha(alpha_t)
        
        return super().step(action)

def main():
    print("Initializing Multi-Patient Curriculum Environment...")
    dataset_dir = "/local/scratch/scratch-hd/desmond/full_dukedataset"
    
    base_env = MultiPatientMedicalEnv(dataset_dir, max_steps=500000)
    env = DreamerWrapper(base_env)
    
    # Initialize the first state (triggers the first patient load)
    timestep = env.reset()
    
    print(f"Environment successfully initialized with Curriculum Reward. (Alpha parameter active)")
    print(f"Current Loaded Patient: {base_env.current_patient}")
    print("Action Spec:", env.action_spec())
    print("Observation Spec:", env.observation_spec())
    
    # Pass `env` directly into DreamerV4's RL interactive module!
    print("Ready to commence Stage 6 DreamerV4 RL Training.")

if __name__ == "__main__":
    main()
