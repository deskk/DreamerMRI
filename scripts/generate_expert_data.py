import os
import json
import uuid
import numpy as np
import pandas as pd
import nibabel as nib
import io
import sys

# Assume env.py is in the same directory or project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# For our purposes we can just import MedicalEnv from env handling
try:
    from env import MedicalEnv, DreamerWrapper
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env import MedicalEnv, DreamerWrapper

def compute_target_scale(df_ann, patient_id):
    """ Read the original bounding box and assume Physical dimensions stay constant (as 1mm iso resampled) """
    ann_row = df_ann[df_ann["Patient ID"] == patient_id]
    if len(ann_row) == 0: return np.array([10.0, 10.0, 10.0])
    ann_row = ann_row.iloc[0]
    
    # We'll just assume original spacing was isotropic or we approximate physical size 
    # to be end - start because spacing is now 1mm.
    # Technically we should multiply by original spacing but this is functionally analogous.
    w = abs(ann_row["End Row"] - ann_row["Start Row"])
    h = abs(ann_row["End Column"] - ann_row["Start Column"])
    d = abs(ann_row["End Slice"] - ann_row["Start Slice"])
    return np.array([float(w), float(h), float(d)])

def compute_best_action(current_state, target_state):
    """
    Returns the action [0-11] that most efficiently minimizes the distance to target_state.
    current_state: [x, y, z, w, h, d]
    target_state: [Vx, Vy, Vz, tw, th, td]
    """
    distances = np.abs(current_state - target_state)
    max_err_dim = np.argmax(distances) # The dimension with the biggest error
    
    # Threshold for deciding "we are close enough"
    if distances[max_err_dim] < 1.0:
        return None # Target reached!
        
    diff = target_state[max_err_dim] - current_state[max_err_dim]
    
    # Map index to action
    # 0: +x, 1: -x, 2: +y, 3: -y, 4: +z, 5: -z
    # 6: +w, 7: -w, 8: +h, 9: -h, 10: +d, 11: -d
    
    action_idx = max_err_dim * 2 
    if diff < 0:
        action_idx += 1 # Negative direction (-x, -y, etc)
        
    return action_idx

def save_episode(directory, episode_data):
    """
    Save the episode to disk in the npz format Nicklas Hansen's DreamerV4 expects.
    Keys: image (uint8 / float32), action (int), reward (float32), discount (float32), 
          is_first (bool), is_last (bool), is_terminal (bool)
    We pad dictionaries to align shapes.
    """
    os.makedirs(directory, exist_ok=True)
    length = len(episode_data['reward'])
    filename = f"{uuid.uuid4().hex}_{length}.npz"
    filepath = os.path.join(directory, filename)
    
    # Convert lists to packed numpy arrays
    np_ep = {
        'image': np.array(episode_data['image'], dtype=np.float32),
        'action': np.array(episode_data['action'], dtype=np.int32),
        'reward': np.array(episode_data['reward'], dtype=np.float32),
        'discount': np.array(episode_data['discount'], dtype=np.float32),
        'is_first': np.array(episode_data['is_first'], dtype=bool),
        'is_last': np.array(episode_data['is_last'], dtype=bool),
        'is_terminal': np.array(episode_data['is_terminal'], dtype=bool)
    }
    
    # Usually Dreamer expects a dummy action at step 0 because action t occurs AFTER obs t
    # For robust alignment matching the episode length identically,
    # let's just save via savez_compressed
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **np_ep)
        f1.seek(0)
        with open(filepath, 'wb') as f2:
            f2.write(f1.read())

def generate_patient_trajectories(patient_id, df_ann, dataset_dir, out_dir):
    subj_dir = os.path.join(dataset_dir, "preprocessed", patient_id)
    nifti_path = os.path.join(subj_dir, "subtraction_1mm.nii.gz")
    json_path = os.path.join(subj_dir, "dataset.json")
    
    if not os.path.exists(nifti_path) or not os.path.exists(json_path):
        return
        
    with open(json_path, "r") as f:
        meta = json.load(f)
        
    vol = nib.load(nifti_path)
    data = vol.get_fdata()
    
    env_base = MedicalEnv(volume=data, affine=vol.affine)
    env = DreamerWrapper(env_base)
    
    V_new = np.array(meta["V_new"])
    target_scale = compute_target_scale(df_ann, patient_id)
    target_state = np.concatenate([V_new, target_scale])
    
    # Generate 5 random trajectories for this patient
    for _ in range(5):
        timestep = env.reset()
        
        episode = {
            'image': [timestep.observation['image']],
            'action': [0], # Dummy first action
            'reward': [0.0],
            'discount': [1.0],
            'is_first': [True],
            'is_last': [False],
            'is_terminal': [False]
        }
        
        # Expert Loop
        done = False
        steps = 0
        while not done and steps < 300: # limit to 300 steps
            current_state = env_base.state
            action = compute_best_action(current_state, target_state)
            
            if action is None:
                # Target precisely reached!
                done = True
                episode['is_last'][-1] = True
                episode['is_terminal'][-1] = True
                break
                
            # Act
            timestep = env.step(action)
            obs = timestep.observation['image']
            
            # Simple reward for expert: +1 for moving closer, though expert action inherently is optimal
            # We record a generic reward
            reward = 0.1 
            
            episode['image'].append(obs)
            episode['action'].append(action)
            episode['reward'].append(reward)
            episode['discount'].append(timestep.discount)
            episode['is_first'].append(False)
            episode['is_last'].append(False)
            episode['is_terminal'].append(False)
            
            steps += 1
            
        print(f"[{patient_id}] Trajectory reached target in {steps} steps.")
        save_episode(out_dir, episode)

def main():
    dataset_dir = "/local/scratch/scratch-hd/desmond/full_dukedataset"
    out_dir = os.path.join(dataset_dir, "offline_replay_buffer")
    ann_path = os.path.join(dataset_dir, "Annotation_Boxes.xlsx")
    
    if not os.path.exists(ann_path):
        print("Annotation file not found!")
        return
        
    df_ann = pd.read_excel(ann_path)
    
    preprocessed_dir = os.path.join(dataset_dir, "preprocessed")
    if not os.path.exists(preprocessed_dir):
        print("Preprocessed dir not found.")
        return
        
    patients = [d for d in os.listdir(preprocessed_dir) if os.path.isdir(os.path.join(preprocessed_dir, d))]
    
    print(f"Generating expert data for {len(patients)} patients...")
    for p in patients:
        generate_patient_trajectories(p, df_ann, dataset_dir, out_dir)
        
    print("Expert data generation complete!")

if __name__ == "__main__":
    main()
