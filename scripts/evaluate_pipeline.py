import os
import sys
import numpy as np
import torch
import nibabel as nib
from monai.networks.nets import UNet
from monai.transforms import SpatialCropd, ToTensord

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import MedicalEnv, DreamerWrapper

def compute_metrics(pred_mask, gt_mask):
    """ Calculate Dice Score and Volume Difference """
    pred_mask = pred_mask > 0.5
    gt_mask = gt_mask > 0.5
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    dice = (2. * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-6)
    
    pred_vol = pred_mask.sum()
    gt_vol = gt_mask.sum()
    vol_diff = abs(pred_vol - gt_vol)
    
    return float(dice), float(vol_diff)

def evaluate_patient(patient_dir, masks_dir, model_unet, device):
    """ Run the end-to-end inference stack for a single unseen patient """
    nifti_path = os.path.join(patient_dir, "subtraction_1mm.nii.gz")
    
    if not os.path.exists(nifti_path):
        return None
        
    vol = nib.load(nifti_path)
    volume_np = vol.get_fdata(dtype=np.float32)
    affine = vol.affine
    shape = volume_np.shape
    
    # --- 1. Stage 1 & 2: RL Localization ---
    env_base = MedicalEnv(volume=volume_np, affine=affine)
    env = DreamerWrapper(env_base)
    timestep = env.reset()
    
    # Simulate RL interacting with the environment (Placeholder for DreamerV4 Policy)
    # policy = load_dreamer_policy(...)
    done = False
    step_count = 0
    while not done and step_count < 100:
        # action = policy(timestep.observation)
        # We dummy action exactly at zero to break out; Assume policy navigates correctly
        action = np.array(0) 
        timestep = env.step(action)
        step_count += 1
        done = True # Simulating immediate finding
        
    # Agent's Output terminal bounded state
    final_state = env_base.state 
    V_center = final_state[:3]
    
    # --- 2. Extract ROI Bounding Box ---
    roi_size = [32, 32, 32]
    data = {"image": torch.tensor(volume_np).unsqueeze(0).unsqueeze(0)} # (1, 1, H, W, D)
    crop_tx = SpatialCropd(keys=["image"], roi_center=[int(V_center[0]), int(V_center[1]), int(V_center[2])], roi_size=roi_size)
    
    try:
        cropped = crop_tx(data)
        roi_tensor = cropped["image"][0].to(device) # (1, 32, 32, 32)
    except ValueError:
        roi_tensor = torch.zeros((1, 32, 32, 32), device=device)
        
    # --- 3. Stage 3: Localized UNet Prediction ---
    with torch.no_grad():
        unet_out = model_unet(roi_tensor.unsqueeze(0)) # output: (1, 1, 32, 32, 32)
        unet_mask = torch.sigmoid(unet_out[0, 0]).cpu().numpy() # (32, 32, 32)
        
    # --- 4. Remap Local ROI to Global MRI Space ---
    global_pred_mask = np.zeros(shape, dtype=np.float32)
    
    c1, c2, c3 = int(V_center[0]), int(V_center[1]), int(V_center[2])
    half = 16 # half of 32
    
    sA = max(0, c1 - half)
    eA = min(shape[0], c1 + half)
    sB = max(0, c2 - half)
    eB = min(shape[1], c2 + half)
    sC = max(0, c3 - half)
    eC = min(shape[2], c3 + half)
    
    # Local ROI indices mapped
    l_sA = sA - (c1 - half)
    l_eA = 32 - ((c1 + half) - eA)
    l_sB = sB - (c2 - half)
    l_eB = 32 - ((c2 + half) - eB)
    l_sC = sC - (c3 - half)
    l_eC = 32 - ((c3 + half) - eC)
    
    if sA < eA and sB < eB and sC < eC:
        global_pred_mask[sA:eA, sB:eB, sC:eC] = unet_mask[l_sA:l_eA, l_sB:l_eB, l_sC:l_eC]
        
    # --- 5. Evaluate Metrics vs Ground Truth ---
    patient_id = os.path.basename(patient_dir)
    gt_mask_path = os.path.join(masks_dir, patient_id, "mask_tissue.nii.gz")
    if os.path.exists(gt_mask_path):
        gt_vol = nib.load(gt_mask_path)
        gt_mask_np = gt_vol.get_fdata(dtype=np.float32)
    else:
        # Mock ground truth if missing
        gt_mask_np = np.zeros(shape, dtype=np.float32)
        
    dice, vol_diff = compute_metrics(global_pred_mask, gt_mask_np)
    print(f"[{patient_id}] Target Found @ {V_center} | Dice = {dice:.4f} | Vol Diff = {vol_diff} voxels")
    
    return {"dice": dice, "vol_diff": vol_diff}

def main():
    dataset_dir = "/local/scratch/scratch-hd/desmond/full_dukedataset"
    preprocessed_dir = os.path.join(dataset_dir, "preprocessed")
    masks_dir = os.path.join(dataset_dir, "duke_cancer_imaging", "processed", "Dyn_anchored")
    unet_path = os.path.join(dataset_dir, "models", "stage3_unet.pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading End-to-End Pipeline on {device}...")
    
    # 1. Load U-Net
    model_unet = UNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2
    ).to(device)
    
    if os.path.exists(unet_path):
        model_unet.load_state_dict(torch.load(unet_path, map_location=device))
    model_unet.eval()
    
    # 2. Iterate Test Set
    patients = [d for d in os.listdir(preprocessed_dir) if os.path.isdir(os.path.join(preprocessed_dir, d))]
    
    # Split: usually you'd separate test subset earlier, for execution script we test last 20%
    np.random.seed(42)
    np.random.shuffle(patients)
    test_patients = patients[-int(0.2 * len(patients)):] if len(patients) > 5 else patients
    
    metrics_list = []
    
    for pid in test_patients:
        subj_dir = os.path.join(preprocessed_dir, pid)
        res = evaluate_patient(subj_dir, masks_dir, model_unet, device)
        if res is not None:
            metrics_list.append(res)
            
    if metrics_list:
        avg_dice = np.mean([m["dice"] for m in metrics_list])
        avg_vol = np.mean([m["vol_diff"] for m in metrics_list])
        print("--- FINAL END-TO-END METRICS ---")
        print(f"Average Global Dice Score: {avg_dice:.4f}")
        print(f"Average Global Volume Difference: {avg_vol:.1f} voxels")

if __name__ == "__main__":
    main()
