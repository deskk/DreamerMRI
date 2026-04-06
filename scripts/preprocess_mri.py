import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
import monai.transforms as mt
import SimpleITK as sitk

def list_subdirectories(path):
    return [d for d in Path(path).iterdir() if d.is_dir()]

def find_series_folders(subject_dir):
    """
    Find T1, Pre-contrast (Dyn), and Post-contrast series folders from Duke DICOM tree.
    (Adapted from prior user scripts).
    """
    subject_dir = Path(subject_dir)
    date_dirs = list_subdirectories(subject_dir)
    if not date_dirs:
        return None, None, None
        
    study_dir = date_dirs[0]
    series_dirs = list_subdirectories(study_dir)
    
    def get_series_num(p):
        try:
            return float(p.name.split('-')[0])
        except:
            return 9999.0
            
    sorted_series = sorted(series_dirs, key=get_series_num)
    
    t1_dir = None
    dyn_candidates = []
    
    for s_dir in sorted_series:
        name = s_dir.name.lower()
        if ("t1" in name or "ideal" in name) and "dyn" not in name and "sub" not in name:
            if not t1_dir: t1_dir = s_dir
        if "sub" in name:
            continue
        if "dyn" in name or "vibrant" in name or "multiphase" in name or "dynamic" in name:
             dyn_candidates.append(s_dir)
             
    dyn_dir = None
    post_dir = None
    if dyn_candidates:
        dyn_dir = dyn_candidates[0]
        remaining = dyn_candidates[1:]
        for s_dir in remaining:
            name = s_dir.name.lower()
            if "ph1" in name or "1st" in name:
                post_dir = s_dir
                break
            if "ph2" in name or "2nd" in name:
                 if not post_dir: post_dir = s_dir
        if not post_dir and remaining:
            post_dir = remaining[0]
            
    return t1_dir, dyn_dir, post_dir

def load_dicom_to_nifti_temp(dicom_dir, out_path):
    """ Loads DICOM series robustly using SimpleITK, saves to NIfTI for nibabel/monai ingestion """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not dicom_names:
        return False
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, str(out_path))
    return True

def process_patient(patient_id, df_ann, dataset_dir, out_dir):
    print(f"Processing {patient_id}...")
    subj_dir = os.path.join(dataset_dir, "duke_breast_cancer_mri", patient_id)
    if not os.path.exists(subj_dir):
        print(f"  Missing directory for {patient_id}")
        return False
        
    t1_dir, dyn_dir, post_dir = find_series_folders(subj_dir)
    if not dyn_dir or not post_dir:
        print(f"  Missing Pre/Post sequence for {patient_id}")
        return False
        
    temp_dir = os.path.join(out_dir, "temp", patient_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    pre_path = os.path.join(temp_dir, "pre.nii.gz")
    post_path = os.path.join(temp_dir, "post.nii.gz")
    
    if not load_dicom_to_nifti_temp(dyn_dir, pre_path): return False
    if not load_dicom_to_nifti_temp(post_dir, post_path): return False
    
    # --- The Affine Coordinate Map ---
    # Read the DICOM/NIfTI affine matrix M_raw using nibabel
    pre_nii = nib.load(pre_path)
    post_nii = nib.load(post_path)
    
    M_raw = pre_nii.affine
    pre_data = pre_nii.get_fdata(dtype=np.float32)
    post_data = post_nii.get_fdata(dtype=np.float32)
    
    # Volumetric Subtraction
    if pre_data.shape != post_data.shape:
        print(f"  Shape mismatch {pre_data.shape} vs {post_data.shape}")
        # Simplistic fix if shapes mismatch - truncate
        min_shape = np.minimum(pre_data.shape, post_data.shape)
        pre_data = pre_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        post_data = post_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        
    sub_data = post_data - pre_data
    
    # Bounding Box Read
    ann_row = df_ann[df_ann["Patient ID"] == patient_id]
    if len(ann_row) == 0:
        print(f"  No bounding box in Annotation_Boxes.xlsx for {patient_id}")
        return False
        
    ann_row = ann_row.iloc[0]
    # Duke convention: End Row / End Col / End Slice. Voxel center:
    vx = (ann_row["Start Row"] + ann_row["End Row"]) / 2.0
    vy = (ann_row["Start Column"] + ann_row["End Column"]) / 2.0
    vz = (ann_row["Start Slice"] + ann_row["End Slice"]) / 2.0
    V_raw = np.array([vx, vy, vz])
    
    # Convert voxel to physical world coordinates: P = M_raw * [V_raw, 1]^T
    V_raw_homog = np.array([V_raw[0], V_raw[1], V_raw[2], 1.0])
    P = M_raw @ V_raw_homog
    
    # --- MONAI Resampling ---
    # Apply monai.transforms.Spacingd to resample volume to 1x1x1 mm
    # Since Spacingd expects channel first: (C, H, W, D)
    sub_data_ch = np.expand_dims(sub_data, axis=0)
    data_dict = {"image": sub_data_ch}
    
    # We must provide MONAI with the original affine in a MetaTensor or dictionary
    from monai.data import MetaTensor
    img_tensor = MetaTensor(sub_data_ch, affine=M_raw)
    data_dict = {"image": img_tensor}
    
    spacing_tx = mt.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear")
    resampled_dict = spacing_tx(data_dict)
    
    resampled_img = resampled_dict["image"]
    M_new = resampled_img.affine
    
    # Reverse Mapping (Do this AFTER resampling)
    # Calculate new ground truth voxel coordinates: V_new = M_new^-1 * [P, 1]^T
    M_new_inv = np.linalg.inv(M_new)
    V_new_homog = M_new_inv @ np.array([P[0], P[1], P[2], 1.0])
    V_new = V_new_homog[:3]
    
    # Save Output
    subj_out_dir = os.path.join(out_dir, patient_id)
    os.makedirs(subj_out_dir, exist_ok=True)
    
    out_nifti_path = os.path.join(subj_out_dir, "subtraction_1mm.nii.gz")
    # Save the NIfTI array
    out_nii = nib.Nifti1Image(resampled_img.numpy()[0], affine=M_new)
    nib.save(out_nii, out_nifti_path)
    
    # Export dataset.json
    dataset_info = {
        "patient_id": patient_id,
        "V_raw": V_raw.tolist(),
        "P_world": P[:3].tolist(),
        "V_new": V_new.tolist(),
    }
    with open(os.path.join(subj_out_dir, "dataset.json"), "w") as f:
        json.dump(dataset_info, f, indent=4)
        
    print(f"  Successfully processed {patient_id}. V_new: {V_new}")
    return True

def main():
    dataset_dir = "/local/scratch/scratch-hd/desmond/full_dukedataset"
    out_dir = os.path.join(dataset_dir, "preprocessed")
    os.makedirs(out_dir, exist_ok=True)
    
    ann_path = os.path.join(dataset_dir, "Annotation_Boxes.xlsx")
    df_ann = pd.read_excel(ann_path)
    
    patients = [d.name for d in Path(os.path.join(dataset_dir, "duke_breast_cancer_mri")).iterdir() if d.is_dir() and d.name.startswith("Breast_MRI")]
    patients = sorted(patients)
    
    print(f"Found {len(patients)} patients. Starting pipeline...")
    # For testing, just run on first few patients initially to verify
    for p in patients[:2]:  
        process_patient(p, df_ann, dataset_dir, out_dir)

if __name__ == "__main__":
    main()
