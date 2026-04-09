import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import nibabel as nib
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def solve_affine_trap(raw_nifti_path, iso_nifti_path, patient_data):
    x_min, x_max = patient_data['Start Row'], patient_data['End Row']
    y_min, y_max = patient_data['Start Column'], patient_data['End Column']
    z_min, z_max = patient_data['Start Slice'], patient_data['End Slice']
    
    # Mathematical center in the RAW voxel grid
    center_raw_voxel = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0, 1.0])
    
    # Raw Physical Space
    raw_img = nib.load(raw_nifti_path)
    raw_affine = raw_img.affine
    center_physical_mm = raw_affine @ center_raw_voxel
    
    # Isotropic Inverse Space
    iso_img = nib.load(iso_nifti_path)
    iso_affine = iso_img.affine
    iso_affine_inv = np.linalg.inv(iso_affine)
    
    center_new_voxel = iso_affine_inv @ center_physical_mm
    return center_raw_voxel[:3], center_new_voxel[:3]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/local/scratch/scratch-hd/desmond/full_dukedataset")
    args = parser.parse_args()

    preprocessed_dir = os.path.join(args.dataset_dir, "preprocessed")
    temp_dir = os.path.join(preprocessed_dir, "temp")
    csv_path = os.path.join(args.dataset_dir, "Annotation_Boxes.xlsx")
    
    df = pd.read_excel(csv_path)
    
    patients = [d for d in os.listdir(preprocessed_dir) if d.startswith("Breast_MRI_")]
    logging.info(f"Targeting {len(patients)} Patients in mapped isolate directory.")
    
    for pid in patients:
        raw_nifti_path = os.path.join(temp_dir, pid, "pre.nii.gz")
        iso_nifti_path = os.path.join(preprocessed_dir, pid, "subtraction_1mm.nii.gz")
        
        if not os.path.exists(raw_nifti_path) or not os.path.exists(iso_nifti_path):
            logging.warning(f"[{pid}] Missing raw/iso volume paths. Skipping...")
            continue
            
        p_data = df[df['Patient ID'] == pid]
        if len(p_data) == 0:
            continue
            
        old_vox, new_vox = solve_affine_trap(raw_nifti_path, iso_nifti_path, p_data.iloc[0])
        
        print(f"[{pid}] Before Affine (Original CSV Index): {old_vox.tolist()}")
        print(f"[{pid}] After Affine (New Iso-Voxel Map): {new_vox.tolist()}")
        print("-" * 50)
        
        target_json = os.path.join(preprocessed_dir, pid, "target.json")
        with open(target_json, "w") as f:
            json.dump({"target_voxel": new_vox.tolist()}, f)

if __name__ == "__main__":
    main()
