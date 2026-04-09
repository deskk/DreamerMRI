import os
import shutil
import pandas as pd

def isolate_subset():
    src_dir = "/local/scratch/scratch-hd/desmond/full_dukedataset"
    dst_dir = "/local/scratch/scratch-hd/desmond/duke_micro_subset"
    
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(os.path.join(dst_dir, "preprocessed"), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, "preprocessed", "temp"), exist_ok=True)
    
    # 1. Isolate Clinical Array Mapping
    csv_src = os.path.join(src_dir, "Annotation_Boxes.xlsx")
    csv_dst = os.path.join(dst_dir, "Annotation_Boxes.xlsx")
    if os.path.exists(csv_src):
        shutil.copy(csv_src, csv_dst)
        print("Copied Annotation_Boxes.xlsx successfully.")
    
    # 2. Extract deterministic patient batch
    preprocessed_src = os.path.join(src_dir, "preprocessed")
    if not os.path.exists(preprocessed_src):
        print(f"Error: {preprocessed_src} missing.")
        return
        
    patients = [d for d in os.listdir(preprocessed_src) if d.startswith("Breast_MRI_")]
    patients = sorted(patients)[:5]
    
    print(f"Isolating {len(patients)} sequential patients: {patients}")
    for p in patients:
        src_iso = os.path.join(preprocessed_src, p)
        dst_iso = os.path.join(dst_dir, "preprocessed", p)
        if os.path.exists(src_iso):
            shutil.copytree(src_iso, dst_iso, dirs_exist_ok=True)
            print(f"[{p}] Processed 1mm ISO Volume Copied.")
            
        src_raw = os.path.join(preprocessed_src, "temp", p)
        dst_raw = os.path.join(dst_dir, "preprocessed", "temp", p)
        if os.path.exists(src_raw):
            shutil.copytree(src_raw, dst_raw, dirs_exist_ok=True)
            print(f"[{p}] Raw Affine Metadata Copied.")

if __name__ == "__main__":
    isolate_subset()
