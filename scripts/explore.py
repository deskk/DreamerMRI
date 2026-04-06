import os
import sys

def main():
    try:
        import pandas as pd
    except ImportError:
        print("Pandas not installed")
        sys.exit(1)

    dataset_dir = "/local/scratch/scratch-hd/desmond/full_dukedataset"
    
    # 1. Read Annotation
    ann_path = os.path.join(dataset_dir, "Annotation_Boxes.xlsx")
    df_ann = pd.read_excel(ann_path)
    print("--- Annotation_Boxes.xlsx ---")
    print("Columns:", df_ann.columns.tolist())
    print("Shape:", df_ann.shape)
    print(df_ann.head(2))
    print()
    
    # 2. Read Mapping
    map_path = os.path.join(dataset_dir, "Breast-Cancer-MRI-filepath_filename-mapping.xlsx")
    df_map = pd.read_excel(map_path)
    print("--- Mapping.xlsx ---")
    print("Columns:", df_map.columns.tolist())
    print("Shape:", df_map.shape)
    print(df_map.head(2))

if __name__ == "__main__":
    main()
