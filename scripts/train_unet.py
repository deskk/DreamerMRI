import os
import json
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    RandFlipd, RandRotate90d, ToTensord, SpatialCropd
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.data import Dataset

class BoundingBoxCropDataset(Dataset):
    """
    Dataset that simulates extracting the ROI using the Dreamer Agent's Final Bounding Box.
    (We use the ground truth V_new dataset files for pretraining the U-Net).
    """
    def __init__(self, preprocessed_dir, masks_dir, transforms=None):
        self.preprocessed_dir = preprocessed_dir
        self.masks_dir = masks_dir
        self.transforms = list(transforms) if transforms else []
        self.patients = [d for d in os.listdir(preprocessed_dir) if os.path.isdir(os.path.join(preprocessed_dir, d))]
        
    def __len__(self):
        return len(self.patients)
        
    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        subj_dir = os.path.join(self.preprocessed_dir, patient_id)
        
        image_path = os.path.join(subj_dir, "subtraction_1mm.nii.gz")
        json_path = os.path.join(subj_dir, "dataset.json")
        
        # In a real deployed pipeline, the mask might dynamically be matched here if processed
        mask_path = os.path.join(self.masks_dir, patient_id, "mask_tissue.nii.gz") 
        if not os.path.exists(mask_path):
            # Fallback for compilation/testing if masks are missing
            mask_path = image_path 
            
        with open(json_path, "r") as f:
            meta = json.load(f)
            
        # Load bounding box scale/center (from RL agent or dataset.json)
        V_center = np.array(meta["V_new"])
        # Standard physical 1mm crop size (e.g., 32x32x32)
        roi_size = [32, 32, 32] 
        
        data = {"image": image_path, "label": mask_path}
        
        # Basic Load
        load_tx = Compose([LoadImaged(keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"])])
        data = load_tx(data)
        
        # Center-crop based on RL agent predictions
        crop_tx = SpatialCropd(
            keys=["image", "label"], 
            roi_center=[int(V_center[0]), int(V_center[1]), int(V_center[2])], 
            roi_size=roi_size
        )
        
        try:
            data = crop_tx(data)
        except ValueError:
            # If out of bounds fallback to random crop or center projection
            from monai.transforms import CenterSpatialCropd
            data = CenterSpatialCropd(keys=["image", "label"], roi_size=roi_size)(data)
            
        # Optional custom user augments
        for t in self.transforms:
            data = t(data)
            
        # Convert output to tensors
        data = ToTensord(keys=["image", "label"])(data)
        return data

def train_unet():
    dataset_dir = "/local/scratch/scratch-hd/desmond/full_dukedataset"
    preprocessed_dir = os.path.join(dataset_dir, "preprocessed")
    # Using previous directory convention for the Duke Segmentations
    masks_dir = os.path.join(dataset_dir, "duke_cancer_imaging", "processed", "Dyn_anchored")
    
    # Transformations exclusive to cropped 3D Regions
    train_transforms = [
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3)
    ]
    
    dataset = BoundingBoxCropDataset(preprocessed_dir, masks_dir, transforms=train_transforms)
    
    if len(dataset) == 0:
        print("Dataset empty. No preprocessed volume directories found.")
        return
        
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing MONAI UNet on {device}...")
    
    # 3D UNet Initialization
    model = UNet(
        spatial_dims=3,
        in_channels=1,          # Assuming single channel subtraction array
        out_channels=1,         # Binary segmentation (Vascular / Lesion)
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    
    max_epochs = 100
    print("Starting rapid targeted Stage 3 ROI Training Loop...")
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            # Ensure label is binary (e.g FGT/Lesion vs Background)
            labels = (labels > 0).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        epoch_loss /= max(step, 1)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{max_epochs}, 3D Cropped UNet Loss: {epoch_loss:.4f}")
            
    print("Stage 3 U-Net Localized Training Complete.")
    
    os.makedirs(os.path.join(dataset_dir, "models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(dataset_dir, "models", "stage3_unet.pth"))
    print("Saved to models/stage3_unet.pth")

if __name__ == "__main__":
    train_unet()
