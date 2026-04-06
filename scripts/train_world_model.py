import sys
import os
import wandb
import inspect

# Ensure dreamer4 is accessible
dreamer_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "dreamer4")
sys.path.insert(0, dreamer_dir)

try:
    from dreamer4 import train_dynamics
except ImportError:
    # If the user's dreamer4 submodule does not have an __init__.py at root, we import direct
    sys.path.insert(0, os.path.join(dreamer_dir, "dreamer4"))
    import train_dynamics

original_log_dynamics_eval_wandb = train_dynamics.log_dynamics_eval_wandb

def patched_log_dynamics_eval_wandb(gt, pred, ctx_length, step, tag, max_items=4, gap_px=16):
    """ Wrapped WandB logger fulfilling the Phase 5 requirements """
    # Execute standard DreamerV4 logging
    original_log_dynamics_eval_wandb(
        gt=gt, pred=pred, ctx_length=ctx_length, step=step, 
        tag=tag, max_items=max_items, gap_px=gap_px
    )
    
    # Execute custom Medical RL imagination visualization
    if pred is not None and pred.ndim == 5:
        # pred shape: (Batch, Time, Channels, Height, Width)
        # We assume 3-channel orthogonal views mapping (Axial, Coronal, Sagittal)
        if pred.shape[2] == 3:
            # Extract first sample in batch, first timestep sequence
            reconstructed_tensor = pred[0, 0]
            
            # Map [0, 1] tensor to [0, 255] uint8 ndarray for WandB
            recon_np = (reconstructed_tensor.clamp(0, 1) * 255.0).permute(1, 2, 0).cpu().numpy().astype('uint8')
            
            # Log exact key as instructed in engineering plan
            wandb.log({"World_Model/Reconstruction": wandb.Image(recon_np)}, step=step)

# Apply dynamic patch without modifying core repository
train_dynamics.log_dynamics_eval_wandb = patched_log_dynamics_eval_wandb

def run_main():
    """ Dynamically execute the argument parsing and training entry point from train_dynamics.py """
    source = inspect.getsource(train_dynamics)
    if 'if __name__ == "__main__":' in source:
        main_block = source.split('if __name__ == "__main__":')[1]
        exec(main_block, train_dynamics.__dict__)
    else:
        print("Error: Could not locate main block in train_dynamics.")

if __name__ == "__main__":
    print("Starting Patched World Model Training Loop...")
    run_main()
