# True 3D Medical Continuous RL (DreamerV3)

This repository encompasses a fully migrated, state-of-the-art Continuous Reinforcement Learning pipeline orchestrating autonomous Active Vision for 3D MRI localized tumor segmentation. 

We utilize Danijar Hafner's official `DreamerV3` decoupled architecture tightly bound via `jax[cuda12]`. We mathematically flatten the True 3D $64 \times 64 \times 64$ Breast MRI isotropic volumes strictly along the token channels dimension. This elegantly tricks massive sequence models (RSSM) into securely resolving massive 3-dimensional Cartesian spaces natively within hardware Tensor Cores without resorting to deprecated 2.5D mapping strategies.

## Core Pipeline Architecture

* **Environment Integration**: `RealMedicalEnv` implements array caching directly into RAM, delivering 64³ patient matrices at explicit millisecond latencies, averting standard disk I/O network storage block bottlenecks entirely.
* **Affine Coordinate Map**: Built-in spatial affine matrix inversions natively translate un-resampled voxel array tuples directly into absolute $1\text{mm}^3$ index geometry boundaries.
* **Dual-Telemetry Isolation**: Native `wandb.init(sync_tensorboard=True)` wraps standard Tensorboard outputs. We explicitly map the dense True `obs_3d` arrays to bypass typical RGB $(H, W, 3)$ Video Logging recursively, solving framework-level 64-Channel XLA broadcast tracebacks while retaining full fidelity.

---

## Environment Setup & Versioning

1. Ensure your A6000/A100 compute instances share the correct 3.10 boundaries:
```bash
conda create -n dreamer_mri python=3.10
conda activate dreamer_mri
```

2. Establish native dependency baselines. **CRITICAL:** JAX bindings must map onto CUDA 12 to resolve the Dreamerv3 pipeline securely!
```bash
# Explicit JAX bindings bound to CUDA 12 hardware paths
pip install -U "jax[cuda12]"

# Pipeline execution extensions
pip install gymnasium wandb tensorboard pandas openpyxl nibabel
   
# Core internal framework instantiation
pip install -r external/dreamerv3/requirements.txt
```

---

## Quickstart Usage (N=1 Overfitting Constraints)

### 1. N=5 Micro-Subset Isolation
Quickly extract an initial deterministical subset array preventing massive execution locks sweeping over the monolithic 922-patient cluster network drive.
```bash
python scripts/isolate_micro_subset.py
```

### 2. Default Isotropic Preprocessing & Affine Verification
Ensure raw clinical CSV coordinates safely translate into the normalized $1\text{mm}^3$ resampled geometries computationally. *Verify the printed Affine metrics!*
```bash
python scripts/preprocess_duke.py --dataset_dir /local/scratch/scratch-hd/desmond/duke_micro_subset
```

### 3. JAX XLA Graph Compilation Target Tracking
Trigger the Continuous Active Vision loop tracking explicitly against `Breast_MRI_001`. This parses the Center-Slice video pipeline intrinsically into your active Weights & Biases dashboard mapping spatial loss geometries directly towards the ground truth anomaly coordinates!
```bash
python scripts/train_overfit.py
```
