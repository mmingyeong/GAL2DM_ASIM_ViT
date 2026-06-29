# GAL2DM_ASIM_ViT

Vision Transformer-based model implementation for three-dimensional dark matter density field reconstruction from galaxy observables using the A-SIM simulation.

## Features

- 3D Vision Transformer-based reconstruction model
- Conditional reconstruction from galaxy number density and peculiar velocity fields
- Patch-based 3D volume processing
- Training and prediction pipelines
- Evaluation scripts for voxel-level and cosmological statistics
- Support for benchmark experiments, base-sweep studies, and random-seed analysis

## Repository Structure

```text
src/        # Model architecture and utilities
scripts/    # Training and prediction scripts
eval/       # Evaluation and analysis scripts
etc/        # Configuration files
debug/      # Debugging utilities
logs/       # Log files, ignored by git
results/    # Outputs/checkpoints, ignored by git
```

## Requirements

- Python 3.10+
- PyTorch
- CUDA
- NumPy
- h5py
- tqdm

## Notes

Large files such as datasets, model checkpoints, prediction outputs, and logs are not tracked in this repository.

## Citation

If you use this repository in your research, please cite the corresponding publication (to be added).