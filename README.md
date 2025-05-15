# Terrain Diffuser

A configurable training pipeline for terrain-to-satellite image diffusion models.

## Requirements

- Python 3.10  
- PyTorch 2.1.2  
- CUDA‐enabled GPU (optional, but highly recommended)

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running Training
You **must** invoke the trainer as a Python module. From the project root:

```bash
python -m terrain_diffuser.training.trainer configs/default.yaml
```

probably won't work so gotta debug!