# Urban LiDAR Segmentation

Semantic segmentation of urban LiDAR point clouds with **PointNet** and **PointNet++**, plus a small preprocessing and post-training quantization pipeline.

## What it does

1. Clean and downsample a point cloud (statistical outlier removal, voxel grid).
2. Train a PointNet or PointNet++ segmenter (6 urban classes by default).
3. Report per-point accuracy.
4. Apply dynamic int8 quantization and compare model size / accuracy.

Default training uses **synthetic** city-block scenes so the full pipeline runs without downloading a multi-GB dataset. A loader stub accepts preprocessed real scans (`points/*.npy` + `labels/*.npy`) when you have them.

## Classes

`ground`, `building`, `vegetation`, `vehicle`, `pole`, `other`

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# PointNet++ on synthetic data (CPU or CUDA)
python main.py --model pointnet2 --epochs 10 --num_points 4096

# Classic PointNet (better dynamic-quantization gains)
python main.py --model pointnet --epochs 10 --num_points 2048

# Real preprocessed data
python main.py --model pointnet2 --data_root /path/to/dataset --epochs 20
```

Optional EDA / screenshots:

```bash
python explore_point_cloud.py --demo
```

## Example run (synthetic, CPU smoke test)

```bash
python main.py --model pointnet --epochs 2 --num_points 2048
```

| Epoch | Val Acc | mIoU  |
|-------|---------|-------|
| 1     | 0.419   | 0.110 |
| 2     | 0.830   | 0.571 |

Per-class IoU after epoch 2: ground 0.92, building 0.70, vegetation 0.47, vehicle 0.56, pole 0.01, other 0.77.

`pole` stays near zero at 2 epochs (rare + geometrically thin). Longer training and PointNet++ will improve it. **These are synthetic-data numbers, not a SemanticKITTI / Toronto-3D result.**

## Results (synthetic, illustrative)

Numbers depend on seed, epochs, and machine. After a short CPU run you should see:

| Model | Notes |
|-------|--------|
| PointNet / PointNet++ | Train and val per-point accuracy printed each epoch |
| PointNet + dynamic int8 | Typically ~**50%** smaller `state_dict` on disk; accuracy usually within noise |
| PointNet++ + dynamic int8 | Little or no size reduction — set-abstraction uses `Conv2d`, which `quantize_dynamic` does not rewrite |

Treat these as pipeline checks, not a SemanticKITTI / Toronto-3D benchmark.

## Repo layout

| Path | Role |
|------|------|
| `main.py` | Train, validate, quantize, print report |
| `models.py` | PointNet and PointNet++ (pure PyTorch) |
| `dataset.py` | Synthetic generator + real `.npy` loader |
| `transformations.py` | Denoise, voxel / random downsample, augment |
| `quantization.py` | Dynamic int8 + size / accuracy / latency |
| `explore_point_cloud.py` | Open3D visualization walkthrough |
| `requirements.txt` | Dependencies |

## Limitations

- Default metrics report **per-point accuracy and mIoU** (with per-class breakdown after training).
- Synthetic geometry is for bringing up the pipeline; public leaderboard claims need a real dataset.
- Dynamic quantization helps PointNet much more than PointNet++ (see above).
- PointNet++ FPS / ball query are pure PyTorch (correct, slower than CUDA ops on huge clouds).

## References

- Qi et al., PointNet (CVPR 2017) · PointNet++ (NeurIPS 2017)
- PyTorch port reference: [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
- SemanticKITTI, Toronto-3D (optional real data)

## License

Personal portfolio code. Cite the PointNet papers if you reuse the model ideas.
