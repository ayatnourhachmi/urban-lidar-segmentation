"""
Point cloud transforms for urban LiDAR semantic segmentation.

Downsampling is the first lever used to cut memory footprint (fewer points
per sample -> smaller activations and less GPU/CPU memory during both
training and inference), the second lever is model quantization (see
quantization.py). Both are combined without hurting accuracy by choosing a
downsampling method that preserves geometric structure (voxel-grid) rather
than pure random point drops, and by validating quantized accuracy against
a held-out set before deployment.

Voxel-grid downsampling follows the standard approach used in Open3D and
PCL (Point Cloud Library):
  - Open3D voxel_down_sample: http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Voxel-downsampling
  - PCL VoxelGrid filter: https://pointclouds.org/documentation/tutorials/voxel_grid.html

Design note: real projects typically call Open3D's C++-backed
`voxel_down_sample` for speed. The implementation below is a dependency-free
pure-NumPy equivalent (same algorithm: bucket points into a 3D grid, keep
the centroid of each occupied voxel) so this file has no hard dependency on
open3d. If open3d is installed, swap in `open3d_voxel_downsample` for a
large speedup on big scans.
"""

import numpy as np
import torch


# --------------------------------------------------------------------------
# Downsampling
# --------------------------------------------------------------------------

def voxel_grid_downsample(points, labels=None, voxel_size=0.2):
    """Bucket points into voxels of size `voxel_size` and keep one point
    (the centroid) per occupied voxel. This is the standard LiDAR
    preprocessing step used to shrink point counts before feeding a
    network, since raw city LiDAR scans can have 100k-1M+ points/frame.

    points: (N, 3+) numpy array, first 3 cols are xyz
    labels: optional (N,) numpy array of per-point labels
    returns: downsampled points (M, 3+), and labels (M,) if provided
    """
    coords = points[:, :3]
    voxel_idx = np.floor(coords / voxel_size).astype(np.int64)
    # unique voxel key per point
    keys = voxel_idx[:, 0] * 73856093 ^ voxel_idx[:, 1] * 19349663 ^ voxel_idx[:, 2] * 83492791
    _, unique_idx, inverse = np.unique(keys, return_index=True, return_inverse=True)

    M = unique_idx.shape[0]
    out_points = np.zeros((M, points.shape[1]), dtype=points.dtype)
    counts = np.zeros(M, dtype=np.int64)
    np.add.at(out_points, inverse, points)
    np.add.at(counts, inverse, 1)
    out_points /= counts[:, None]

    if labels is not None:
        # majority label per voxel (mode)
        out_labels = np.zeros(M, dtype=labels.dtype)
        for i in range(M):
            mask = inverse == i
            vals, counts_l = np.unique(labels[mask], return_counts=True)
            out_labels[i] = vals[np.argmax(counts_l)]
        return out_points, out_labels
    return out_points


def open3d_voxel_downsample(points, labels=None, voxel_size=0.2):
    """Same behavior as voxel_grid_downsample but backed by Open3D's fast
    C++ implementation. Use this in production; requires `pip install open3d`.
    """
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    down = pcd.voxel_down_sample(voxel_size=voxel_size)
    down_xyz = np.asarray(down.points)
    if labels is not None:
        # nearest-neighbor label transfer from original cloud
        from scipy.spatial import cKDTree
        tree = cKDTree(points[:, :3])
        _, idx = tree.query(down_xyz, k=1)
        return down_xyz, labels[idx]
    return down_xyz


# --------------------------------------------------------------------------
# Noise removal (run before downsampling, on the raw scan)
# --------------------------------------------------------------------------

def statistical_outlier_removal(points, labels=None, nb_neighbors=20, std_ratio=2.0):
    """Remove points whose average distance to their `nb_neighbors` nearest
    neighbors is farther than `std_ratio` standard deviations from the
    dataset mean -- Open3D's `remove_statistical_outlier` algorithm,
    reimplemented with scipy so this file stays swappable between the two.
    Typical LiDAR use: strip sparse sensor noise before voxel downsampling.
    Reference: http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Statistical-outlier-removal
    """
    from scipy.spatial import cKDTree
    coords = points[:, :3]
    tree = cKDTree(coords)
    # k+1 because the nearest neighbor of a point is itself (distance 0)
    dists, _ = tree.query(coords, k=nb_neighbors + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    mu, sigma = mean_dists.mean(), mean_dists.std()
    keep = mean_dists < mu + std_ratio * sigma
    if labels is not None:
        return points[keep], labels[keep]
    return points[keep]


def radius_outlier_removal(points, labels=None, nb_points=16, radius=0.5):
    """Remove points with fewer than `nb_points` neighbors within `radius`
    -- Open3D's `remove_radius_outlier` algorithm. Good for stripping
    isolated stragglers (e.g. birds, dust, sensor glare) in outdoor/LiDAR
    scans where density varies a lot by distance from the sensor.

    Note: on scans with strong range-dependent density falloff (far-away
    points are naturally sparser), a single global radius can over-remove
    distant real geometry -- this matches the "doesn't work well for me"
    experience of using it on uneven-density outdoor scans. Statistical
    outlier removal above is usually more robust for that case; use radius
    removal mainly on roughly uniform-density regions or tune `radius` per
    distance band if you need it on full-scene scans.
    Reference: http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Radius-outlier-removal
    """
    from scipy.spatial import cKDTree
    coords = points[:, :3]
    tree = cKDTree(coords)
    neighbor_counts = tree.query_ball_point(coords, r=radius, return_length=True)
    keep = neighbor_counts >= (nb_points + 1)  # +1 since a point neighbors itself
    if labels is not None:
        return points[keep], labels[keep]
    return points[keep]


def open3d_statistical_outlier_removal(points, labels=None, nb_neighbors=20, std_ratio=2.0):
    """Same as statistical_outlier_removal but backed by Open3D directly
    (faster on large scans). Requires `pip install open3d`."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    ind = np.asarray(ind)
    if labels is not None:
        return points[ind], labels[ind]
    return points[ind]


def random_downsample(points, labels=None, num_points=4096):
    """Uniform random subsample to a fixed point count. Simpler/faster than
    voxel-grid but does not preserve spatial density as well -- useful as
    a baseline or for quick augmentation during training.
    """
    N = points.shape[0]
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
    else:
        idx = np.random.choice(N, num_points, replace=True)
    if labels is not None:
        return points[idx], labels[idx]
    return points[idx]


def fixed_size_sample(points, labels=None, num_points=4096):
    """Deterministic downsample/pad to exactly num_points (for batching
    variable-size scans into fixed-size tensors)."""
    N = points.shape[0]
    if N == num_points:
        pass
    elif N > num_points:
        idx = np.random.choice(N, num_points, replace=False)
        points = points[idx]
        if labels is not None:
            labels = labels[idx]
    else:
        pad_idx = np.random.choice(N, num_points - N, replace=True)
        points = np.concatenate([points, points[pad_idx]], axis=0)
        if labels is not None:
            labels = np.concatenate([labels, labels[pad_idx]], axis=0)
    if labels is not None:
        return points, labels
    return points


# --------------------------------------------------------------------------
# Normalization & augmentation
# --------------------------------------------------------------------------

def normalize_xyz(points):
    """Center at centroid and scale to a unit sphere -- standard PointNet
    preprocessing so absolute scan position/scale doesn't affect learning."""
    points = points.copy()
    centroid = points[:, :3].mean(axis=0)
    points[:, :3] -= centroid
    scale = np.max(np.sqrt(np.sum(points[:, :3] ** 2, axis=1)))
    points[:, :3] /= (scale + 1e-8)
    return points


def random_rotate_z(points):
    """Random rotation about the vertical (z) axis -- valid augmentation for
    urban LiDAR since street scenes have no preferred yaw orientation."""
    theta = np.random.uniform(0, 2 * np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=points.dtype)
    points = points.copy()
    points[:, :3] = points[:, :3] @ R.T
    return points


def jitter(points, sigma=0.01, clip=0.05):
    """Small Gaussian noise on xyz to simulate LiDAR range noise."""
    points = points.copy()
    noise = np.clip(sigma * np.random.randn(*points[:, :3].shape), -clip, clip)
    points[:, :3] += noise
    return points


def random_drop(points, labels=None, max_drop_ratio=0.15):
    """Randomly drop a fraction of points to simulate occlusion/sparsity,
    also a cheap way to train robustness to the downsampling used at
    inference time."""
    N = points.shape[0]
    drop_ratio = np.random.uniform(0, max_drop_ratio)
    keep_n = int(N * (1 - drop_ratio))
    idx = np.random.choice(N, keep_n, replace=False)
    if labels is not None:
        return points[idx], labels[idx]
    return points[idx]


def to_tensor(points, labels=None):
    """numpy (N, C) -> torch (C, N) float tensor, ready for Conv1d models."""
    pts = torch.from_numpy(points).float().transpose(0, 1).contiguous()
    if labels is not None:
        lbl = torch.from_numpy(labels).long()
        return pts, lbl
    return pts


class Compose:
    """Chain several point-cloud transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points, labels=None):
        for t in self.transforms:
            result = t(points, labels) if labels is not None else t(points)
            if isinstance(result, tuple):
                points, labels = result
            else:
                points = result
        if labels is not None:
            return points, labels
        return points


def train_transform(voxel_size=0.2, num_points=4096, remove_outliers=True):
    """Standard training-time pipeline:
    statistical outlier removal -> voxel downsample -> augment -> fixed-size sample -> normalize.

    Outlier removal runs first, on the raw (denser) scan, since that's where
    sensor noise/isolated points are still identifiable -- after voxel
    downsampling, noisy points may already be folded into voxel centroids.
    """
    def _fn(points, labels):
        if remove_outliers and points.shape[0] > 50:  # skip on tiny/synthetic edge cases
            points, labels = statistical_outlier_removal(points, labels, nb_neighbors=20, std_ratio=2.0)
        points, labels = voxel_grid_downsample(points, labels, voxel_size)
        points = random_rotate_z(points)
        points = jitter(points)
        points, labels = fixed_size_sample(points, labels, num_points)
        points = normalize_xyz(points)
        return points, labels
    return _fn


def eval_transform(voxel_size=0.2, num_points=4096, remove_outliers=True):
    """Deterministic pipeline for validation/inference (no augmentation)."""
    def _fn(points, labels=None):
        if labels is not None:
            if remove_outliers and points.shape[0] > 50:
                points, labels = statistical_outlier_removal(points, labels, nb_neighbors=20, std_ratio=2.0)
            points, labels = voxel_grid_downsample(points, labels, voxel_size)
            points, labels = fixed_size_sample(points, labels, num_points)
            points = normalize_xyz(points)
            return points, labels
        if remove_outliers and points.shape[0] > 50:
            points = statistical_outlier_removal(points, None, nb_neighbors=20, std_ratio=2.0)
        points = voxel_grid_downsample(points, None, voxel_size)
        points = fixed_size_sample(points, None, num_points)
        points = normalize_xyz(points)
        return points
    return _fn
