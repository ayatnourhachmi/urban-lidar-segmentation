"""
Open3D exploration demo for a single point cloud scan: visualize, denoise,
downsample, estimate normals, and color-code by height/density.

This mirrors the standard first pass on any new point cloud dataset before
model training: look at the raw scan, clean it, then simplify it. It also
gives you a self-contained script/notebook you can screenshot for a
GitHub README (before/after visuals are the single best thing to put in a
portfolio repo for this kind of project).

Requires: pip install open3d

Usage:
    python explore_point_cloud.py --path /path/to/scan.ply
    python explore_point_cloud.py --demo    # uses Open3D's built-in sample data
"""

import argparse

import numpy as np
import open3d as o3d


def load_cloud(path=None):
    if path:
        return o3d.io.read_point_cloud(path)
    # Open3D ships a small sample point cloud for exactly this kind of demo
    sample = o3d.data.PLYPointCloud()
    return o3d.io.read_point_cloud(sample.path)


def visualize(pcd, title="Point Cloud"):
    o3d.visualization.draw_geometries([pcd], window_name=title)


def denoise(pcd, nb_neighbors=20, std_ratio=2.0):
    """Statistical outlier removal -- strips isolated sensor-noise points."""
    clean, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"Statistical outlier removal: {len(pcd.points)} -> {len(clean.points)} points "
          f"({len(pcd.points) - len(clean.points)} removed)")
    return clean


def downsample(pcd, voxel_size=0.05):
    down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Voxel downsample (size={voxel_size}): {len(pcd.points)} -> {len(down.points)} points")
    return down


def estimate_normals(pcd, radius=0.1, max_nn=30):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return pcd


def color_by_height(pcd):
    points = np.asarray(pcd.points)
    z = points[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    colors = np.column_stack([z_norm, 1 - z_norm, np.zeros_like(z_norm)])  # low=blue-ish, high=red
    colored = o3d.geometry.PointCloud(pcd)
    colored.colors = o3d.utility.Vector3dVector(colors)
    return colored


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=str, default=None, help="Path to a .ply/.pcd/.las point cloud file")
    p.add_argument("--demo", action="store_true", help="Use Open3D's built-in sample point cloud")
    p.add_argument("--voxel_size", type=float, default=0.05)
    p.add_argument("--no_viz", action="store_true", help="Skip interactive windows (for headless/CI runs)")
    args = p.parse_args()

    pcd = load_cloud(args.path if not args.demo else None)
    print(f"Loaded cloud: {pcd}")

    if not args.no_viz:
        visualize(pcd, "1. Raw point cloud")

    clean = denoise(pcd)
    if not args.no_viz:
        visualize(clean, "2. After statistical outlier removal")

    down = downsample(clean, voxel_size=args.voxel_size)
    if not args.no_viz:
        visualize(down, "3. After voxel downsampling")

    down = estimate_normals(down)
    if not args.no_viz:
        visualize(down, "4. With estimated normals (press N to toggle)")

    colored = color_by_height(down)
    if not args.no_viz:
        visualize(colored, "5. Colored by height (blue=low, red=high)")


if __name__ == "__main__":
    main()
