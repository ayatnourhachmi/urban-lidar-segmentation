"""
Datasets for urban LiDAR semantic segmentation.

For a resume/portfolio project you generally want to train on a real,
public urban LiDAR dataset. The two most common choices:

  - SemanticKITTI: sequential LiDAR scans of driving scenes, 19 semantic
    classes (road, building, vegetation, car, pedestrian, ...).
    Paper: Behley et al., "SemanticKITTI: A Dataset for Semantic Scene
    Understanding of LiDAR Sequences", ICCV 2019.
    https://arxiv.org/abs/1904.01416
    Data + devkit: http://www.semantic-kitti.org/

  - Toronto-3D: mobile-LiDAR point cloud of an urban street in Toronto,
    labeled with 8 classes (road, road markings, natural, building, ...).
    Paper: Tan et al., "Toronto-3D: A Large-scale Mobile LiDAR Dataset for
    Semantic Segmentation of Urban Roadways", CVPRW 2020.
    https://arxiv.org/abs/2003.08284
    Data: https://github.com/WeikaiTan/Toronto-3D

  - Also relevant / frequently benchmarked against: S3DIS (indoor, Stanford)
    and nuScenes-lidarseg (driving), if you want alternatives.

`RealLiDARDataset` below is a thin loader stub for the common case where
each scan is stored as a .bin/.npy/.las/.laz file of (x,y,z,intensity,...)
plus a parallel label file -- point it at your local copy of one of the
datasets above.

`SyntheticUrbanLiDAR` generates procedural point clouds that mimic a city
block (ground plane, building facades, poles, vegetation blobs, cars) so
you can run/debug/benchmark the full pipeline without downloading a
multi-GB dataset first.
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_NAMES = ["ground", "building", "vegetation", "vehicle", "pole", "other"]
NUM_CLASSES = len(CLASS_NAMES)


class SyntheticUrbanLiDAR(Dataset):
    """Procedurally generated city-block point clouds with 6 semantic
    classes, for pipeline development, unit tests, and benchmarking
    without a real dataset on disk.
    """

    def __init__(self, num_samples=200, points_per_sample=8000, transform=None, seed=0):
        self.num_samples = num_samples
        self.points_per_sample = points_per_sample
        self.transform = transform
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.num_samples

    def _make_scene(self):
        n = self.points_per_sample
        pts, labels = [], []

        # ground plane (class 0)
        n_ground = int(n * 0.35)
        g_xy = self.rng.uniform(-20, 20, size=(n_ground, 2))
        g_z = self.rng.normal(0, 0.03, size=(n_ground, 1))
        pts.append(np.concatenate([g_xy, g_z], axis=1))
        labels.append(np.zeros(n_ground, dtype=np.int64))

        # building facades (class 1) - a few vertical planes
        n_building = int(n * 0.30)
        n_buildings = self.rng.randint(2, 5)
        per = n_building // n_buildings
        for _ in range(n_buildings):
            cx, cy = self.rng.uniform(-18, 18, size=2)
            height = self.rng.uniform(6, 20)
            wall = np.stack([
                np.full(per, cx) + self.rng.normal(0, 0.05, per),
                cy + self.rng.uniform(-4, 4, per),
                self.rng.uniform(0, height, per),
            ], axis=1)
            pts.append(wall)
            labels.append(np.full(per, 1, dtype=np.int64))

        # vegetation blobs (class 2)
        n_veg = int(n * 0.15)
        n_trees = self.rng.randint(3, 8)
        per = n_veg // n_trees
        for _ in range(n_trees):
            center = np.array([self.rng.uniform(-18, 18), self.rng.uniform(-18, 18), self.rng.uniform(3, 6)])
            blob = center + self.rng.normal(0, 1.5, size=(per, 3))
            pts.append(blob)
            labels.append(np.full(per, 2, dtype=np.int64))

        # vehicles (class 3) - small boxes near ground
        n_veh = int(n * 0.10)
        n_cars = self.rng.randint(2, 6)
        per = n_veh // max(n_cars, 1)
        for _ in range(n_cars):
            center = np.array([self.rng.uniform(-15, 15), self.rng.uniform(-15, 15), 0.75])
            box = center + self.rng.uniform(-1, 1, size=(per, 3)) * np.array([2.2, 1.0, 0.75])
            pts.append(box)
            labels.append(np.full(per, 3, dtype=np.int64))

        # poles (class 4)
        n_pole = int(n * 0.05)
        n_poles = self.rng.randint(3, 6)
        per = n_pole // max(n_poles, 1)
        for _ in range(n_poles):
            base = np.array([self.rng.uniform(-18, 18), self.rng.uniform(-18, 18)])
            h = self.rng.uniform(0, 5, per)
            col = np.stack([np.full(per, base[0]), np.full(per, base[1]), h], axis=1)
            col[:, :2] += self.rng.normal(0, 0.02, size=(per, 2))
            pts.append(col)
            labels.append(np.full(per, 4, dtype=np.int64))

        pts = np.concatenate(pts, axis=0)
        labels = np.concatenate(labels, axis=0)

        # pad/truncate to exactly n points, rest labeled "other" (class 5) with noise
        cur = pts.shape[0]
        if cur < n:
            extra = self.rng.uniform(-20, 20, size=(n - cur, 3))
            extra[:, 2] = self.rng.uniform(0, 3, n - cur)
            pts = np.concatenate([pts, extra], axis=0)
            labels = np.concatenate([labels, np.full(n - cur, 5, dtype=np.int64)], axis=0)
        else:
            idx = self.rng.choice(cur, n, replace=False)
            pts, labels = pts[idx], labels[idx]

        # add a synthetic "intensity" channel (common LiDAR attribute, class-correlated + noise)
        intensity = 0.2 * labels.reshape(-1, 1) + self.rng.uniform(0, 0.3, size=(n, 1))
        pts = np.concatenate([pts, intensity], axis=1).astype(np.float32)  # (N, 4): x,y,z,intensity
        return pts, labels

    def __getitem__(self, i):
        # deterministic per-index scene via seeded local RNG state
        state = self.rng.get_state()
        self.rng = np.random.RandomState(i)
        pts, labels = self._make_scene()
        self.rng.set_state(state)

        if self.transform is not None:
            pts, labels = self.transform(pts, labels)
        return torch.from_numpy(pts).float().transpose(0, 1).contiguous(), torch.from_numpy(labels).long()


class RealLiDARDataset(Dataset):
    """Generic loader for point-cloud-file + label-file pairs, e.g. after
    converting SemanticKITTI (.bin) or Toronto-3D (.ply) into matching
    .npy point/label arrays. Expected layout:

        root/
          points/000000.npy   # (N, 3) or (N, 4) xyz(+intensity)
          labels/000000.npy   # (N,) integer class ids
          points/000001.npy
          labels/000001.npy
          ...

    For SemanticKITTI specifically, use the official devkit to read
    .bin/.label files (http://www.semantic-kitti.org/) and dump matching
    .npy pairs into this layout as a one-time preprocessing step.
    """

    def __init__(self, root, transform=None):
        self.point_files = sorted(glob.glob(os.path.join(root, "points", "*.npy")))
        self.label_files = sorted(glob.glob(os.path.join(root, "labels", "*.npy")))
        assert len(self.point_files) == len(self.label_files) and len(self.point_files) > 0, (
            f"Expected matching, non-empty points/ and labels/ .npy files under {root}"
        )
        self.transform = transform

    def __len__(self):
        return len(self.point_files)

    def __getitem__(self, i):
        pts = np.load(self.point_files[i]).astype(np.float32)
        labels = np.load(self.label_files[i]).astype(np.int64)
        if self.transform is not None:
            pts, labels = self.transform(pts, labels)
        return torch.from_numpy(pts).float().transpose(0, 1).contiguous(), torch.from_numpy(labels).long()
