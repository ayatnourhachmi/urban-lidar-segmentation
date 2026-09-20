"""
PointNet and PointNet++ segmentation models for 3D urban LiDAR point clouds.

References:
  - PointNet: Qi et al., "PointNet: Deep Learning on Point Sets for 3D
    Classification and Segmentation", CVPR 2017.
    Paper: https://arxiv.org/abs/1612.00593
    Official (TF) repo: https://github.com/charlesq34/pointnet

  - PointNet++: Qi et al., "PointNet++: Deep Hierarchical Feature Learning
    on Point Sets in a Metric Space", NeurIPS 2017.
    Paper: https://arxiv.org/abs/1706.02413
    Official (TF) repo: https://github.com/charlesq34/pointnet2

  - This implementation follows the widely-used PyTorch port by Xu Yan,
    which is a common reference for reproducing both models:
    https://github.com/yanx27/Pointnet_Pointnet2_pytorch

The PointNet++ set-abstraction / feature-propagation layers here are
implemented in pure PyTorch (farthest point sampling + ball query done
with tensor ops) so the code runs on CPU or GPU without compiling custom
CUDA kernels. This is slower than the official CUDA ops for very large
point clouds, but is correct and dependency-light -- appropriate for a
portfolio project. If you scale to full-size city LiDAR tiles (100k+
points/sample), swap in the CUDA point-ops from the repo above.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Shared low-level ops (farthest point sampling, ball query, grouping)
# --------------------------------------------------------------------------

def square_distance(src, dst):
    """Pairwise squared Euclidean distance. src: (B,N,C) dst: (B,M,C) -> (B,N,M)"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """Gather points at given indices. points: (B,N,C), idx: (B,S) or (B,S,K) -> (B,S,C) or (B,S,K,C)"""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    """Iterative farthest point sampling. xyz: (B,N,3) -> indices (B,npoint)"""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """Group points within `radius` of each query point (cap at nsample)."""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """Farthest-point-sample centroids, then group + relative-normalize neighbors."""
    B, N, C = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) - new_xyz.view(B, npoint, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# --------------------------------------------------------------------------
# PointNet++ set abstraction / feature propagation
# --------------------------------------------------------------------------

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """xyz: (B,3,N) points: (B,C,N) or None -> new_xyz (B,3,npoint), new_points (B,C',npoint)"""
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)  # (B, C+3, nsample, npoint)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]  # (B, C', npoint)
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """Interpolate features from the coarser level (xyz2) back onto xyz1 (finer)."""
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNetPlusPlusSeg(nn.Module):
    """PointNet++ (single-scale grouping) semantic segmentation head.

    Input:  (B, 3+extra, N)  xyz (+ optional intensity/color channels)
    Output: (B, N, num_classes) per-point class logits
    """

    def __init__(self, num_classes, extra_channels=0):
        super().__init__()
        extra = extra_channels
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=1.0, nsample=32,
                                           in_channel=3 + extra, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=2.0, nsample=64,
                                           in_channel=128 + 3, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=32, radius=4.0, nsample=64,
                                           in_channel=256 + 3, mlp=[256, 256, 512])

        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + extra, mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        """xyz: (B, 3+extra, N)"""
        l0_xyz = xyz[:, :3, :]
        l0_points = xyz[:, 3:, :] if xyz.shape[1] > 3 else None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(x)
        x = self.conv2(x)
        return x.permute(0, 2, 1)  # (B, N, num_classes)


# --------------------------------------------------------------------------
# Classic PointNet segmentation (T-Net + shared MLP + global feature)
# --------------------------------------------------------------------------

class TNet(nn.Module):
    """Spatial/feature transform network, predicts a k x k transform matrix."""

    def __init__(self, k):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(64), nn.BatchNorm1d(128), nn.BatchNorm1d(1024)
        self.bn4, self.bn5 = nn.BatchNorm1d(512), nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(B, 1)
        x = x + iden
        return x.view(B, self.k, self.k)


class PointNetSeg(nn.Module):
    """Classic PointNet semantic segmentation.

    Input:  (B, 3+extra, N)
    Output: (B, N, num_classes)
    """

    def __init__(self, num_classes, extra_channels=0):
        super().__init__()
        in_ch = 3 + extra_channels
        self.input_tnet = TNet(k=in_ch)
        self.conv1 = nn.Conv1d(in_ch, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.feature_tnet = TNet(k=64)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1, self.bn2 = nn.BatchNorm1d(64), nn.BatchNorm1d(64)
        self.bn3, self.bn4, self.bn5 = nn.BatchNorm1d(64), nn.BatchNorm1d(128), nn.BatchNorm1d(1024)

        # segmentation head: per-point local (64) + global (1024) features
        self.seg_conv1 = nn.Conv1d(1088, 512, 1)
        self.seg_conv2 = nn.Conv1d(512, 256, 1)
        self.seg_conv3 = nn.Conv1d(256, 128, 1)
        self.seg_conv4 = nn.Conv1d(128, num_classes, 1)
        self.seg_bn1, self.seg_bn2, self.seg_bn3 = nn.BatchNorm1d(512), nn.BatchNorm1d(256), nn.BatchNorm1d(128)

    def forward(self, x):
        B, C, N = x.shape
        trans = self.input_tnet(x)
        x = torch.bmm(trans, x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        trans_feat = self.feature_tnet(x)
        x = torch.bmm(trans_feat, x)
        point_features = x  # (B, 64, N)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        global_feature = torch.max(x, 2, keepdim=True)[0]  # (B, 1024, 1)
        global_feature = global_feature.repeat(1, 1, N)

        x = torch.cat([point_features, global_feature], dim=1)  # (B, 1088, N)
        x = F.relu(self.seg_bn1(self.seg_conv1(x)))
        x = F.relu(self.seg_bn2(self.seg_conv2(x)))
        x = F.relu(self.seg_bn3(self.seg_conv3(x)))
        x = self.seg_conv4(x)
        return x.permute(0, 2, 1)  # (B, N, num_classes)


def build_model(name, num_classes, extra_channels=0):
    name = name.lower()
    if name == "pointnet":
        return PointNetSeg(num_classes, extra_channels)
    elif name == "pointnet2":
        return PointNetPlusPlusSeg(num_classes, extra_channels)
    else:
        raise ValueError(f"Unknown model '{name}', choose 'pointnet' or 'pointnet2'")
