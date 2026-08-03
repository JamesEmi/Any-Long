"""
Convert Any-Long (VGGT-Long) camera_poses.txt output + VBR ground truth
to MAC-Loop protocol .npy files.

Any-Long outputs:
  camera_poses.txt: each line = 16 floats = flattened 4x4 C2W matrix (OpenCV convention)

VBR GT format: {seq}_gt.txt with header '#timestamp tx ty tz qx qy qz qw'
  - timestamps in seconds, poses in body/lidar frame (lidar T_b = identity)

Camera→body transform is applied using T_b from vbr_calib.yaml.
GT poses are kept in body frame (no additional transform needed).

Usage:
    python convert_vbr_to_macloop.py \
        --camera_poses output/camera_poses.txt \
        --vbr_seq /media/airlab-storage/datasets/VBRome/campus_train0 \
        --output_dir results/VGGT-Long@campus_train0/
"""

import os
import re
import glob
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

# VBR cam_l T_b (body-to-camera, same across all VBR sequences)
T_B_CAM_L = np.array([
    [0.005617112780309785, -0.0012881145325502978, 0.9999832905520013, 0.07073856167431194],
    [-0.9999833070153535, -0.0013285236613379636, 0.005615378227348316, 0.23435089305558293],
    [0.001321217132003304, -0.9999982616209818, -0.0012955402899334433, -0.6660491439765341],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float64)

T_CAM_L_B = np.linalg.inv(T_B_CAM_L)


def mat_to_se3(mat):
    """Convert 4x4 matrix to (tx, ty, tz, qx, qy, qz, qw)."""
    t = mat[:3, 3]
    q = R.from_matrix(mat[:3, :3]).as_quat()  # returns [x, y, z, w]
    return np.array([t[0], t[1], t[2], q[0], q[1], q[2], q[3]])


def transform_c2w_to_body(c2w_mat):
    """
    Transform a C2W (camera-to-world) pose to body frame.
    T_world_body = C2W @ inv(T_b)
    where T_b is body-to-camera, so inv(T_b) = camera-to-body.
    """
    return c2w_mat @ T_CAM_L_B


def load_anylong_poses(camera_poses_txt):
    """
    Load Any-Long camera_poses.txt: each line = 16 floats = flattened 4x4 C2W matrix.
    Returns list of 4x4 numpy arrays.
    """
    c2w_matrices = []
    with open(camera_poses_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = [float(v) for v in line.split()]
            assert len(values) == 16, f"Expected 16 values per line, got {len(values)}"
            c2w = np.array(values).reshape(4, 4)
            c2w_matrices.append(c2w)
    return c2w_matrices


def parse_vbr_timestamps(timestamps_txt):
    """
    Parse VBR camera timestamps.txt (ISO 8601 format) to nanoseconds.
    Format: 1970-01-01T00:06:06.660458140
    """
    timestamps_ns = []
    with open(timestamps_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse time part: HH:MM:SS.nnnnnnnnn
            time_part = line.split('T')[1]
            h, m, s = time_part.split(':')
            total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
            timestamps_ns.append(int(total_seconds * 1_000_000_000))
    return np.array(timestamps_ns, dtype=np.int64)


def load_vbr_gt(gt_path):
    """
    Load VBR ground truth: #timestamp tx ty tz qx qy qz qw
    Returns timestamps in nanoseconds and poses as (N, 7).
    """
    data = np.loadtxt(gt_path, comments='#')
    time_s = data[:, 0]
    time_ns = (time_s * 1_000_000_000).astype(np.int64)
    poses_7 = data[:, 1:]  # tx ty tz qx qy qz qw
    return time_ns, poses_7


def convert_poses(camera_poses_txt, vbr_seq_path, output_dir, skip_transform=False, downsample=1):
    """
    Full conversion pipeline:
    1. Load Any-Long C2W poses
    2. Match with camera timestamps
    3. Transform estimated poses from camera frame to body frame
    4. Load VBR GT poses (already in body frame)
    5. Save as .npy in MAC-Loop protocol format (N,8): [timestamp, tx, ty, tz, qx, qy, qz, qw]
    """
    os.makedirs(output_dir, exist_ok=True)

    seq_name = os.path.basename(vbr_seq_path)

    # Load C2W poses
    c2w_matrices = load_anylong_poses(camera_poses_txt)
    N = len(c2w_matrices)

    # Get timestamps from camera timestamps file
    timestamps_txt = os.path.join(vbr_seq_path, "camera_left", "timestamps.txt")
    cam_timestamps_ns = parse_vbr_timestamps(timestamps_txt)

    # Any-Long produces one pose per (downsampled) image
    # With downsampling, pose i corresponds to original image i * downsample
    n_downsampled = (len(cam_timestamps_ns) + downsample - 1) // downsample
    if N != n_downsampled:
        print(f"WARNING: Pose count ({N}) != expected downsampled image count ({n_downsampled}). "
              f"Using min({N}, {n_downsampled}) poses.")
        N = min(N, n_downsampled)
        c2w_matrices = c2w_matrices[:N]

    # Map pose index to original timestamp via downsample factor
    original_indices = np.arange(N) * downsample
    original_indices = np.minimum(original_indices, len(cam_timestamps_ns) - 1)
    time_ns = cam_timestamps_ns[original_indices]

    if downsample > 1:
        print(f"Downsample factor {downsample}: mapping {N} poses to original timestamps")

    # Transform poses
    poses_7 = []
    for i in range(N):
        c2w = c2w_matrices[i]
        if not skip_transform:
            c2w = transform_c2w_to_body(c2w)
        poses_7.append(mat_to_se3(c2w))
    poses_7 = np.array(poses_7, dtype=np.float64)

    # Save estimated poses as (N, 8): [timestamp, tx, ty, tz, qx, qy, qz, qw]
    est_path = os.path.join(output_dir, "poses.npy")
    combined_est = np.concatenate([
        time_ns.astype(np.float64).reshape(-1, 1),
        poses_7.astype(np.float64)
    ], axis=-1)
    np.save(est_path, combined_est)
    print(f"Saved {N} estimated poses to {est_path}")

    # Load and save ground truth (already in body frame)
    gt_path = os.path.join(vbr_seq_path, f"{seq_name}_gt.txt")
    if os.path.exists(gt_path):
        gt_time_ns, gt_poses_7 = load_vbr_gt(gt_path)
        N_gt = len(gt_poses_7)

        ref_path = os.path.join(output_dir, "ref_poses.npy")
        combined_gt = np.concatenate([
            gt_time_ns.astype(np.float64).reshape(-1, 1),
            gt_poses_7.astype(np.float64)
        ], axis=-1)
        np.save(ref_path, combined_gt)
        print(f"Saved {N_gt} ground truth poses to {ref_path}")
    else:
        print(f"WARNING: GT file not found: {gt_path}")

    # Write config.yaml for MAC-SLAM Sandbox compatibility
    sandbox_name = os.path.basename(output_dir)
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(f"Project: {sandbox_name}\n")
    print(f"Saved config to {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Any-Long output to MAC-Loop protocol (VBR)")
    parser.add_argument("--camera_poses", type=str, required=True,
                        help="Path to Any-Long camera_poses.txt")
    parser.add_argument("--vbr_seq", type=str, required=True,
                        help="Path to VBR sequence root (e.g. .../VBRome/campus_train0)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for .npy files")
    parser.add_argument("--skip_transform", action="store_true",
                        help="Skip camera→body coordinate frame transform")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Downsample factor used when running inference (maps pose_idx to original_idx = pose_idx * downsample)")
    args = parser.parse_args()

    convert_poses(args.camera_poses, args.vbr_seq, args.output_dir, args.skip_transform, args.downsample)
