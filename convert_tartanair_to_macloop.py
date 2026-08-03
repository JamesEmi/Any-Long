"""
Convert Any-Long (VGGT-Long) camera_poses.txt output + TartanAir V2 ground truth
to MAC-Loop protocol .npz files.

Any-Long outputs:
  camera_poses.txt: each line = 16 floats = flattened 4x4 C2W matrix (OpenCV convention)

TartanAir V2 GT: pose_lcam_front.txt with N lines of (tx ty tz qx qy qz qw)

No EuRoC-specific coordinate transforms are applied — evo alignment handles frame differences.

Usage:
    python convert_tartanair_to_macloop.py \
        --camera_poses output/camera_poses.txt \
        --image_dir /path/to/Data_easy/P000/image_lcam_front/ \
        --gt_poses /path/to/Data_easy/P000/pose_lcam_front.txt \
        --output_dir results/VGGT-Long@TA_E_P000/
"""

import os
import re
import glob
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R


def mat_to_se3(mat):
    """Convert 4x4 matrix to (tx, ty, tz, qx, qy, qz, qw)."""
    t = mat[:3, 3]
    q = R.from_matrix(mat[:3, :3]).as_quat()  # returns [x, y, z, w]
    return np.array([t[0], t[1], t[2], q[0], q[1], q[2], q[3]])


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


def get_tartanair_timestamps(image_dir):
    """
    Get timestamps from TartanAir image filenames.
    Filenames are like 000000_lcam_front.png — extract frame index, synthesize 10Hz timestamps.
    """
    image_files = sorted(
        glob.glob(os.path.join(image_dir, "*.png")) +
        glob.glob(os.path.join(image_dir, "*.jpg"))
    )
    timestamps_ns = []
    for path in image_files:
        filename = os.path.basename(path)
        match = re.search(r'(\d+)', filename)
        if match:
            frame_idx = int(match.group(1))
            timestamps_ns.append(frame_idx * 100_000_000)  # 10Hz in nanoseconds
        else:
            raise ValueError(f"No numeric index found in filename: {filename}")
    return timestamps_ns


def load_tartanair_gt(gt_path):
    """Load TartanAir V2 ground truth: N lines of (tx ty tz qx qy qz qw)"""
    return np.loadtxt(gt_path)


def convert_poses(camera_poses_txt, image_dir, gt_path, output_dir):
    """
    Full conversion pipeline:
    1. Load Any-Long C2W poses (no coordinate transform — evo aligns at eval time)
    2. Match with image timestamps
    3. Load TartanAir GT poses
    4. Save as .npy in MAC-Loop protocol format (N,8): [timestamp, tx, ty, tz, qx, qy, qz, qw]
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load C2W poses
    c2w_matrices = load_anylong_poses(camera_poses_txt)
    N = len(c2w_matrices)

    # Get timestamps from image filenames
    timestamps_ns = get_tartanair_timestamps(image_dir)
    assert len(timestamps_ns) == N, \
        f"Pose count ({N}) != image count ({len(timestamps_ns)}). " \
        f"Any-Long should produce one pose per image."

    time_ns = np.array(timestamps_ns, dtype=np.int64)

    # Convert C2W matrices to SE3 (no frame transform)
    poses_7 = []
    for c2w in c2w_matrices:
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

    # Load and save ground truth
    if gt_path and os.path.exists(gt_path):
        gt_poses_7 = load_tartanair_gt(gt_path)
        N_gt = len(gt_poses_7)
        gt_time_ns = (np.arange(N_gt) * 100_000_000).astype(np.int64)

        ref_path = os.path.join(output_dir, "ref_poses.npy")
        combined_gt = np.concatenate([
            gt_time_ns.astype(np.float64).reshape(-1, 1),
            gt_poses_7.astype(np.float64)
        ], axis=-1)
        np.save(ref_path, combined_gt)
        print(f"Saved {N_gt} ground truth poses to {ref_path}")

    # Write config.yaml for MAC-SLAM Sandbox compatibility
    seq_name = os.path.basename(output_dir)
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(f"Project: {seq_name}\n")
    print(f"Saved config to {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Any-Long output to MAC-Loop protocol (TartanAir)")
    parser.add_argument("--camera_poses", type=str, required=True,
                        help="Path to Any-Long camera_poses.txt")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to image directory (for timestamp recovery)")
    parser.add_argument("--gt_poses", type=str, default=None,
                        help="Path to TartanAir pose_lcam_front.txt")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for .npz files")
    args = parser.parse_args()

    convert_poses(args.camera_poses, args.image_dir, args.gt_poses, args.output_dir)
