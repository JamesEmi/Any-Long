"""
Convert Any-Long (VGGT-Long) camera_poses.txt output + EuRoC ground truth
to MAC-Loop protocol .npz files.

Any-Long outputs:
  camera_poses.txt: each line = 16 floats = flattened 4x4 C2W matrix (OpenCV convention)

Usage:
    python convert_to_macloop.py \
        --camera_poses output/camera_poses.txt \
        --image_dir /media/airlab-storage/datasets/EuRoC/MH_01_easy/mav0/cam0/data/ \
        --euroc_seq /media/airlab-storage/datasets/EuRoC/MH_01_easy \
        --output_dir results/VGGT-Long@MH01/
"""

import os
import re
import glob
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R


# EuRoC cam0 T_BS (body-to-sensor, same across all EuRoC sequences)
T_BS_CAM0 = np.array([
    [0.0148655429818, -0.999880929698,  0.00414029679422, -0.0216401454975],
    [0.999557249008,   0.0149672133247, 0.025715529948,   -0.064676986768],
    [-0.0257744366974, 0.00375618835797, 0.999660727178,   0.00981073058949],
    [0.0,              0.0,              0.0,               1.0]
])

# EDN (East-Down-North) to NED (North-East-Down) transform
EDN2NED = np.array([
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=np.float64)

NED2EDN = np.linalg.inv(EDN2NED)


def mat_to_se3(mat):
    """Convert 4x4 matrix to (tx, ty, tz, qx, qy, qz, qw)."""
    t = mat[:3, 3]
    q = R.from_matrix(mat[:3, :3]).as_quat()  # returns [x, y, z, w]
    return np.array([t[0], t[1], t[2], q[0], q[1], q[2], q[3]])


def se3_to_mat(tx, ty, tz, qx, qy, qz, qw):
    """Convert translation + quaternion (x,y,z,w) to 4x4 matrix."""
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    mat[:3, 3] = [tx, ty, tz]
    return mat


def transform_c2w_to_body_ned(c2w_mat):
    """
    Transform a C2W (camera-to-world) pose to NED body frame.

    T_world_body = C2W @ inv(T_BS)
    T_NED = inv(EDN2NED) @ T_world_body @ EDN2NED
    """
    T_BS_inv = np.linalg.inv(T_BS_CAM0)
    T_world_body = c2w_mat @ T_BS_inv
    T_ned = NED2EDN @ T_world_body @ EDN2NED
    return T_ned


def transform_gt_pose_to_ned(pose_mat):
    """
    Transform EuRoC GT body-frame pose to NED convention.
    GT is already in body frame (T_BS = identity for state_groundtruth_estimate0).
    Only need EDN→NED conversion.
    """
    return NED2EDN @ pose_mat @ EDN2NED


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


def get_euroc_timestamps_from_images(image_dir):
    """
    Get timestamps from EuRoC image filenames.
    Any-Long uses sorted(glob(...)) for lexicographic ordering.
    EuRoC filenames are nanosecond timestamps (e.g. 1403636579763555584.png).
    """
    image_files = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg")) +
        glob.glob(os.path.join(image_dir, "*.png"))
    )
    timestamps_ns = []
    for path in image_files:
        filename = os.path.basename(path)
        match = re.search(r'\d+', filename)
        if match:
            timestamps_ns.append(int(match.group()))
        else:
            raise ValueError(f"No numeric timestamp found in filename: {filename}")
    return timestamps_ns


def load_euroc_gt(euroc_seq_path):
    """Load EuRoC ground truth from state_groundtruth_estimate0/data.csv"""
    gt_path = os.path.join(euroc_seq_path, "mav0", "state_groundtruth_estimate0", "data.csv")
    data = np.loadtxt(gt_path, delimiter=",", skiprows=1)
    time_ns = data[:, 0].astype(np.int64)
    positions = data[:, 1:4]        # px, py, pz
    quats_wxyz = data[:, 4:8]       # qw, qx, qy, qz
    # Convert to PyPose convention: x, y, z, w
    quats_xyzw = np.column_stack([quats_wxyz[:, 1], quats_wxyz[:, 2], quats_wxyz[:, 3], quats_wxyz[:, 0]])
    poses_7 = np.column_stack([positions, quats_xyzw])
    return time_ns, poses_7


def convert_poses(camera_poses_txt, image_dir, euroc_seq_path, output_dir, skip_transform=False):
    """
    Full conversion pipeline:
    1. Load Any-Long C2W poses
    2. Match with image timestamps
    3. Apply coordinate transforms (camera→body, EDN→NED)
    4. Load EuRoC GT and transform to NED
    5. Save both as .npy in MAC-Loop protocol format (N,8): [timestamp, tx, ty, tz, qx, qy, qz, qw]
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load C2W poses
    c2w_matrices = load_anylong_poses(camera_poses_txt)
    N = len(c2w_matrices)

    # Get timestamps from image filenames (same ordering as Any-Long)
    timestamps_ns = get_euroc_timestamps_from_images(image_dir)
    assert len(timestamps_ns) == N, \
        f"Pose count ({N}) != image count ({len(timestamps_ns)}). " \
        f"Any-Long should produce one pose per image."

    time_ns = np.array(timestamps_ns, dtype=np.int64)

    # Transform poses
    poses_7 = []
    for i in range(N):
        c2w = c2w_matrices[i]
        if not skip_transform:
            c2w = transform_c2w_to_body_ned(c2w)
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

    # Load and transform ground truth
    if euroc_seq_path:
        gt_time_ns, gt_poses_7 = load_euroc_gt(euroc_seq_path)

        if not skip_transform:
            transformed_gt = []
            for i in range(len(gt_poses_7)):
                pose_mat = se3_to_mat(*gt_poses_7[i])
                pose_ned = transform_gt_pose_to_ned(pose_mat)
                transformed_gt.append(mat_to_se3(pose_ned))
            gt_poses_7 = np.array(transformed_gt)

        N_gt = len(gt_poses_7)

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
    parser = argparse.ArgumentParser(description="Convert Any-Long output to MAC-Loop protocol")
    parser.add_argument("--camera_poses", type=str, required=True,
                        help="Path to Any-Long camera_poses.txt")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to image directory (for timestamp recovery)")
    parser.add_argument("--euroc_seq", type=str, default=None,
                        help="Path to EuRoC sequence root (e.g. .../MH_01_easy)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for .npz files")
    parser.add_argument("--skip_transform", action="store_true",
                        help="Skip coordinate frame transforms")
    args = parser.parse_args()

    convert_poses(args.camera_poses, args.image_dir, args.euroc_seq, args.output_dir,
                  args.skip_transform)
