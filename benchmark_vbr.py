#!/usr/bin/env python3
"""
Benchmark Any-Long (VGGT-Long) on VBR (Visual Benchmark Rome) sequences and output results in MAC-Loop protocol format.

Usage:
    # Run all VBR train sequences:
    python benchmark_vbr.py --vbr_root /media/airlab-storage/datasets/VBRome

    # Run specific sequences:
    python benchmark_vbr.py --vbr_root /media/airlab-storage/datasets/VBRome --sequences campus_train0 pincio_train0

    # Convert only (skip inference, just convert existing camera_poses.txt):
    python benchmark_vbr.py --vbr_root /media/airlab-storage/datasets/VBRome --convert_only --raw_results_dir ./raw_output/03_02_120000

    # Custom config:
    python benchmark_vbr.py --vbr_root /media/airlab-storage/datasets/VBRome --config configs/vbr.yaml
"""

import os
import sys
import glob
import time
import shutil
import argparse
import subprocess
from datetime import datetime

# All VBR train sequences (only train sequences have GT poses)
VBR_SEQUENCES = [
    "campus_train0",
    "campus_train1",
    "ciampino_train0",
    "ciampino_train1",
    "colosseo_train0",
    "diag_train0",
    "pincio_train0",
    "spagna_train0",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_anylong(image_dir, save_dir, config_path, downsample=1):
    """Run Any-Long (VGGT-Long) on a single sequence."""
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "vggt_long.py"),
        "--image_dir", image_dir,
        "--config", config_path,
        "--save_dir", save_dir,
        "--downsample", str(downsample),
    ]

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"Save dir: {save_dir}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"WARNING: Any-Long exited with code {result.returncode}")
        return None

    camera_poses_path = os.path.join(save_dir, "camera_poses.txt")
    if not os.path.exists(camera_poses_path):
        print(f"WARNING: camera_poses.txt not found in {save_dir}")
        return None

    print(f"Output found at: {save_dir}")
    return save_dir


def convert_poses(camera_poses_txt, vbr_seq_path, output_dir, downsample=1):
    """Convert Any-Long output to MAC-Loop protocol (VBR)."""
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "convert_vbr_to_macloop.py"),
        "--camera_poses", camera_poses_txt,
        "--vbr_seq", vbr_seq_path,
        "--output_dir", output_dir,
    ]
    if downsample > 1:
        cmd += ["--downsample", str(downsample)]
    print(f"Converting: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Benchmark Any-Long on VBR")
    parser.add_argument("--vbr_root", type=str, required=True,
                        help="Root directory of VBR dataset (contains campus_train0/, etc.)")
    parser.add_argument("--sequences", nargs="+", default=None,
                        help="Specific sequences to run (e.g. campus_train0 pincio_train0). Default: all train")
    parser.add_argument("--results_root", type=str, default="/mnt/data/anyslam/macloop/vggt-long-results",
                        help="Root directory for MAC-Loop protocol results")
    parser.add_argument("--config", type=str, default=os.path.join(SCRIPT_DIR, "configs", "vbr.yaml"),
                        help="Any-Long config YAML path")
    parser.add_argument("--raw_output_root", type=str, default="/mnt/data/anyslam/macloop/vggt-long-results/raw_output",
                        help="Root directory for raw Any-Long outputs")
    parser.add_argument("--convert_only", action="store_true",
                        help="Skip inference, only convert existing camera_poses.txt files")
    parser.add_argument("--raw_results_dir", type=str, default=None,
                        help="Directory containing raw Any-Long outputs (for --convert_only). "
                             "Should contain VGGT-Long@{seq}/camera_poses.txt subdirs")
    parser.add_argument("--downsample", type=int, default=2,
                        help="Pick every Nth image (default: 2, i.e. half framerate)")

    args = parser.parse_args()

    # Determine which sequences to run
    if args.sequences:
        sequences = [s for s in args.sequences if s in VBR_SEQUENCES]
        unknown = [s for s in args.sequences if s not in VBR_SEQUENCES]
        if unknown:
            print(f"WARNING: Unknown sequences: {unknown}")
            print(f"Available: {VBR_SEQUENCES}")
    else:
        sequences = VBR_SEQUENCES

    # Create timestamp directory for protocol output
    timestamp = datetime.now().strftime("%m_%d_%H%M%S")
    protocol_dir = os.path.join(args.results_root, "VGGT-Long", timestamp)
    raw_output_dir = os.path.join(args.raw_output_root, timestamp)
    os.makedirs(protocol_dir, exist_ok=True)
    os.makedirs(raw_output_dir, exist_ok=True)

    print(f"VBR root:        {args.vbr_root}")
    print(f"Sequences:       {sequences}")
    print(f"Config:          {args.config}")
    print(f"Raw output:      {raw_output_dir}")
    print(f"Protocol output: {protocol_dir}")
    print()

    results_summary = []
    total_start = time.time()

    for seq_name in sequences:
        seq_start = time.time()
        seq_path = os.path.join(args.vbr_root, seq_name)
        image_dir = os.path.join(seq_path, "camera_left", "data")
        gt_path = os.path.join(seq_path, f"{seq_name}_gt.txt")

        if not os.path.isdir(image_dir):
            print(f"SKIP: Image folder not found: {image_dir}")
            results_summary.append((seq_name, "SKIP", 0))
            continue

        if not os.path.exists(gt_path):
            print(f"SKIP: GT file not found: {gt_path}")
            results_summary.append((seq_name, "NO_GT", 0))
            continue

        protocol_seq_dir = os.path.join(protocol_dir, f"VGGT-Long@{seq_name}")

        # Step 1: Run Any-Long (unless convert_only)
        if args.convert_only:
            if args.raw_results_dir:
                seq_raw_dir = os.path.join(args.raw_results_dir, f"VGGT-Long@{seq_name}")
                # Also try without prefix
                if not os.path.isdir(seq_raw_dir):
                    seq_raw_dir = os.path.join(args.raw_results_dir, seq_name)
            else:
                print(f"SKIP: --convert_only requires --raw_results_dir")
                results_summary.append((seq_name, "NO_RAW_DIR", 0))
                continue
        else:
            seq_raw_dir = os.path.join(raw_output_dir, f"VGGT-Long@{seq_name}")
            result = run_anylong(image_dir, seq_raw_dir, args.config, downsample=args.downsample)
            if result is None:
                results_summary.append((seq_name, "SLAM_FAIL", time.time() - seq_start))
                continue

        # Step 2: Convert to MAC-Loop protocol
        camera_poses_txt = os.path.join(seq_raw_dir, "camera_poses.txt")
        if not os.path.exists(camera_poses_txt):
            print(f"SKIP: camera_poses.txt not found: {camera_poses_txt}")
            results_summary.append((seq_name, "NO_POSES", time.time() - seq_start))
            continue

        success = convert_poses(camera_poses_txt, seq_path, protocol_seq_dir, downsample=args.downsample)
        elapsed = time.time() - seq_start
        status = "OK" if success else "CONVERT_FAIL"
        results_summary.append((seq_name, status, elapsed))
        print(f"{seq_name}: {status} ({elapsed:.1f}s)")

        # Clean up per-chunk PLY files (keep combined_pcd.ply)
        pcd_dir = os.path.join(seq_raw_dir, "pcd")
        if os.path.isdir(pcd_dir):
            for f in glob.glob(os.path.join(pcd_dir, "*_pcd.ply")):
                if os.path.basename(f) != "combined_pcd.ply":
                    size_mb = os.path.getsize(f) / (1024 * 1024)
                    os.remove(f)
                    print(f"  Cleaned up {os.path.basename(f)} ({size_mb:.1f} MB)")

    total_time = time.time() - total_start

    # Print summary
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for seq, status, elapsed in results_summary:
        print(f"  {seq:20s}  {status:12s}  {elapsed:7.1f}s")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results at: {protocol_dir}")


if __name__ == "__main__":
    main()
