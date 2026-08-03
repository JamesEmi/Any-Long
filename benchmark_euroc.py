#!/usr/bin/env python3
"""
Benchmark Any-Long (VGGT-Long) on EuRoC sequences and output results in MAC-Loop protocol format.

Usage:
    # Run all EuRoC sequences:
    python benchmark_euroc.py --euroc_root /media/airlab-storage/datasets/EuRoC

    # Run specific sequences:
    python benchmark_euroc.py --euroc_root /media/airlab-storage/datasets/EuRoC --sequences MH01 MH02

    # Convert only (skip inference, just convert existing camera_poses.txt):
    python benchmark_euroc.py --euroc_root /media/airlab-storage/datasets/EuRoC --convert_only --raw_results_dir ./exps/some_run

    # Custom config:
    python benchmark_euroc.py --euroc_root /media/airlab-storage/datasets/EuRoC --config configs/base_config.yaml
"""

import os
import sys
import glob
import time
import argparse
import subprocess
from datetime import datetime

EUROC_SEQUENCES = {
    "MH01": "MH_01_easy",
    "MH02": "MH_02_easy",
    "MH03": "MH_03_medium",
    "MH04": "MH_04_difficult",
    "MH05": "MH_05_difficult",
    "V101": "V1_01_easy",
    "V102": "V1_02_medium",
    "V103": "V1_03_difficult",
    "V201": "V2_01_easy",
    "V202": "V2_02_medium",
    "V203": "V2_03_difficult",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_anylong(image_dir, save_dir, config_path):
    """Run Any-Long (VGGT-Long) on a single sequence."""
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "vggt_long.py"),
        "--image_dir", image_dir,
        "--config", config_path,
        "--save_dir", save_dir,
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


def convert_poses(camera_poses_txt, image_dir, euroc_seq_path, output_dir):
    """Convert Any-Long output to MAC-Loop protocol."""
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "convert_to_macloop.py"),
        "--camera_poses", camera_poses_txt,
        "--image_dir", image_dir,
        "--euroc_seq", euroc_seq_path,
        "--output_dir", output_dir,
    ]
    print(f"Converting: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Benchmark Any-Long on EuRoC")
    parser.add_argument("--euroc_root", type=str, required=True,
                        help="Root directory of EuRoC dataset (contains MH_01_easy/ etc.)")
    parser.add_argument("--sequences", nargs="+", default=None,
                        help="Specific sequences to run (e.g. MH01 MH02). Default: all")
    parser.add_argument("--results_root", type=str, default="/mnt/data/anyslam/macloop/vggt-long-results",
                        help="Root directory for MAC-Loop protocol results")
    parser.add_argument("--config", type=str, default=os.path.join(SCRIPT_DIR, "configs", "base_config.yaml"),
                        help="Any-Long config YAML path")
    parser.add_argument("--raw_output_root", type=str, default="/mnt/data/anyslam/macloop/vggt-long-results/raw_output",
                        help="Root directory for raw Any-Long outputs (temp files, camera_poses.txt)")
    parser.add_argument("--convert_only", action="store_true",
                        help="Skip inference, only convert existing camera_poses.txt files")
    parser.add_argument("--raw_results_dir", type=str, default=None,
                        help="Directory containing raw Any-Long outputs (for --convert_only). "
                             "Should contain camera_poses.txt")

    args = parser.parse_args()

    # Determine which sequences to run
    if args.sequences:
        sequences = {k: EUROC_SEQUENCES[k] for k in args.sequences if k in EUROC_SEQUENCES}
        unknown = [k for k in args.sequences if k not in EUROC_SEQUENCES]
        if unknown:
            print(f"WARNING: Unknown sequences: {unknown}")
            print(f"Available: {list(EUROC_SEQUENCES.keys())}")
    else:
        sequences = EUROC_SEQUENCES

    # Create timestamp directory for protocol output
    timestamp = datetime.now().strftime("%m_%d_%H%M%S")
    protocol_dir = os.path.join(args.results_root, "VGGT-Long", timestamp)
    raw_output_dir = os.path.join(args.raw_output_root, timestamp)
    os.makedirs(protocol_dir, exist_ok=True)
    os.makedirs(raw_output_dir, exist_ok=True)

    print(f"EuRoC root:      {args.euroc_root}")
    print(f"Sequences:       {list(sequences.keys())}")
    print(f"Config:          {args.config}")
    print(f"Raw output:      {raw_output_dir}")
    print(f"Protocol output: {protocol_dir}")
    print()

    results_summary = []
    total_start = time.time()

    for seq_short, seq_full in sequences.items():
        seq_start = time.time()
        euroc_seq_path = os.path.join(args.euroc_root, seq_full)
        image_dir = os.path.join(euroc_seq_path, "mav0", "cam0", "data")

        if not os.path.isdir(image_dir):
            print(f"SKIP: Image folder not found: {image_dir}")
            results_summary.append((seq_short, "SKIP", 0))
            continue

        protocol_seq_dir = os.path.join(protocol_dir, f"VGGT-Long@{seq_short}")

        # Step 1: Run Any-Long (unless convert_only)
        if args.convert_only:
            if args.raw_results_dir:
                seq_raw_dir = os.path.join(args.raw_results_dir, seq_short)
            else:
                print(f"SKIP: --convert_only requires --raw_results_dir")
                results_summary.append((seq_short, "NO_RAW_DIR", 0))
                continue
        else:
            seq_raw_dir = os.path.join(raw_output_dir, seq_short)
            result = run_anylong(image_dir, seq_raw_dir, args.config)
            if result is None:
                results_summary.append((seq_short, "SLAM_FAIL", time.time() - seq_start))
                continue

        # Step 2: Convert to MAC-Loop protocol
        camera_poses_txt = os.path.join(seq_raw_dir, "camera_poses.txt")
        if not os.path.exists(camera_poses_txt):
            print(f"SKIP: camera_poses.txt not found: {camera_poses_txt}")
            results_summary.append((seq_short, "NO_POSES", time.time() - seq_start))
            continue

        success = convert_poses(camera_poses_txt, image_dir, euroc_seq_path, protocol_seq_dir)
        elapsed = time.time() - seq_start
        status = "OK" if success else "CONVERT_FAIL"
        results_summary.append((seq_short, status, elapsed))
        print(f"{seq_short}: {status} ({elapsed:.1f}s)")

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
        print(f"  {seq:6s}  {status:12s}  {elapsed:7.1f}s")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results at: {protocol_dir}")


if __name__ == "__main__":
    main()
