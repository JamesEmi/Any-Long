"""
vggt_long_rr.py — VGGT-Long with Rerun visualization.

Runs the same pipeline as vggt_long.py, then visualises in Rerun:
  - trajectories/pred        (white)   : camera path from sequential chunk alignment, before loop closure
  - trajectories/final       (cyan)    : camera path after Sim3 loop-closure optimization
  - trajectories/gt          (green)   : KITTI ground-truth trajectory, aligned to model frame
  - map/pointcloud           (RGB)     : confidence-filtered, loop-closure-corrected point cloud

Usage:
    python vggt_long_rr.py \
        --image_dir /mnt/data/anyslam/KITTI_odometry/dataset/sequences/07/image_2 \
        --config configs/kitti.yaml \
        --save_dir /mnt/data/slam-proj/exps/kitti07_rr \
        --kitti_poses /mnt/data/anyslam/KITTI_odometry/dataset/poses/07.txt \
        --save /mnt/data/slam-proj/exps/kitti07_rr/kitti07.rrd
"""

import copy
import gc
import glob
import os
import sys
import numpy as np
import torch

import rerun as rr

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from vggt_long import VGGT_Long, copy_file
from loop_utils.sim3utils import accumulate_sim3_transforms, estimate_sim3, merge_ply_files, warmup_numba
from loop_utils.config_utils import load_config
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_kitti_poses(pose_path: str) -> np.ndarray:
    """Load KITTI ground-truth poses. Returns [N, 4, 4] float64."""
    raw = np.loadtxt(pose_path).reshape(-1, 3, 4)
    N = raw.shape[0]
    poses = np.zeros((N, 4, 4), dtype=np.float64)
    poses[:, :3, :4] = raw
    poses[:, 3, 3] = 1.0
    return poses


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class VGGT_Long_RR(VGGT_Long):
    """Extends VGGT_Long with Rerun visualisation hooks."""

    def __init__(self, image_dir, save_dir, config, kitti_poses_path=None, downsample=1):
        super().__init__(image_dir, save_dir, config, downsample)
        self.kitti_poses_path = kitti_poses_path
        self._rr_time = 0
        # Snapshots of the sequential sim3 list (before accumulation)
        self._pre_loop_sim3 = None   # captured before loop-closure optimisation
        self._post_loop_sim3 = None  # captured after  loop-closure optimisation

    # ------------------------------------------------------------------
    # Override process_long_sequence to capture intermediate sim3 states
    # ------------------------------------------------------------------

    def process_long_sequence(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"[SETTING ERROR] Overlap ({self.overlap}) must be less than chunk size ({self.chunk_size})"
            )
        if len(self.img_list) <= self.chunk_size:
            num_chunks = 1
            self.chunk_indices = [(0, len(self.img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            self.chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                self.chunk_indices.append((start_idx, end_idx))

        for chunk_idx in range(len(self.chunk_indices)):
            print(f'[Progress]: {chunk_idx}/{len(self.chunk_indices)-1}')
            self.process_single_chunk(self.chunk_indices[chunk_idx], chunk_idx=chunk_idx)
            torch.cuda.empty_cache()

        if self.loop_enable:
            from vggt_long import remove_duplicates
            from loop_utils.sim3utils import process_loop_list
            print('Loop SIM(3) estimating...')
            loop_results = process_loop_list(
                self.chunk_indices,
                self.loop_list,
                half_window=int(self.config['Model']['loop_chunk_size'] / 2),
            )
            loop_results = remove_duplicates(loop_results)
            print(loop_results)
            for item in loop_results:
                single_chunk_predictions = self.process_single_chunk(
                    item[1], range_2=item[3], is_loop=True
                )
                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)
        print(
            f"Processing {len(self.img_list)} images in {num_chunks} chunks "
            f"of size {self.chunk_size} with {self.overlap} overlap"
        )

        del self.model
        torch.cuda.empty_cache()

        print("Aligning all the chunks...")
        from loop_utils.sim3utils import weighted_align_point_maps
        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(f"Aligning {chunk_idx} and {chunk_idx+1} (Total {len(self.chunk_indices)-1})")
            chunk_data1 = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"),
                allow_pickle=True,
            ).item()
            chunk_data2 = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"),
                allow_pickle=True,
            ).item()

            point_map1 = chunk_data1['world_points'][-self.overlap:]
            point_map2 = chunk_data2['world_points'][:self.overlap]
            conf1 = chunk_data1['world_points_conf'][-self.overlap:]
            conf2 = chunk_data2['world_points_conf'][:self.overlap]

            mask = None
            if chunk_data1["mask"] is not None:
                mask1 = chunk_data1["mask"][-self.overlap:]
                mask2 = chunk_data2["mask"][:self.overlap]
                mask = mask1.squeeze() & mask2.squeeze()

            conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
            s, R, t = weighted_align_point_maps(
                point_map1, conf1, point_map2, conf2, mask,
                conf_threshold=conf_threshold, config=self.config,
            )
            print("Estimated Scale:", s)
            print("Estimated Rotation:\n", R)
            print("Estimated Translation:", t)
            self.sim3_list.append((s, R, t))

        # ---- CAPTURE 1: sequential sim3 before loop-closure optimisation ----
        self._pre_loop_sim3 = copy.deepcopy(self.sim3_list)

        if self.loop_enable:
            from loop_utils.sim3utils import compute_sim3_ab
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            for item in self.loop_predict_list:
                chunk_idx_a = item[0][0]
                chunk_idx_b = item[0][2]
                chunk_a_range = item[0][1]
                chunk_b_range = item[0][3]

                print('chunk_a align')
                point_map_loop = item[1]['world_points'][:chunk_a_range[1] - chunk_a_range[0]]
                conf_loop = item[1]['world_points_conf'][:chunk_a_range[1] - chunk_a_range[0]]
                chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
                chunk_a_rela_end = chunk_a_rela_begin + chunk_a_range[1] - chunk_a_range[0]
                print(self.chunk_indices[chunk_idx_a])
                print(chunk_a_range)
                print(chunk_a_rela_begin, chunk_a_rela_end)
                chunk_data_a = np.load(
                    os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"),
                    allow_pickle=True,
                ).item()

                point_map_a = chunk_data_a['world_points'][chunk_a_rela_begin:chunk_a_rela_end]
                conf_a = chunk_data_a['world_points_conf'][chunk_a_rela_begin:chunk_a_rela_end]

                conf_threshold = min(np.median(conf_a), np.median(conf_loop)) * 0.1
                mask = None
                if item[1]['mask'] is not None:
                    mask_loop = item[1]['mask'][:chunk_a_range[1] - chunk_a_range[0]]
                    mask_a = chunk_data_a['mask'][chunk_a_rela_begin:chunk_a_rela_end]
                    mask = mask_loop.squeeze() & mask_a.squeeze()
                s_a, R_a, t_a = weighted_align_point_maps(
                    point_map_a, conf_a, point_map_loop, conf_loop, mask,
                    conf_threshold=conf_threshold, config=self.config,
                )
                print("Estimated Scale:", s_a)
                print("Estimated Rotation:\n", R_a)
                print("Estimated Translation:", t_a)

                print('chunk_b align')
                point_map_loop = item[1]['world_points'][-chunk_b_range[1] + chunk_b_range[0]:]
                conf_loop = item[1]['world_points_conf'][-chunk_b_range[1] + chunk_b_range[0]:]
                chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
                chunk_b_rela_end = chunk_b_rela_begin + chunk_b_range[1] - chunk_b_range[0]
                print(self.chunk_indices[chunk_idx_b])
                print(chunk_b_range)
                print(chunk_b_rela_begin, chunk_b_rela_end)
                chunk_data_b = np.load(
                    os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"),
                    allow_pickle=True,
                ).item()

                point_map_b = chunk_data_b['world_points'][chunk_b_rela_begin:chunk_b_rela_end]
                conf_b = chunk_data_b['world_points_conf'][chunk_b_rela_begin:chunk_b_rela_end]

                conf_threshold = min(np.median(conf_b), np.median(conf_loop)) * 0.1
                mask = None
                if item[1]['mask'] is not None:
                    mask_loop = item[1]['mask'][-chunk_b_range[1] + chunk_b_range[0]:]
                    mask_b = chunk_data_b['mask'][chunk_b_rela_begin:chunk_b_rela_end]
                    mask = mask_loop.squeeze() & mask_b.squeeze()
                s_b, R_b, t_b = weighted_align_point_maps(
                    point_map_b, conf_b, point_map_loop, conf_loop, mask,
                    conf_threshold=conf_threshold, config=self.config,
                )
                print("Estimated Scale:", s_b)
                print("Estimated Rotation:\n", R_b)
                print("Estimated Translation:", t_b)

                print('a -> b SIM(3)')
                s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
                print("Estimated Scale:", s_ab)
                print("Estimated Rotation:\n", R_ab)
                print("Estimated Translation:", t_ab)
                self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

        if self.loop_enable:
            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)
            self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)

            # ---- CAPTURE 2: sequential sim3 after loop-closure optimisation ----
            self._post_loop_sim3 = copy.deepcopy(self.sim3_list)

            def extract_xyz(pose_tensor):
                poses = pose_tensor.cpu().numpy()
                return poses[:, 0], poses[:, 1], poses[:, 2]

            x0, _, y0 = extract_xyz(input_abs_poses)
            x1, _, y1 = extract_xyz(optimized_abs_poses)

            plt.figure(figsize=(8, 6))
            plt.plot(x0, y0, 'o--', alpha=0.45, label='Before Optimization')
            plt.plot(x1, y1, 'o-', label='After Optimization')
            for i, j, _ in self.loop_sim3_list:
                plt.plot([x0[i], x0[j]], [y0[i], y0[j]], 'r--', alpha=0.25,
                         label='Loop (Before)' if i == 5 else "")
                plt.plot([x1[i], x1[j]], [y1[i], y1[j]], 'g-', alpha=0.35,
                         label='Loop (After)' if i == 5 else "")
            plt.gca().set_aspect('equal')
            plt.title("Sim3 Loop Closure Optimization")
            plt.xlabel("x")
            plt.ylabel("z")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            save_path = os.path.join(self.output_dir, 'sim3_opt_result.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            # No loop closure: post == pre
            self._post_loop_sim3 = copy.deepcopy(self.sim3_list)

        print('Apply alignment')
        from loop_utils.sim3utils import apply_sim3_direct, save_confident_pointcloud_batch
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)
        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(f'Applying {chunk_idx + 1} -> {chunk_idx} (Total {len(self.chunk_indices) - 1})')
            s, R, t = self.sim3_list[chunk_idx]

            chunk_data = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx + 1}.npy"),
                allow_pickle=True,
            ).item()
            chunk_data['world_points'] = apply_sim3_direct(chunk_data['world_points'], s, R, t)

            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx + 1}.npy")
            np.save(aligned_path, chunk_data)

            if chunk_idx == 0:
                chunk_data_first = np.load(
                    os.path.join(self.result_unaligned_dir, "chunk_0.npy"),
                    allow_pickle=True,
                ).item()
                np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)

                points_first = chunk_data_first['world_points'].reshape(-1, 3)
                colors_first = (
                    chunk_data_first['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255
                ).astype(np.uint8)
                confs_first = chunk_data_first['world_points_conf'].reshape(-1)
                save_confident_pointcloud_batch(
                    points=points_first,
                    colors=colors_first,
                    confs=confs_first,
                    output_path=os.path.join(self.pcd_dir, '0_pcd.ply'),
                    conf_threshold=np.mean(confs_first)
                    * self.config['Model']['Pointcloud_Save']['conf_threshold_coef'],
                    sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio'],
                )

            aligned_chunk_data = (
                np.load(
                    os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy"),
                    allow_pickle=True,
                ).item()
                if chunk_idx > 0
                else chunk_data_first
            )
            points = aligned_chunk_data['world_points'].reshape(-1, 3)
            colors = (
                aligned_chunk_data['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255
            ).astype(np.uint8)
            confs = aligned_chunk_data['world_points_conf'].reshape(-1)
            save_confident_pointcloud_batch(
                points=points,
                colors=colors,
                confs=confs,
                output_path=os.path.join(self.pcd_dir, f'{chunk_idx + 1}_pcd.ply'),
                conf_threshold=np.mean(confs)
                * self.config['Model']['Pointcloud_Save']['conf_threshold_coef'],
                sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio'],
            )

        self.save_camera_poses()

        # ---- Rerun visualisation ----
        self._log_all_rerun()

        print('Done.')

    # ------------------------------------------------------------------
    # Rerun helpers
    # ------------------------------------------------------------------

    def _extract_camera_positions(self, sim3_list_sequential):
        """Return [N_frames, 3] camera positions for all chunks.

        Args:
            sim3_list_sequential: sequential (pre-accumulation) sim3 list with N_chunks-1 entries.
        """
        if not sim3_list_sequential:
            # Single chunk – no alignment needed
            _, extrinsics = self.all_camera_poses[0]
            return np.array([c2w[:3, 3] for c2w in extrinsics], dtype=np.float32)

        accumulated = accumulate_sim3_transforms(sim3_list_sequential)

        all_positions = []
        # Chunk 0: already in the reference frame
        _, first_extrinsics = self.all_camera_poses[0]
        for c2w in first_extrinsics:
            all_positions.append(c2w[:3, 3])

        # Chunks 1..N: apply cumulative SIM(3)
        for chunk_idx in range(1, len(self.all_camera_poses)):
            _, extrinsics = self.all_camera_poses[chunk_idx]
            s, R, t = accumulated[chunk_idx - 1]
            S = np.eye(4, dtype=np.float64)
            S[:3, :3] = s * R
            S[:3, 3] = t
            for c2w in extrinsics:
                transformed = S @ c2w.astype(np.float64)
                # Position is unaffected by rotation normalisation
                all_positions.append(transformed[:3, 3])

        return np.array(all_positions, dtype=np.float32)

    def _log_pointcloud_rerun(self):
        """Load loop-closure-corrected aligned npy files and log to Rerun."""
        conf_coef = self.config['Model']['Pointcloud_Save']['conf_threshold_coef']
        sample_ratio = self.config['Model']['Pointcloud_Save']['sample_ratio']
        step = max(1, int(round(1.0 / sample_ratio))) if sample_ratio < 1.0 else 1

        all_points = []
        all_colors = []

        for chunk_idx in range(len(self.chunk_indices)):
            # Aligned dir has chunks 0..N (chunk 0 is copied there too)
            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx}.npy")
            if not os.path.exists(aligned_path):
                # Fallback: only one chunk, lives in unaligned dir
                aligned_path = os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy")
            if not os.path.exists(aligned_path):
                print(f"[Rerun] Warning: chunk {chunk_idx} npy not found, skipping.")
                continue

            chunk_data = np.load(aligned_path, allow_pickle=True).item()

            points = chunk_data['world_points'].reshape(-1, 3).astype(np.float32)
            colors = (
                chunk_data['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255
            ).astype(np.uint8)
            confs = chunk_data['world_points_conf'].reshape(-1)

            conf_threshold = np.mean(confs) * conf_coef
            mask = confs >= conf_threshold

            all_points.append(points[mask][::step])
            all_colors.append(colors[mask][::step])

        if not all_points:
            print("[Rerun] No point cloud data to log.")
            return

        pts = np.concatenate(all_points, axis=0)
        cols = np.concatenate(all_colors, axis=0)

        MAX_PTS = 3_000_000
        if len(pts) > MAX_PTS:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(pts), MAX_PTS, replace=False)
            pts = pts[idx]
            cols = cols[idx]

        rr.set_time("stable_time", sequence=self._rr_time)
        rr.log("map/pointcloud", rr.Points3D(positions=pts, colors=cols))
        print(f"[Rerun] Logged {len(pts):,} points to map/pointcloud at t={self._rr_time}")

    def _log_gt_rerun(self, pred_positions):
        """Load KITTI GT poses, Umeyama-align to model frame, log as green static."""
        poses = _load_kitti_poses(self.kitti_poses_path)
        n = min(len(poses), len(pred_positions))
        gt_pos = poses[:n, :3, 3].astype(np.float32)
        pred_pos = pred_positions[:n]

        # estimate_sim3(source, target): target ≈ s * R @ source + t
        # We want pred ≈ s * R @ gt + t  →  source=gt, target=pred
        s, R, t = estimate_sim3(gt_pos, pred_pos)
        gt_model = (s * (R @ gt_pos.T).T + t).astype(np.float32)

        rr.log(
            "trajectories/gt",
            rr.Points3D(positions=gt_model, colors=np.tile([0, 255, 0], (len(gt_model), 1))),
            static=True,
        )
        rr.log(
            "trajectories/gt_line",
            rr.LineStrips3D([gt_model], colors=[[0, 255, 0]]),
            static=True,
        )
        print(f"[Rerun] Logged GT trajectory ({len(gt_model)} poses) to trajectories/gt")

    def _log_all_rerun(self):
        """Log pred/final trajectories, GT trajectory, and corrected point cloud."""
        rr.log("map", rr.ViewCoordinates.RDF, static=True)

        # ------------------------------------------------------------------
        # t=0 : predicted trajectory (sequential alignment, white)
        # ------------------------------------------------------------------
        rr.set_time("stable_time", sequence=0)
        self._rr_time = 0

        pred_positions = self._extract_camera_positions(self._pre_loop_sim3)
        rr.log(
            "trajectories/pred",
            rr.Points3D(
                positions=pred_positions,
                colors=np.tile([255, 255, 255], (len(pred_positions), 1)),
            ),
        )
        rr.log(
            "trajectories/pred_line",
            rr.LineStrips3D([pred_positions], colors=[[255, 255, 255]]),
        )
        print(f"[Rerun] Logged pred trajectory ({len(pred_positions)} poses) at t=0")

        # GT (static — visible across all timeline steps)
        if self.kitti_poses_path:
            self._log_gt_rerun(pred_positions)

        # ------------------------------------------------------------------
        # t=1 : final trajectory (loop-closure corrected, cyan) + point cloud
        # ------------------------------------------------------------------
        self._rr_time = 1
        rr.set_time("stable_time", sequence=1)

        final_positions = self._extract_camera_positions(self._post_loop_sim3)
        rr.log(
            "trajectories/final",
            rr.Points3D(
                positions=final_positions,
                colors=np.tile([0, 220, 255], (len(final_positions), 1)),
            ),
        )
        rr.log(
            "trajectories/final_line",
            rr.LineStrips3D([final_positions], colors=[[0, 220, 255]]),
        )
        print(f"[Rerun] Logged final trajectory ({len(final_positions)} poses) at t=1")

        self._log_pointcloud_rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='VGGT-Long with Rerun visualisation')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to directory containing input images')
    parser.add_argument('--config', type=str, default='./configs/base_config.yaml',
                        help='Path to config YAML (default: ./configs/base_config.yaml)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Output directory (auto-generated under ./exps/ if not set)')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Keep every Nth image (1 = no downsampling)')
    parser.add_argument('--kitti_poses', type=str, default=None,
                        help='Path to KITTI ground-truth poses txt (optional)')
    # Rerun: adds --save / --rr-addr / --connect / --serve flags
    rr.script_add_args(parser)

    args = parser.parse_args()

    config = load_config(args.config)

    image_dir = args.image_dir
    if args.save_dir:
        save_dir = args.save_dir
    else:
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_dir = os.path.join('./exps', image_dir.replace("/", "_"), current_datetime)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'Experiment will be saved under: {save_dir}')
        copy_file(args.config, save_dir)

    if config['Model']['align_method'] == 'numba':
        warmup_numba()

    # Initialise Rerun recording
    rr.script_setup(args, "vggt_long_rr")

    vggt_long_rr = VGGT_Long_RR(
        image_dir=image_dir,
        save_dir=save_dir,
        config=config,
        kitti_poses_path=args.kitti_poses,
        downsample=args.downsample,
    )
    vggt_long_rr.run()
    vggt_long_rr.close()

    del vggt_long_rr
    torch.cuda.empty_cache()
    gc.collect()

    all_ply_path = os.path.join(save_dir, 'pcd/combined_pcd.ply')
    input_dir = os.path.join(save_dir, 'pcd')
    print("Merging individual chunk PLY files...")
    merge_ply_files(input_dir, all_ply_path)

    rr.script_teardown(args)
    print('All done.')
    sys.exit()
