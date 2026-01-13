import numpy as np
import argparse
import time

import os
import glob
import threading
import torch
from tqdm.auto import tqdm
import cv2
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
base_models_path = os.path.join(current_dir, 'base_models')
if base_models_path not in sys.path:
    sys.path.append(base_models_path)

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from LoopModels.LoopModel import LoopDetector
from LoopModelDBoW.retrieval.retrieval_dbow import RetrievalDBOW

from base_models.base_model import VGGTAdapter,Pi3Adapter,MapAnythingAdapter

import numpy as np

from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import *
from datetime import datetime

from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from loop_utils.config_utils import load_config
from pathlib import Path

try:
    import rerun as rr
    from loop_utils.rerun_utils import *
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False
    print("Warning: rerun-sdk not installed. Visualization features will be disabled.")

class StageTimer:
    """
    Context manager for timing pipeling stages with automatic logging.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.log_path = os.path.join(save_dir, 'timing_log.txt')
        self.timings = []
        self.stack = []

    def start(self, stage_name):
        """Start timing a stage."""
        start_time = time.time()
        self.stack.append((stage_name, start_time))
        print(f"[TIMING] Starting: {stage_name}")
        return self
    
    def end(self):
        """End timing a stage"""
        if not self.stack:
            return
    
        stage_name, start_time = self.stack.pop()
        elapsed = time.time() - start_time
        self.timings.append((stage_name, elapsed))
        print(f"[Timing] Completed: {stage_name} ({elapsed:.2f}s)")

    def save_log(self):
        """Write log to file"""
        if not self.timings:
            return
        
        with open(self.log_path, 'w') as f:
            f.write(f"VGGT-Long Timing Report\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")

            total_time = sum(t for _, t in self.timings)

            for stage, elapsed in self.timings:
                percentage = (elapsed / total_time) * 100 if total_time > 0 else 0
                f.write(f"{stage:40s} {elapsed:10.2f}s ({percentage:5.1f}%)\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)\n")

        print(f"[TIMING] Saved timing log to: {self.log_path}")



def remove_duplicates(data_list):
    """
        data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {} 
    result = []
    
    for item in data_list:
        if item[0] == item[2]:
            continue

        key = (item[0], item[2])
        
        if key not in seen.keys():
            seen[key] = True
            result.append(item)
    
    return result


def extract_p2_k_matrix(calib_path):
    """from calib.txt get K  (kitti)"""

    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('P2:'):
                values = line.split(':')[1].split()
                values = [float(v) for v in values]
                p2_matrix = np.array(values).reshape(3, 4)
                k_matrix = p2_matrix[:3, :3]
                return k_matrix, p2_matrix

    raise ValueError("P2 not found in calibration file")

class LongSeqResult:
    def __init__(self):
        self.combined_extrinsics = []
        self.combined_intrinsics = []
        self.combined_depth_maps = []
        self.combined_depth_confs = []
        self.combined_world_points = []
        self.combined_world_points_confs = []
        self.all_camera_poses = []
        self.all_camera_intrinsics = [] 

class VGGT_Long:
    def __init__(self, image_dir, save_dir, config, max_points=None, timer=None,
                    enable_viz=False, viz_mode='final', save_rrd=None):
        self.config = config
        self.max_points = max_points
        self.timer = timer if timer is not None else StageTimer(save_dir)

        # Visualization setup
        self.enable_viz = enable_viz and RERUN_AVAILABLE
        self.viz_mode = viz_mode
        self.save_rrd = save_rrd

        if self.enable_viz:
            rr.init("VGGT-Long Visualization", spawn=False)
            if self.save_rrd:
                rr.save(self.save_rrd)
                print(f"[VIZ] Rerun visualization enabled (mode: {viz_mode}, saving to: {save_rrd})")
            else:
                auto_rrd_path = os.path.join(save_dir, 'visualization.rrd')
                rr.save(auto_rrd_path)
                self.save_rrd = auto_rrd_path  # Store for reference
                print(f"[VIZ] Rerun visualization enabled (mode: {viz_mode}, saving to: {auto_rrd_path})")
            rr.log("world", rr.ViewCoordinates.RDF, static=True)
        elif enable_viz and not RERUN_AVAILABLE:
            print("[VIZ] Warning: Visualization requested but rerun-sdk not installed")

        self.chunk_size = self.config['Model']['chunk_size']
        self.overlap = self.config['Model']['overlap']
        self.conf_threshold = 1.5
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.sky_mask = False
        self.useDBoW = self.config['Model']['useDBoW']

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir

        self.result_unaligned_dir = os.path.join(save_dir, '_tmp_results_unaligned')
        self.result_aligned_dir = os.path.join(save_dir, '_tmp_results_aligned')
        self.result_loop_dir = os.path.join(save_dir, '_tmp_results_loop')
        self.pcd_dir = os.path.join(save_dir, 'pcd')
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)
        
        self.all_camera_poses = []
        self.all_camera_intrinsics = [] 
        
        self.delete_temp_files = self.config['Model']['delete_temp_files']

        if self.config['Weights']['model'] == 'VGGT':
            self.model = VGGTAdapter(self.config)
        elif self.config['Weights']['model'] == 'Pi3':
            self.model = Pi3Adapter(self.config)
        elif self.config['Weights']['model'] == 'Mapanything':
            self.model = MapAnythingAdapter(self.config)
        else:
            raise ValueError(f"Unsupported model: {self.config['Weights']['model']}. ")

        self.skyseg_session = None
        
        self.chunk_indices = None # [(begin_idx, end_idx), ...]

        self.loop_list = [] # e.g. [(1584, 139), ...]

        self.loop_optimizer = Sim3LoopOptimizer(self.config)

        self.sim3_list = [] # [(s [1,], R [3,3], T [3,]), ...]

        self.loop_sim3_list = [] # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]

        self.loop_predict_list = []

        self.loop_enable = self.config['Model']['loop_enable']

        if self.loop_enable:
            if self.useDBoW:
                self.retrieval = RetrievalDBOW(config=self.config)
            else:
                loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
                self.loop_detector = LoopDetector(
                    image_dir=image_dir,
                    output=loop_info_save_path,
                    config=self.config
                )

        print('init done.')

    def _estimate_total_points(self):
        """
        Estimate total points in final PLY based on image count and config.

        Returns rough estimate using:
        - Number of images
        - Image resolution from config (h x w)
        - Confidence threshold coefficient
        - Sample ratio
        """
        num_images = len(self.img_list)
        sample_ratio = self.config['Model']['Pointcloud_Save']['sample_ratio']
        conf_threshold_coef = self.config['Model']['Pointcloud_Save']['conf_threshold_coef']

        # Use configured image resolution
        height = self.config['Model']['h']
        width = self.config['Model']['w']
        pixels_per_image = height * width

        # Assume ~75% pass confidence threshold on average (based on conf_threshold_coef of 0.75)
        avg_valid_ratio = conf_threshold_coef

        estimated_points = num_images * pixels_per_image * avg_valid_ratio * sample_ratio

        return int(estimated_points)

    def get_loop_pairs(self):

        if self.useDBoW: # DBoW2
            for frame_id, img_path in tqdm(enumerate(self.img_list)):
                image_ori = np.array(Image.open(img_path))
                if len(image_ori.shape) == 2:
                    # gray to rgb
                    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)

                frame = image_ori # (height, width, 3)
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                self.retrieval(frame, frame_id)
                cands = self.retrieval.detect_loop(thresh=self.config['Loop']['DBoW']['thresh'], 
                                                   num_repeat=self.config['Loop']['DBoW']['num_repeat'])

                if cands is not None:
                    (i, j) = cands # e.g. cands = (812, 67)
                    self.retrieval.confirm_loop(i, j)
                    self.retrieval.found.clear()
                    self.loop_list.append(cands)

                self.retrieval.save_up_to(frame_id)

        else: # DNIO v2
            self.loop_detector.run()
            self.loop_list = self.loop_detector.get_loop_list()
    
    def _estimate_total_points(self):
        """
        Estimate total points in final PLY based on image count and config.

        Returns rough estimate using:
        - Number of images
        - Image resolution from config (h, w)
        - Average points per image (based on resolution and sample_ratio)
        - Confidence threshold coefficient
        """
        num_images = len(self.img_list)
        sample_ratio = self.config['Model']['Pointcloud_Save']['sample_ratio']
        conf_threshold_coef = self.config['Model']['Pointcloud_Save']['conf_threshold_coef']

        # Get image resolution from config
        height = self.config['Model']['h']
        width = self.config['Model']['w']
        pixels_per_image = height * width

        # Estimate fraction of pixels that pass confidence threshold
        # Using conf_threshold_coef as proxy (typically ~0.75)
        avg_valid_ratio = conf_threshold_coef

        estimated_points = num_images * pixels_per_image * avg_valid_ratio * sample_ratio

        return int(estimated_points)

    def _log_chunk_to_rerun(self, chunk_idx, chunk_data):
        """
        Log per-chunk results to Rerun (realtime mode).
        Uses timeline-based logging

        Args:
            chunk_idx: Index of the chunk
            chunk_data: Dictionary containing world_points, images, world_points_conf, extrinsic
        """
        if not self.enable_viz or self.viz_mode not in ['realtime', 'all']:
            return

        viz_config = self.config.get('Visualization', {})
        max_viz_points = viz_config.get('max_viz_points', 2_000_000)
        camera_frustum_scale = viz_config.get('camera_frustum_scale', 0.05)
        camera_axes_scale = viz_config.get('camera_axes_scale', 0.1)
        show_camera_axes = viz_config.get('show_camera_axes', True)
        downsample_frames = viz_config.get('downsample_frames', 1)

        # set timeline for this chunk (sequential time steps)
        rr.set_time_sequence("chunk_processing", chunk_idx)

        # Get chunk color for cameras/frustums
        chunk_color = get_chunk_color(chunk_idx)

        # Extract data
        world_points = chunk_data['world_points']  # (N, H, W, 3)
        images = chunk_data['images']  # (N, 3, H, W)
        confs = chunk_data['world_points_conf']  # (N, H, W)
        extrinsics = chunk_data['extrinsic']  # (N, 4, 4) C2W matrices
        intrinsics = chunk_data.get('intrinsic', None)  # (N, 3, 3) or None

        # import ipdb; ipdb.set_trace()
        num_frames = world_points.shape[0]
        height, width = world_points.shape[1], world_points.shape[2]

        # Reshape for point cloud
        points = world_points.reshape(-1, 3)
        colors = (images.transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
        confs_flat = confs.reshape(-1)
        
        if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
            # Filter by confidence
            conf_threshold = np.mean(confs_flat) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef']
            valid_mask = confs_flat > conf_threshold
            points_viz = points[valid_mask]
            colors_viz = colors[valid_mask]
        else:
            conf_threshold = -1.0
            points_viz = points # conf_threshold_coef
            colors_viz = colors

        # Downsample for visualization (each chunk gets full 2M budget)
        points_viz, colors_viz = downsample_for_viz(points_viz, colors_viz, max_viz_points)

        # Log point cloud to SAME entity path (timeline separates them)
        rr.log(
            "world/realtime/chunk_points",
            rr.Points3D(points_viz, colors=colors_viz, radii=0.01)
        )

        # Log cameras with chunk-specific color
        for frame_idx in range(0, num_frames, downsample_frames):
            c2w = extrinsics[frame_idx]
            entity_path = f"world/realtime/cameras/camera_{frame_idx}"

            # Log frustum
            if intrinsics is not None:
                K = intrinsics[frame_idx]
                log_camera_frustum(
                    entity_path + "/frustum",
                    c2w, K, chunk_color,
                    width, height,
                    scale=camera_frustum_scale
                )

            # Log camera axes
            if show_camera_axes:
                log_camera_axes(entity_path + "/axes", c2w, scale=camera_axes_scale)

        print(f"[VIZ] Logged chunk {chunk_idx} to Rerun ({len(points_viz):,} points, {num_frames} cameras)")

    def _log_alignment_to_rerun(self, aligned_poses):
        """
        Log alignment trajectory to Rerun (alignment mode).

        Shows two stages:
        1. Pre-alignment: Unaligned chunks with drift (timeline step 0)
        2. Post-alignment: Globally aligned reconstruction (timeline step 1)
        Use global downsampling to stay under max_viz_points.

        Args:
            aligned_poses: List of (chunk_idx, C2W_matrix) tuples for post-alignment poses
        """
        if not self.enable_viz or self.viz_mode not in ['alignment', 'all']:
            return
        
        viz_config = self.config.get('Visualization', {})
        max_viz_points = viz_config.get('max_viz_points', 2_000_000)
        camera_frustum_scale = viz_config.get('camera_frustum_scale', 0.1)
        camera_axes_scale = viz_config.get('camera_axes_scale', 0.1)
        show_camera_axes = viz_config.get('show_camera_axes', True)

        # stage 1a - pre-alignment (show drift)
        rr.set_time_sequence("alignment_stage", 1)  # 0=realtime, 1=alignment, 2=loop_closure, 3=final

        # Load unaligned chunk pcs (each chunk in its own coord frame)
        all_points_pre = []
        all_colors_pre = []
        unaligned_cam_positions = []

        for chunk_idx in range(len(aligned_poses)):
            chunk_file = os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy")
            if not os.path.exists(chunk_file):
                continue

            chunk_data = np.load(chunk_file, allow_pickle=True).item()
            world_points = chunk_data['world_points'] # (N, H, W, 3)
            images = chunk_data['images'] # (N, 3, H, W)
            confs = chunk_data['world_points_conf'] # (N, H, W)
            extrinsics = chunk_data['extrinsic']  # (N, 4, 4) - local C2W

            # Get first camera position (origin of this chunk's coordinate frame)
            unaligned_cam_positions.append(extrinsics[0, :3, 3])

            # Reshape
            points = world_points.reshape(-1, 3)
            colors = (images.transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
            confs_flat = confs.reshape(-1)

            if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
                conf_threshold = np.mean(confs_flat) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef']
            else:
                conf_threshold = -1.0
            valid_mask = confs_flat > conf_threshold

            all_points_pre.append(points[valid_mask])
            all_colors_pre.append(colors[valid_mask])
        
        if len(all_points_pre) > 0:
            # Concatenate and globally downsample
            all_points_pre = np.concatenate(all_points_pre, axis=0)
            all_colors_pre = np.concatenate(all_colors_pre, axis=0)
            all_points_pre_viz, all_colors_pre_viz = downsample_for_viz(all_points_pre, all_colors_pre, max_viz_points)

            # log pre-aligned pc
            rr.log(
                "world/alignment/unaligned_pc",
                rr.Points3D(all_points_pre_viz, colors = all_colors_pre_viz, radii=0.01)
            )

            # log prealigned "traj" (shows drift)
            unaligned_cam_positions = np.array(unaligned_cam_positions)
            rr.log(
                "world/alignment/unaligned_traj",
                rr.LineStrips3D([unaligned_cam_positions], colors=[[255, 0, 0]])  # Red = drift
            )
            rr.log(
                "world/alignment/unaligned_camera_positions",
                rr.Points3D(unaligned_cam_positions, colors=[[255, 0, 0]] * len(unaligned_cam_positions), radii=0.05)
            )

            print(f"[VIZ] Logged PRE-alignment to Rerun ({len(all_points_pre_viz):,} points, {len(unaligned_cam_positions)} chunks)")
        
        # stage 1b - post-alignment
        rr.set_time_sequence("alignment_stage", 1)

        all_points_post = []
        all_colors_post = []

        for chunk_idx, c2w in aligned_poses:
            chunk_file = os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy")
            if not os.path.exists(chunk_file):
                continue

            chunk_data = np.load(chunk_file, allow_pickle=True).item()
            world_points = chunk_data['world_points']
            images = chunk_data['images']
            confs = chunk_data['world_points_conf']

            points = world_points.reshape(-1, 3)
            colors = (images.transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
            confs_flat = confs.reshape(-1)

            if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
                conf_threshold = np.mean(confs_flat) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef']
            else:
                conf_threshold = -1.0    
            valid_mask = confs_flat > conf_threshold
            # transform points to world frame using aligned pose (c2w transforms chunk's local frame to world frame)
            points_local = points[valid_mask]  # (M, 3)
            points_world = (c2w[:3, :3] @ points_local.T).T + c2w[:3, 3]  # apply rot + trans

            all_points_post.append(points_world)
            all_colors_post.append(colors[valid_mask])

        if len(all_points_post) == 0:
            print('[VIZ] No pc found for alignment viz.')
            return
        
        all_points_post = np.concatenate(all_points_post, axis=0)
        all_colors_post = np.concatenate(all_colors_post, axis=0)

        all_points_post_viz, all_colors_post_viz = downsample_for_viz(all_points_post, all_colors_post, max_viz_points)
        rr.log(
            "world/alignment/aligned_pointcloud",
            rr.Points3D(all_points_post_viz, colors=all_colors_post_viz, radii=0.01)
        )

        # Extract trajectory positions for aligned cams
        positions = []
        colors_cam = []

        for chunk_idx, c2w in aligned_poses:
            position = c2w[:3, 3]
            positions.append(position)
            colors_cam.append(get_chunk_color(chunk_idx))

        positions = np.array(positions)
        colors_cam = np.array(colors_cam)

        # Log trajectory as line strip
        rr.log(
            "world/alignment/aligned_traj",
            rr.LineStrips3D([positions], colors=[[0, 255, 0]])  # Yellow trajectory
        )

        # Log camera positions as points
        rr.log(
            "world/alignment/aligned_camera_positions",
            rr.Points3D(positions, colors=colors_cam, radii=0.05)
        )

        for chunk_idx, c2w in aligned_poses:
            chunk_color = get_chunk_color(chunk_idx)
            entity_path = f"world/alignment/cameras/chunk_{chunk_idx}"

            # get intrinsics
            chunk_file = os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy")
            if os.path.exists(chunk_file):
                chunk_data = np.load(chunk_file, allow_pickle=True).item()
                intrinsics = chunk_data.get('intrinsic', None)
                if intrinsics is not None:
                    K = intrinsics[0]
                    height, width = chunk_data['world_points'].shape[1:3]

                    log_camera_frustum(
                        entity_path + '\frustum',
                        c2w, K, chunk_color,
                        width, height, 
                        scale = camera_frustum_scale
                    )
                    if show_camera_axes:
                        log_camera_axes(entity_path + "/axes", c2w, scale=camera_axes_scale)
        
        print(f"[VIZ] Logged alignment to Rerun ({len(all_points_post_viz):,} points globally downsampled, {len(positions)} poses)")

    def _log_loop_closure_to_rerun(self, optimized_poses, loop_connections):
        """
        Log loop closure connections to Rerun (final mode).

        Uses timeline separation and global downsampling to stay under max_viz_points.

        Args:
            optimized_poses: List of optimized C2W matrices (one per chunk)
            loop_connections: List of (chunk_idx_a, chunk_idx_b, sim3) tuples
        """
        if not self.enable_viz or self.viz_mode not in ['final', 'all']:
            return

        viz_config = self.config.get('Visualization', {})
        max_viz_points = viz_config.get('max_viz_points', 2_000_000)
        camera_frustum_scale = viz_config.get('camera_frustum_scale', 0.1)
        camera_axes_scale = viz_config.get('camera_axes_scale', 0.1)
        show_camera_axes = viz_config.get('show_camera_axes', True)

        rr.set_time_sequence("pipeline_stage", 2)  # 0=realtime, 1=alignment, 2=loop_closure, 3=final

        all_points_loop = []
        all_colors_loop = []

        for chunk_idx, optimized_c2w in enumerate(optimized_poses):
            chunk_file = os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy")
            if not os.path.exists(chunk_file):
                continue

            chunk_data = np.load(chunk_file, allow_pickle=True).item()
            world_points = chunk_data['world_points']  # (N, H, W, 3)
            images = chunk_data['images']  # (N, 3, H, W)
            confs = chunk_data['world_points_conf']  # (N, H, W)

            # Reshape
            points = world_points.reshape(-1, 3)
            colors = (images.transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
            confs_flat = confs.reshape(-1)
                
            if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
                # Filter by confidence
                conf_threshold = np.mean(confs_flat) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef']
            else:
                conf_threshold = -1.0
            valid_mask = confs_flat > conf_threshold
            # import ipdb; ipdb.set_trace()
            # Transform points using optimized pose
            points_local = points[valid_mask]
            points_world = (optimized_c2w[:3, :3] @ points_local.T).T + optimized_c2w[:3, 3]

            all_points_loop.append(points_world)
            all_colors_loop.append(colors[valid_mask])
        
        if len(all_points_loop) > 0:
            # Concatenate and globally downsample
            all_points_loop = np.concatenate(all_points_loop, axis=0)
            all_colors_loop = np.concatenate(all_colors_loop, axis=0)
            all_points_loop_viz, all_colors_loop_viz = downsample_for_viz(all_points_loop, all_colors_loop, max_viz_points)

            # Log loop-optimized point cloud
            rr.log(
                "world/loop_closure/optimized_pointcloud",
                rr.Points3D(all_points_loop_viz, colors=all_colors_loop_viz, radii=0.01)
            )

        # Extract positions from optimized poses
        positions = np.array([pose[:3, 3] for pose in optimized_poses])

         # optimized traj
        rr.log(
            "world/loop_closure/optimized_trajectory",
            rr.LineStrips3D([positions], colors=[[0, 255, 255]])  # Cyan = loop-optimized
        )

        # camera positions
        colors_cam = [get_chunk_color(i) for i in range(len(positions))]
        rr.log(
            "world/loop_closure/camera_positions",
            rr.Points3D(positions, colors=colors_cam, radii=0.05)
        )

        # Log loop closure connections as lines
        for idx_a, idx_b, _ in loop_connections:
            line = np.array([positions[idx_a], positions[idx_b]])
            rr.log(
                f"world/loop_closures/connections/loop_{idx_a}_{idx_b}",
                rr.LineStrips3D([line], colors=[[255, 0, 255, 200]])  # Green with transparency
            )
        
        # Log camera frustums for optimized poses
        for chunk_idx, optimized_c2w in enumerate(optimized_poses):
            chunk_color = get_chunk_color(chunk_idx)
            entity_path = f"world/loop_closure/cameras/chunk_{chunk_idx}"

            chunk_file = os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy")
            if os.path.exists(chunk_file):
                chunk_data = np.load(chunk_file, allow_pickle=True).item()
                intrinsics = chunk_data.get('intrinsic', None)
                if intrinsics is not None:
                    K = intrinsics[0]
                    height, width = chunk_data['world_points'].shape[1:3]

                    log_camera_frustum(
                        entity_path + "/frustum",
                        optimized_c2w, K, chunk_color,
                        width, height,
                        scale=camera_frustum_scale
                    )

                    if show_camera_axes:
                        log_camera_axes(entity_path + "/axes", optimized_c2w, scale=camera_axes_scale)

        print(f"[VIZ] Logged loop closure to Rerun ({len(all_points_loop_viz):,} points, {len(loop_connections)} loop connections)")

    def _log_final_pointcloud_to_rerun(self, ply_path):
        """
        Log final combined point cloud to Rerun (final mode).
        Uses timeline separation to avoid accumulation with other stages.
        Args:
            ply_path: Path to final combined PLY file
        """
        if not self.enable_viz or self.viz_mode not in ['final', 'all']:
            return

        viz_config = self.config.get('Visualization', {})
        max_viz_points = viz_config.get('max_viz_points', 2_000_000)
        rr.set_time_sequence("pipeline_stage", 3)  # 0=realtime, 1=alignment, 2=loop_closure, 3=final
        points, colors = read_ply_points_colors(ply_path)

        points_viz, colors_viz = downsample_for_viz(points, colors, max_viz_points)
        
        rr.log(
            "world/final/combined_pointcloud",
            rr.Points3D(points_viz, colors=colors_viz, radii=0.01)
        )

        print(f"[VIZ] Logged final point cloud to Rerun ({len(points_viz):,} points)")


    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        predictions = self.model.infer_chunk(chunk_image_paths)
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)
        
        # Save predictions to disk instead of keeping in memory
        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"
        
        save_path = os.path.join(save_dir, filename)
                    
        if not is_loop and range_2 is None:
            extrinsics = predictions['extrinsic']
            intrinsics = predictions['intrinsic']
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))

        predictions['depth'] = np.squeeze(predictions['depth'])

        np.save(save_path, predictions)
        
        return predictions if is_loop or range_2 is not None else None
    
    def process_long_sequence(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(f"[SETTING ERROR] Overlap ({self.overlap}) must be less than chunk size ({self.chunk_size})")
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
            if self.timer:
                self.timer.start(f"Chunk {chunk_idx} Inference")
            print(f'[Progress]: {chunk_idx}/{len(self.chunk_indices)-1}')
            self.process_single_chunk(self.chunk_indices[chunk_idx], chunk_idx=chunk_idx)

            if self.enable_viz and self.viz_mode in ['realtime', 'all']:
                chunk_data = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item()
                self._log_chunk_to_rerun(chunk_idx, chunk_data)

            torch.cuda.empty_cache()

            if self.timer:
                self.timer.end()


        if self.loop_enable:
            print('Loop SIM(3) estimating...')
            loop_results = process_loop_list(self.chunk_indices,
                                             self.loop_list,
                                             half_window = int(self.config['Model']['loop_chunk_size'] / 2))
            loop_results = remove_duplicates(loop_results)
            print(loop_results)
            # return e.g. (31, (1574, 1594), 2, (129, 149))
            for item in loop_results:
                single_chunk_predictions = self.process_single_chunk(item[1], range_2=item[3], is_loop=True)

                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)
        print(
            f"Processing {len(self.img_list)} images in {num_chunks} chunks of size {self.chunk_size} with {self.overlap} overlap")

        del self.model # Save GPU Memory
        torch.cuda.empty_cache()

        if self.timer:
            self.timer.start("Chunk-to-Chunk Alignment")

        print("Aligning all the chunks...")
        for chunk_idx in range(len(self.chunk_indices)-1):

            print(f"Aligning {chunk_idx} and {chunk_idx+1} (Total {len(self.chunk_indices)-1})")
            chunk_data1 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item()
            chunk_data2 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"), allow_pickle=True).item()
            
            point_map1 = chunk_data1['world_points'][-self.overlap:]
            point_map2 = chunk_data2['world_points'][:self.overlap]
            conf1 = chunk_data1['world_points_conf'][-self.overlap:]
            conf2 = chunk_data2['world_points_conf'][:self.overlap]

            mask = None
            if chunk_data1["mask"] is not None:
                mask1 = chunk_data1["mask"][-self.overlap:]
                mask2 = chunk_data2["mask"][:self.overlap]
                mask = mask1.squeeze() & mask2.squeeze()
            
            # TODO: add the condition for Mapanything here.
            # TODO: If alignment relies so heavily on model provided confidences; 
            # and the one via pred['conf'] for Mapanything is unreliable; what to do?
            if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
                conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
            else:
                conf_threshold = -1.0
            s, R, t = weighted_align_point_maps(point_map1, 
                                                conf1, 
                                                point_map2, 
                                                conf2,
                                                mask,
                                                conf_threshold=conf_threshold,
                                                config=self.config)
            print("Estimated Scale:", s)
            print("Estimated Rotation:\n", R)
            print("Estimated Translation:", t)

            self.sim3_list.append((s, R, t))

        if self.enable_viz and self.viz_mode in ['alignment', 'all']:
            accum_transforms = accumulate_sim3_transforms(self.sim3_list)
            aligned_poses = []
            # first chunk
            aligned_poses.append((0, np.eye(4))) #is this correct? what does it do?
            # subsequent chunks w accumulated transforms
            for chunk_idx, (s, R, t) in enumerate(accum_transforms): #TODO - Check data flow here.
                T = np.eye(4)
                T[:3, :3] = s*R
                T[:3, 3] = t
                aligned_poses.append((chunk_idx + 1, T))
            self._log_alignment_to_rerun(aligned_poses)

        if self.timer:
            self.timer.end()  # End "Chunk-to-Chunk Alignment"

        if self.loop_enable:
            if self.timer:
                self.timer.start("Loop Closure Optimization")
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
                chunk_data_a = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"), allow_pickle=True).item()
                
                point_map_a = chunk_data_a['world_points'][chunk_a_rela_begin:chunk_a_rela_end]
                conf_a = chunk_data_a['world_points_conf'][chunk_a_rela_begin:chunk_a_rela_end]

                if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
                    conf_threshold = min(np.median(conf_a), np.median(conf_loop)) * 0.1
                else:
                    conf_threshold = -1.0
                mask = None
                if item[1]['mask'] is not None:
                    mask_loop = item[1]['mask'][:chunk_a_range[1] - chunk_a_range[0]]
                    mask_a = chunk_data_a['mask'][chunk_a_rela_begin:chunk_a_rela_end]
                    mask = mask_loop.squeeze() & mask_a.squeeze()
                s_a, R_a, t_a = weighted_align_point_maps(point_map_a, 
                                                          conf_a, 
                                                          point_map_loop, 
                                                          conf_loop,
                                                          mask,
                                                          conf_threshold=conf_threshold,
                                                          config=self.config)
                print("Estimated Scale:", s_a)
                print("Estimated Rotation:\n", R_a)
                print("Estimated Translation:", t_a)

                print('chunk_a align')
                point_map_loop = item[1]['world_points'][-chunk_b_range[1] + chunk_b_range[0]:]
                conf_loop = item[1]['world_points_conf'][-chunk_b_range[1] + chunk_b_range[0]:]
                chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
                chunk_b_rela_end = chunk_b_rela_begin + chunk_b_range[1] - chunk_b_range[0]
                print(self.chunk_indices[chunk_idx_b])
                print(chunk_b_range)
                print(chunk_b_rela_begin, chunk_b_rela_end)
                chunk_data_b = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"), allow_pickle=True).item()
                
                point_map_b = chunk_data_b['world_points'][chunk_b_rela_begin:chunk_b_rela_end]
                conf_b = chunk_data_b['world_points_conf'][chunk_b_rela_begin:chunk_b_rela_end]

                if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
                    conf_threshold = min(np.median(conf_b), np.median(conf_loop)) * 0.1
                else:
                    conf_threshold = -1.0
                mask = None
                if item[1]['mask'] is not None:
                    mask_loop = item[1]['mask'][-chunk_b_range[1] + chunk_b_range[0]:]
                    mask_b = chunk_data_b['mask'][chunk_b_rela_begin:chunk_b_rela_end]
                    mask = mask_loop.squeeze() & mask_b.squeeze()
                s_b, R_b, t_b = weighted_align_point_maps(point_map_b, 
                                                          conf_b, 
                                                          point_map_loop, 
                                                          conf_loop,
                                                          mask,
                                                          conf_threshold=conf_threshold,
                                                          config=self.config)
                print("Estimated Scale:", s_b)
                print("Estimated Rotation:\n", R_b)
                print("Estimated Translation:", t_b)

                print('a -> b SIM 3')
                s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
                print("Estimated Scale:", s_ab)
                print("Estimated Rotation:\n", R_ab)
                print("Estimated Translation:", t_ab)

                self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))


        if self.loop_enable:
            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)
            self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)

            def extract_xyz(pose_tensor):
                poses = pose_tensor.cpu().numpy()
                return poses[:, 0], poses[:, 1], poses[:, 2]
            
            x0, _, y0 = extract_xyz(input_abs_poses)
            x1, _, y1 = extract_xyz(optimized_abs_poses)

            # Visual in png format
            plt.figure(figsize=(8, 6))
            plt.plot(x0, y0, 'o--', alpha=0.45, label='Before Optimization')
            plt.plot(x1, y1, 'o-', label='After Optimization')
            for i, j, _ in self.loop_sim3_list:
                plt.plot([x0[i], x0[j]], [y0[i], y0[j]], 'r--', alpha=0.25, label='Loop (Before)' if i == 5 else "")
                plt.plot([x1[i], x1[j]], [y1[i], y1[j]], 'g-', alpha=0.35, label='Loop (After)' if i == 5 else "")
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

            # Log loop closure to Rerun
            if self.enable_viz and self.viz_mode in ['final', 'all']:
                optimized_c2w_list = [] # convert pypose Sim3 objs to 4x4 matrices

                for i in range(len(optimized_abs_poses)):
                    # get 4x4 matrix from pypose Sim3 obj
                    pose_matrix = optimized_abs_poses[i].matrix().cpu().numpy()
                    optimized_c2w_list.append(pose_matrix)
                self._log_loop_closure_to_rerun(optimized_c2w_list, self.loop_sim3_list)

            if self.timer:
                self.timer.end()  # End "Loop Closure Optimization"

        if self.timer:
            self.timer.start("PLY Generation")

        print('Apply alignment')
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)
        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(f'Applying {chunk_idx + 1} -> {chunk_idx} (Total {len(self.chunk_indices) - 1})')
            s, R, t = self.sim3_list[chunk_idx]


            chunk_data = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx + 1}.npy"),
                                     allow_pickle=True).item()

            chunk_data['world_points'] = apply_sim3_direct(chunk_data['world_points'], s, R, t)


            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx + 1}.npy")
            np.save(aligned_path, chunk_data)

            if chunk_idx == 0:

                chunk_data_first = np.load(os.path.join(self.result_unaligned_dir, f"chunk_0.npy"),
                                               allow_pickle=True).item()

                np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)

                points_first = chunk_data_first['world_points'].reshape(-1, 3)
                colors_first = (chunk_data_first['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
                confs_first = chunk_data_first['world_points_conf'].reshape(-1)
                ply_path_first = os.path.join(self.pcd_dir, f'0_pcd.ply')
                if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
                    conf_threshold = np.mean(confs_first) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef'] 
                else:
                    conf_threshold = -1.0
                save_confident_pointcloud_batch(
                    points=points_first,  # shape: (H, W, 3)
                    colors=colors_first,  # shape: (H, W, 3)
                    confs=confs_first,  # shape: (H, W)
                    output_path=ply_path_first,
                    conf_threshold=conf_threshold,
                    sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio']
                )


            aligned_chunk_data = np.load(os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy"),
                                             allow_pickle=True).item() if chunk_idx > 0 else chunk_data_first

            points = aligned_chunk_data['world_points'].reshape(-1, 3)
            colors = (aligned_chunk_data['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
            confs = aligned_chunk_data['world_points_conf'].reshape(-1)
            ply_path = os.path.join(self.pcd_dir, f'{chunk_idx + 1}_pcd.ply')
            if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
                conf_threshold = np.mean(confs) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef']
            else:
                conf_threshold = -1.0
            save_confident_pointcloud_batch(
                points=points,  # shape: (H, W, 3)
                colors=colors,  # shape: (H, W, 3)
                confs=confs,  # shape: (H, W)
                output_path=ply_path,
                conf_threshold=conf_threshold,
                sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio']
            )

        self.save_camera_poses()

        if self.timer:
            self.timer.end()  # End "PLY Generation"
        
        print('Done.')

    def run(self, subsample_freq=1):
        print(f"Loading images from {self.img_dir}...")
        all_images = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")) +
                               glob.glob(os.path.join(self.img_dir, "*.png")))
        # print(self.img_list)
        if len(all_images) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(all_images)} images")

        if subsample_freq > 1:
            self.img_list = all_images[::subsample_freq]
            print(f"Using every {subsample_freq}th image: ({len(self.img_list)} images)")
        else:
            self.img_list = all_images
            print(f"Using all ({len(self.img_list)} images)")

        if self.timer and self.loop_enable:
            self.timer.end()  # End "Initialization"
            self.timer.start("Loop Detection")

        if self.loop_enable:
            self.get_loop_pairs()

            if self.useDBoW:
                self.retrieval.close()  # Save CPU Memory
                gc.collect()
            else:
                del self.loop_detector  # Save GPU Memory
        
        if self.timer:
            if self.loop_enable:
                self.timer.end()  # End "Loop Detection"
            else:
                self.timer.end()  # End "Initialization"
            self.timer.start("Model Loading")

        torch.cuda.empty_cache()
        print('Loading model...')
        self.model.load()

        if self.config['Model']['calib']:
            calib_path = Path(self.img_dir).parent / 'calib.txt'
            k, p2_matrix = extract_p2_k_matrix(calib_path)
            self.model.k = k

        if self.timer:
            self.timer.end()  # End "Model Loading"

        # Smart sample_ratio adjustment if max_points specified
        # if self.max_points is not None:
        #     estimated_total_points = self._estimate_total_points()
        #     if estimated_total_points > self.max_points:
        #         adjustment_factor = self.max_points / estimated_total_points
        #         original_sample_ratio = self.config['Model']['Pointcloud_Save']['sample_ratio']
        #         adjusted_sample_ratio = original_sample_ratio * adjustment_factor

        #         print(f"[Point Limiting] Target: {self.max_points:,} points")
        #         print(f"[Point Limiting] Estimated total: {estimated_total_points:,} points")
        #         print(f"[Point Limiting] Adjusting sample_ratio: {original_sample_ratio:.4f} → {adjusted_sample_ratio:.4f}")

        #         self.config['Model']['Pointcloud_Save']['sample_ratio'] = adjusted_sample_ratio

        self.process_long_sequence()

    def save_camera_poses(self):
        '''
        Save camera poses from all chunks to txt and ply files
        - txt file: Each line contains a 4x4 C2W matrix flattened into 16 numbers
        - ply file: Camera poses visualized as points with different colors for each chunk
        '''
        chunk_colors = [
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [128, 0, 0],  # Dark Red
            [0, 128, 0],  # Dark Green
            [0, 0, 128],  # Dark Blue
            [128, 128, 0],  # Olive
        ]
        print("Saving all camera poses to txt file...")

        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        _, first_chunk_intrinsics = self.all_camera_intrinsics[0]
        for i, idx in enumerate(range(first_chunk_range[0], first_chunk_range[1])):
            c2w = first_chunk_extrinsics[i]
            all_poses[idx] = c2w
            if first_chunk_intrinsics is not None:
                all_intrinsics[idx] = first_chunk_intrinsics[i]

        for chunk_idx in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            _, chunk_intrinsics = self.all_camera_intrinsics[chunk_idx]
            s, R, t = self.sim3_list[
                chunk_idx - 1]  # When call self.save_camera_poses(), all the sim3 are aligned to the first chunk.

            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            for i, idx in enumerate(range(chunk_range[0], chunk_range[1])):
                c2w = chunk_extrinsics[i]  #

                transformed_c2w = S @ c2w  # Be aware of the left multiplication!
                transformed_c2w[:3, :3] /= s  # Normalize rotation

                all_poses[idx] = transformed_c2w
                if chunk_intrinsics is not None:
                    all_intrinsics[idx] = chunk_intrinsics[i]

        poses_path = os.path.join(self.output_dir, 'camera_poses.txt')
        with open(poses_path, 'w') as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(' '.join([str(x) for x in flat_pose]) + '\n')

        print(f"Camera poses saved to {poses_path}")
        if all_intrinsics[0] is not None:
            intrinsics_path = os.path.join(self.output_dir, 'intrinsic.txt')
            with open(intrinsics_path, 'w') as f:
                for intrinsic in all_intrinsics:
                    fx = intrinsic[0, 0]
                    fy = intrinsic[1, 1]
                    cx = intrinsic[0, 2]
                    cy = intrinsic[1, 2]
                    f.write(f'{fx} {fy} {cx} {cy}\n')
            print(f"Camera intrinsics saved to {intrinsics_path}")

        ply_path = os.path.join(self.output_dir, 'camera_poses.ply')
        with open(ply_path, 'w') as f:
            # Write PLY header
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(all_poses)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')

            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                f.write(f'{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n')

        print(f"Camera poses visualization saved to {ply_path}")

    def close(self):
        '''
            Clean up temporary files and calculate reclaimed disk space.
            
            This method deletes all temporary files generated during processing from three directories:
            - Unaligned results
            - Aligned results
            - Loop results
            
            ~50 GiB for 4500-frame KITTI 00, 
            ~35 GiB for 2700-frame KITTI 05, 
            or ~5 GiB for 300-frame short seq.
        '''
        if not self.delete_temp_files:
            return
        
        total_space = 0

        print(f'Deleting the temp files under {self.result_unaligned_dir}')
        for filename in os.listdir(self.result_unaligned_dir):
            file_path = os.path.join(self.result_unaligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f'Deleting the temp files under {self.result_aligned_dir}')
        for filename in os.listdir(self.result_aligned_dir):
            file_path = os.path.join(self.result_aligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f'Deleting the temp files under {self.result_loop_dir}')
        for filename in os.listdir(self.result_loop_dir):
            file_path = os.path.join(self.result_loop_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)
        print('Deleting temp files done.')

        print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")


import shutil
def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)
        
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        
        shutil.copy2(src_path, dst_path)
        print(f"config yaml file has been copied to: {dst_path}")
        return dst_path
        
    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as e:
        print(f"Copy Error: {e}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VGGT-Long')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Image path')
    parser.add_argument('--config', type=str, required=False, default='./configs/base_config.yaml',
                        help='config path')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Custom output directory (overrides timestamp-based dir)')
    parser.add_argument('--max_points', type=int, default=None,
                        help='Maximum points for Rerun visualization (does not affect PLY generation)')
    parser.add_argument('--viz', action='store_true',
                        help='Enable Rerun visualization')
    parser.add_argument('--viz_mode', type=str, default='final',
                        choices=['realtime', 'alignment', 'final', 'all'],
                        help='Visualization mode: realtime (per-chunk), alignment (trajectories), final (complete), all (everything)')
    parser.add_argument('--save_rrd', type=str, default=None,
                        help='Save Rerun recording to file (e.g., output.rrd)')
    parser.add_argument('--subsample_freq', type=int, default=1,
                        help='Use every Nth image (e.g., 10 = every 10th image). Default: 1 (all images)')
    args = parser.parse_args()

    config = load_config(args.config)

    image_dir = args.image_dir

    # Check CLI arg first, then config, then fall back to timestamp-based
    if args.save_dir:
        save_dir = args.save_dir
    elif 'Output' in config and 'save_dir' in config['Output'] and config['Output']['save_dir'] is not None:
        save_dir = config['Output']['save_dir']
    else:
        # Original timestamp-based logic
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        exp_dir = './exps'
        save_dir = os.path.join(exp_dir, image_dir.replace("/", "_"), current_datetime)

    # save_dir = os.path.join(
    #     exp_dir, path[-3] + "_" + path[-2] + "_" + path[-1], current_datetime
    # )

    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
        print(f'The exp will be saved under dir: {save_dir}')
        copy_file(args.config, save_dir)

    # Initialize timing system
    timer = StageTimer(save_dir)
    timer.start("Initialization")

    # Determine max_points from CLI or config
    max_points = args.max_points
    if max_points is None and 'Output' in config and 'max_total_points' in config['Output']:
        max_points = config['Output']['max_total_points']

    enable_viz = args.viz
    viz_mode = args.viz_mode
    save_rrd = args.save_rrd
    if not enable_viz and 'Visualization' in config:
        viz_config = config['Visualization']
        enable_viz = viz_config.get('enable', False)
        if enable_viz:
            viz_mode = viz_config.get('mode', 'final')
            save_rrd = viz_config.get('save_rrd', None)

    if config['Model']['align_method'] == 'numba':
        warmup_numba()

    vggt_long = VGGT_Long(image_dir, save_dir, config, max_points=max_points, timer=timer,
                          enable_viz=enable_viz, viz_mode=viz_mode, save_rrd=save_rrd)
    vggt_long.run(subsample_freq=args.subsample_freq)
    if timer:
        timer.start("Cleanup")
    vggt_long.close()

    del vggt_long
    torch.cuda.empty_cache()
    gc.collect()

    if timer:
        timer.end()  # End "Cleanup"
        timer.start("PLY Merging")

    all_ply_path = os.path.join(save_dir, f'pcd/combined_pcd.ply')
    input_dir = os.path.join(save_dir, f'pcd')
    print("Saving all the point clouds")


    merge_ply_files(input_dir, all_ply_path)
    
    # Visualize final point cloud if enabled
    if enable_viz and RERUN_AVAILABLE and (viz_mode in ['final', 'all']):
        if timer:
            timer.start("Final Visualization")

        # Rerun was already initialized in VGGT_Long.__init__
        # Just need to read and log the final PLY
        from loop_utils.rerun_utils import read_ply_points_colors, downsample_for_viz

        viz_config = config.get('Visualization', {})
        max_viz_points = viz_config.get('max_viz_points', 2_000_000)

        print("[VIZ] Loading final point cloud for visualization...")
        points, colors = read_ply_points_colors(all_ply_path)
        points_viz, colors_viz = downsample_for_viz(points, colors, max_viz_points)

        rr.log(
            "world/final/combined_pointcloud",
            rr.Points3D(points_viz, colors=colors_viz, radii=0.01)
        )

        print(f"[VIZ] Logged final point cloud to Rerun ({len(points_viz):,} points)")

        if timer:
            timer.end()

    if timer:
        timer.save_log()
        
    print('All done.')
    sys.exit()