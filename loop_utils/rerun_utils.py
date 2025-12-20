"""
Rerun visualization utilities for VGGT-Long pipeline.

This module provides helper functions for visualizing the VGGT-Long reconstruction
process using Rerun, including camera frustums, trajectories, and point clouds.
"""
import numpy as np
import rerun as rr
import struct


def get_chunk_color(chunk_idx):
    """
    Get unique color for chunk identification (cameras/trajectories only).

    Args:
        chunk_idx: Chunk index (0-based)

    Returns:
        RGB color as numpy array (3,) with dtype uint8
    """
    colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [255, 128, 0],  # Orange
        [128, 0, 255],  # Purple
        [0, 255, 128],  # Spring Green
        [128, 255, 0],  # Chartreuse
        [255, 0, 128],  # Rose
        [0, 128, 255],  # Azure
    ]
    return np.array(colors[chunk_idx % len(colors)], dtype=np.uint8)


def log_camera_frustum(entity_path, pose, intrinsics, color, width, height, scale=0.3):
    """
    Log camera frustum for visualization.

    Args:
        entity_path: Rerun entity path (e.g., "world/cameras/chunk_0/frame_5")
        pose: 4x4 C2W camera-to-world matrix
        intrinsics: 3x3 camera intrinsic matrix K
        color: RGB color [3,] for frustum lines (uint8)
        width: Image width in pixels
        height: Image height in pixels
        scale: Frustum depth in meters (default: 0.3)
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Frustum corners in camera space
    corners_cam = np.array([
        [-cx/fx * scale, -cy/fy * scale, scale],                    # Top-left
        [(width-cx)/fx * scale, -cy/fy * scale, scale],             # Top-right
        [(width-cx)/fx * scale, (height-cy)/fy * scale, scale],     # Bottom-right
        [-cx/fx * scale, (height-cy)/fy * scale, scale],            # Bottom-left
        [0, 0, 0]  # Camera center
    ], dtype=np.float32)

    # Transform to world space
    corners_world = (pose[:3, :3] @ corners_cam.T).T + pose[:3, 3]

    # Define frustum edges
    edges = [
        corners_world[[4, 0]],  # Center to top-left
        corners_world[[4, 1]],  # Center to top-right
        corners_world[[4, 2]],  # Center to bottom-right
        corners_world[[4, 3]],  # Center to bottom-left
        corners_world[[0, 1]],  # Top edge
        corners_world[[1, 2]],  # Right edge
        corners_world[[2, 3]],  # Bottom edge
        corners_world[[3, 0]],  # Left edge
    ]

    # Log all edges as LineStrips3D
    rr.log(
        entity_path,
        rr.LineStrips3D(strips=edges, colors=[color] * len(edges))
    )


def log_camera_axes(entity_path, pose, scale=0.5):
    """
    Log camera coordinate axes as colored arrows (X=red, Y=green, Z=blue).

    Args:
        entity_path: Rerun entity path
        pose: 4x4 C2W camera-to-world matrix
        scale: Arrow length in meters (default: 0.5)
    """
    R = pose[:3, :3]
    t = pose[:3, 3]

    # X (red), Y (green), Z (blue) axes
    xw = R[:, 0] * scale
    yw = R[:, 1] * scale
    zw = R[:, 2] * scale

    origins = np.stack([t, t, t], axis=0)
    vectors = np.stack([xw, yw, zw], axis=0)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

    rr.log(entity_path, rr.Arrows3D(origins=origins, vectors=vectors, colors=colors))


def downsample_for_viz(points, colors, max_points=2_000_000):
    """
    Downsample point cloud for visualization using random sampling.

    Args:
        points: (N, 3) point positions
        colors: (N, 3) point colors (uint8)
        max_points: Maximum number of points to keep

    Returns:
        downsampled_points: (M, 3) where M <= max_points
        downsampled_colors: (M, 3)
    """
    if points.shape[0] <= max_points:
        return points, colors

    indices = np.random.choice(points.shape[0], size=max_points, replace=False)
    return points[indices], colors[indices]


def read_ply_points_colors(ply_path):
    """
    Read points and colors from binary PLY file.

    Assumes PLY format:
    - Binary little-endian
    - Each vertex: 3 floats (x,y,z) + 3 bytes (r,g,b) = 15 bytes

    Args:
        ply_path: Path to PLY file

    Returns:
        points: (N, 3) float32 array
        colors: (N, 3) uint8 array
    """
    # Read header to find data offset and vertex count
    with open(ply_path, 'rb') as f:
        header_size = 0
        num_vertices = 0

        for line in f:
            header_size += len(line)
            line_str = line.decode('utf-8').strip()

            if line_str.startswith('element vertex'):
                num_vertices = int(line_str.split()[-1])

            if line_str == 'end_header':
                break

    if num_vertices == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    # Read binary data
    # Each point: 3 floats (x,y,z) + 3 bytes (r,g,b) = 15 bytes
    point_size = 15

    points = np.zeros((num_vertices, 3), dtype=np.float32)
    colors = np.zeros((num_vertices, 3), dtype=np.uint8)

    with open(ply_path, 'rb') as f:
        f.seek(header_size)

        for i in range(num_vertices):
            data = f.read(point_size)
            if len(data) < point_size:
                # Truncate if file is incomplete
                points = points[:i]
                colors = colors[:i]
                break

            x, y, z = struct.unpack('<fff', data[:12])
            r, g, b = struct.unpack('BBB', data[12:15])

            points[i] = [x, y, z]
            colors[i] = [r, g, b]

    return points, colors
