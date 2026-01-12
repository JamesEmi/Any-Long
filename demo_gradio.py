import os
import sys
import glob
import time
import gc
import shutil
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import gradio as gr
import gradio_client.utils as grc_utils
import numpy as np
import torch

from loop_utils.config_utils import load_config
from loop_utils.sim3utils import apply_sim3_direct
from vggt_long import VGGT_Long

# Make sure local VGGT package is discoverable
sys.path.append("vggt/")

from loop_utils.visual_util import predictions_to_glb  # noqa: E402

# -------------------------------------------------------------------------
# Patch gradio_client json schema helper to tolerate boolean schemas
# (gradio_client 1.5.x can emit `additionalProperties: False`, which
# triggers a TypeError in json_schema_to_python_type)
# -------------------------------------------------------------------------
_orig_get_type = grc_utils.get_type


def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "boolean" if schema else "false"
    return _orig_get_type(schema)


grc_utils.get_type = _safe_get_type


# -------------------------------------------------------------------------
# Runtime / config
# -------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "map_long_darpa.yaml")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# -------------------------------------------------------------------------
# 1) VGGT-Long core pipeline
# -------------------------------------------------------------------------
def build_long_predictions(vlong: VGGT_Long) -> dict:
    """
    Merge aligned chunk outputs into a single predictions dict
    compatible with loop_utils.visual_util.predictions_to_glb.
    """
    all_wp: List[np.ndarray] = []
    all_wpc: List[np.ndarray] = []
    all_images: List[np.ndarray] = []
    all_extrinsic: List[np.ndarray] = []
    all_intrinsic: List[np.ndarray] = []
    all_depth: List[np.ndarray] = []
    all_depth_conf: List[np.ndarray] = []

    sim3_list = vlong.sim3_list
    chunk_indices = vlong.chunk_indices

    for idx, _range in enumerate(chunk_indices):
        npy_path = os.path.join(vlong.result_aligned_dir, f"chunk_{idx}.npy")
        if not os.path.exists(npy_path):
            npy_path = os.path.join(vlong.result_unaligned_dir, f"chunk_{idx}.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Aligned/unaligned chunk file missing: {npy_path}")

        data = np.load(npy_path, allow_pickle=True).item()
        wp = data["world_points"]
        wpc = data.get("world_points_conf", np.ones(wp.shape[:-1], dtype=np.float32))
        imgs = data["images"]
        if imgs.ndim == 4 and imgs.shape[1] == 3:
            imgs = np.transpose(imgs, (0, 2, 3, 1))
        extr = data["extrinsic"]
        intr = data["intrinsic"]
        depth = data.get("depth")
        depth_conf = data.get("depth_conf")

        if idx > 0 and len(sim3_list) >= idx:
            s, R, t = sim3_list[idx - 1]
            wp = apply_sim3_direct(wp, s, R, t)

            aligned_extr = []
            for e in extr:
                w2c = np.eye(4)
                w2c[:3, :] = e
                c2w = np.linalg.inv(w2c)
                S = np.eye(4)
                S[:3, :3] = s * R
                S[:3, 3] = t
                c2w_aligned = S @ c2w
                w2c_aligned = np.linalg.inv(c2w_aligned)[:3, :]
                aligned_extr.append(w2c_aligned)
            extr = np.stack(aligned_extr, axis=0)

        all_wp.append(wp)
        all_wpc.append(wpc)
        all_images.append(imgs)
        all_extrinsic.append(extr)
        all_intrinsic.append(intr)
        if depth is not None:
            all_depth.append(depth)
        if depth_conf is not None:
            all_depth_conf.append(depth_conf)

    predictions = {
        "world_points": np.concatenate(all_wp, axis=0),
        "world_points_conf": np.concatenate(all_wpc, axis=0),
        "images": np.concatenate(all_images, axis=0),
        "extrinsic": np.concatenate(all_extrinsic, axis=0),
        "intrinsic": np.concatenate(all_intrinsic, axis=0),
    }
    if all_depth:
        predictions["depth"] = np.concatenate(all_depth, axis=0)
    if all_depth_conf:
        predictions["depth_conf"] = np.concatenate(all_depth_conf, axis=0)
    return predictions


def run_model_long(target_dir: str, config_path: str = DEFAULT_CONFIG) -> Tuple[dict, str]:
    """
    Execute VGGT-Long on images located at target_dir/images.
    Returns (predictions_dict, save_dir).
    """
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for VGGT-Long.")

    config = load_config(config_path)
    config["Model"]["delete_temp_files"] = False

    image_dir = os.path.join(target_dir, "images")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_dir = os.path.join(target_dir, f"vggt_long_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    vlong = VGGT_Long(image_dir=image_dir, save_dir=save_dir, config=config)
    vlong.run()

    predictions = build_long_predictions(vlong)
    return predictions, save_dir


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images, prev_dir: Optional[str] = None) -> Tuple[str, List[str]]:
    """
    Create a new 'target_dir/images' subfolder, place user uploads, and clear previous temp dir.
    Videos are sampled at ~5 FPS.
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    if prev_dir and os.path.isdir(prev_dir):
        shutil.rmtree(prev_dir, ignore_errors=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = os.path.join(RESULTS_DIR, f"input_images_{timestamp}")
    target_dir_images = os.path.join(target_dir, "images")
    os.makedirs(target_dir_images, exist_ok=True)

    image_paths: List[str] = []

    if input_images is not None:
        for file_data in input_images:
            file_path = file_data["name"] if isinstance(file_data, dict) else file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    if input_video is not None:
        video_path = input_video["name"] if isinstance(input_video, dict) else input_video
        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = max(int(round(fps / 5.0)), 1) if fps and fps > 0 else 1

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1
            count += 1

    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_video, input_images, current_dir):
    """
    Whenever user uploads or changes files, clear previous temp dir, re-import files, and show gallery.
    """
    if not input_video and not input_images:
        return None, None, None, "Please upload data first."
    target_dir, image_paths = handle_uploads(input_video, input_images, prev_dir=current_dir)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


# -------------------------------------------------------------------------
# 4) Reconstruction: uses the target_dir plus any viz parameters
# -------------------------------------------------------------------------
def gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Predicted Pointmap",
):
    """Perform reconstruction using the already-created target_dir/images."""
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    if len(all_files) == 0:
        return (
            None,
            f"No images found in {target_dir_images}. Please upload again.",
            gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame"),
        )
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running VGGT-Long...")
    with torch.no_grad():
        predictions, save_dir = run_model_long(target_dir, config_path=DEFAULT_CONFIG)

    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    if frame_filter is None:
        frame_filter = "All"

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)

    # Clean up raw upload frames to keep final result folder lean
    shutil.rmtree(target_dir_images, ignore_errors=True)

    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization. Total time: {end_time - start_time:.2f}s"
    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def clear_fields():
    """Clears the 3D viewer, the stored target_dir, and empties the gallery."""
    return None


def update_log():
    """Display a quick log message while waiting."""
    return "Loading and Reconstructing..."


def update_visualization(
    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer.
    """
    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    loaded = np.load(predictions_path, allow_pickle=True)
    predictions = {key: loaded[key] for key in loaded.keys()}

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"


# -------------------------------------------------------------------------
# 6) Build Gradio UI (VGGT-Long)
# -------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    theme = gr.themes.Ocean()
    theme.set(
        checkbox_label_background_fill_selected="*button_primary_background_fill",
        checkbox_label_text_color_selected="*button_primary_text_color",
    )

    with gr.Blocks(
        theme=theme,
        css="""
        .custom-log * {
            font-style: italic;
            font-size: 22px !important;
            background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
            -webkit-background-clip: text;
            background-clip: text;
            font-weight: bold !important;
            color: transparent !important;
            text-align: center !important;
        }

        .example-log * {
            font-style: italic;
            font-size: 16px !important;
            background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent !important;
        }

        #my_radio .wrap {
            display: flex;
            flex-wrap: nowrap;
            justify-content: center;
            align-items: center;
        }

        #my_radio .wrap label {
            display: flex;
            width: 50%;
            justify-content: center;
            align-items: center;
            margin: 0;
            padding: 10px 0;
            box-sizing: border-box;
        }

        .upload-images-box {
            max-height: 220px;
            overflow-y: auto;
        }
        """,
    ) as demo:
        gr.HTML(
            """
            <h1>🏛️ VGGT-Long: Chunk it, Loop it, Align it, Pushing VGGT's Limits on Kilometer-scale Long RGB Sequences</h1>
            <p><a href="https://github.com/DengKaiCQ/VGGT-Long">🐙 GitHub Repository</a></p>

            <div style="font-size: 16px; line-height: 1.5;">
            <p>Upload a video (sampled at 5 FPS) or a set of images to create a 3D reconstruction. VGGT-Long splits long sequences into chunks, aligns them, and renders a colored point cloud plus camera frustums.</p>

            <h3>Getting Started:</h3>
            <ol>
                <li><strong>Upload Your Data:</strong> Use the "Upload Video" or "Upload Images" buttons on the left. Videos are sampled at ~5 FPS.</li>
                <li><strong>Preview:</strong> Your uploaded images will appear in the gallery on the left.</li>
                <li><strong>Reconstruct:</strong> Click the "Reconstruct" button to start the 3D reconstruction process.</li>
                <li><strong>Visualize:</strong> The 3D reconstruction will appear in the viewer on the right. You can rotate, pan, and zoom to explore the model, and download the GLB file. Note the visualization of 3D points may be slow for a large number of input images.</li>
                <li>
                <strong>Adjust Visualization (Optional):</strong>
                After reconstruction, you can fine-tune the visualization using the options below
                <details style="display:inline;">
                    <summary style="display:inline;">(<strong>click to expand</strong>):</summary>
                    <ul>
                    <li><em>Confidence Threshold:</em> Adjust the filtering of points based on confidence.</li>
                    <li><em>Show Points from Frame:</em> Select specific frames to display in the point cloud.</li>
                    <li><em>Show Camera:</em> Toggle the display of estimated camera positions.</li>
                    <li><em>Filter Sky / Filter Black Background:</em> Remove sky or black-background points.</li>
                    <li><em>Select a Prediction Mode:</em> Choose between "Predicted Pointmap" or "Depthmap and Camera Branch."</li>
                    </ul>
                </details>
                </li>
                <li><strong>Clear Before Next Upload:</strong> Make sure to clear the previous upload records before the next images/video upload (you can click the cross icon in the upper-right corner of the upload box to do so).</li>
            </ol>
            <p><strong style="color: #0ea5e9;">Note:</strong> Visualization of dense point clouds may take time. If localhost is blocked, set <code>GRADIO_SHARE=1</code> before launching.</p>
            </div>
            """
        )

        target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

        with gr.Row():
            with gr.Column(scale=2):
                input_video = gr.Video(label="Upload Video", interactive=True)
                input_images = gr.File(
                    file_count="multiple",
                    label="Upload Images",
                    interactive=True,
                    height=160,
                    elem_classes=["upload-images-box"],
                )

                image_gallery = gr.Gallery(
                    label="Preview",
                    columns=4,
                    height="300px",
                    show_download_button=True,
                    object_fit="contain",
                    preview=True,
                )

            with gr.Column(scale=4):
                with gr.Column():
                    gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
                    log_output = gr.Markdown(
                        "Please upload a video or images, then click Reconstruct.",
                        elem_classes=["custom-log"],
                    )
                    reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

                with gr.Row():
                    submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                    clear_btn = gr.ClearButton(
                        [input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery],
                        scale=1,
                    )

                with gr.Row():
                    prediction_mode = gr.Radio(
                        ["Predicted Pointmap", "Depthmap and Camera Branch"],
                        label="Select a Prediction Mode",
                        value="Predicted Pointmap",
                        scale=1,
                        elem_id="my_radio",
                    )

                with gr.Row():
                    conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                    frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                    with gr.Column():
                        show_cam = gr.Checkbox(label="Show Camera", value=True)
                        mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                        mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                        mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

        submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
            fn=update_log, inputs=[], outputs=[log_output]
        ).then(
            fn=gradio_demo,
            inputs=[target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode],
            outputs=[reconstruction_output, log_output, frame_filter],
        )

        conf_thres.change(
            update_visualization,
            [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode],
            [reconstruction_output, log_output],
        )
        frame_filter.change(
            update_visualization,
            [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode],
            [reconstruction_output, log_output],
        )
        mask_black_bg.change(
            update_visualization,
            [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode],
            [reconstruction_output, log_output],
        )
        mask_white_bg.change(
            update_visualization,
            [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode],
            [reconstruction_output, log_output],
        )
        show_cam.change(
            update_visualization,
            [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode],
            [reconstruction_output, log_output],
        )
        mask_sky.change(
            update_visualization,
            [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode],
            [reconstruction_output, log_output],
        )
        prediction_mode.change(
            update_visualization,
            [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode],
            [reconstruction_output, log_output],
        )

        input_video.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images, target_dir_output],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )
        input_images.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images, target_dir_output],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )

    return demo


demo = build_demo()


def launch(server_name: str = None, server_port: int = None, share: bool = None) -> None:
    """
    Launch the Gradio app. Environment variable overrides:
      GRADIO_SERVER_NAME (default: 0.0.0.0)
      GRADIO_SERVER_PORT (default: 8080)
      GRADIO_SHARE       (default: True if unset; set 0/false to disable)
    """
    if server_name is None:
        server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    if server_port is None:
        server_port = int(os.environ.get("GRADIO_SERVER_PORT", "8080"))
    if share is None:
        # Default to share=True to avoid "localhost not accessible" errors in restricted envs.
        share_env = os.environ.get("GRADIO_SHARE", "1")
        share = share_env in ("1", "true", "True")

    demo.queue(max_size=20).launch(
        show_error=True,
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_api=False,  # avoid schema generation issues in some gradio_client versions
    )


if __name__ == "__main__":
    launch()

