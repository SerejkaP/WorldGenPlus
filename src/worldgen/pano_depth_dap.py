import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

DAP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "submodules/DAP"))


def _ensure_dap_in_path():
    if DAP_ROOT not in sys.path:
        sys.path.insert(0, DAP_ROOT)


def build_dap_depth_model(weights_path: str, device: torch.device = 'cuda'):
    _ensure_dap_in_path()

    # dinov3_repo_dir in dap.py is a relative path — must run from DAP root
    prev_cwd = os.getcwd()
    os.chdir(DAP_ROOT)
    try:
        from networks.dap import make_model

        model = make_model(
            midas_model_type='vitl',
            fine_tune_type='hypersim',
            min_depth=0.01,
            max_depth=1.0,
            train_decoder=True,
        )

        state = torch.load(weights_path, map_location=device, weights_only=True)

        # Handle DataParallel checkpoints: strip "module." prefix from keys
        if any(k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):]: v for k, v in state.items()}

        model_state = model.state_dict()
        model.load_state_dict(
            {k: v for k, v in state.items() if k in model_state},
            strict=False,
        )
    finally:
        os.chdir(prev_cwd)

    model.eval()
    model = model.to(device)
    return model


def pred_pano_depth_dap(
    model,
    image: Image.Image,
    depth_scale: float = 10.0,
    num_passes: int = 2,
    blend_width: int = 128,
):
    """
    Run DAP inference on an equirectangular panorama.

    DAP processes the panorama as a flat image, so depth near the left/right
    boundary (where the edges of the panorama meet) is unreliable — a "seam".
    To suppress it, the image is processed num_passes times, each shifted by
    W / num_passes pixels.  Each pass has its seam at a different location,
    so passes can be blended to cover each other's weak zones.

    num_passes controls the trade-off:
      1 — single inference, fastest, seam visible
      2 — two passes (shifts: 0, W/2), seams at 0 and W/2 cover each other
      4 — four passes (shifts: 0, W/4, W/2, 3W/4), finer coverage

    For num_passes=k, shifts are equally spaced at [0, W/k, 2W/k, ..., (k-1)W/k].
    Using fewer passes than k for a shift of W/k leaves uncovered seam positions.

    Returns a dict with the same keys as pred_pano_depth (UniK3D):
      rgb      : (H, W, 3) torch.Tensor uint8
      depth    : (H, W) torch.Tensor float32  — Z-depth (meters)
      distance : (H, W) torch.Tensor float32  — radial distance (meters)
      rays     : (H, W, 3) torch.Tensor float32 — unit direction vectors
    """
    import cv2

    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    blend_width = min(blend_width, W // (2 * max(num_passes, 1)))

    # --- Multi-pass inference --------------------------------------------------
    depth_passes = []
    shifts = [round(W * i / num_passes) for i in range(num_passes)]

    for shift in shifts:
        img_shifted = np.roll(img_bgr, shift, axis=1)
        depth = model.infer_image(img_shifted, input_size=518).astype(np.float32)
        depth_passes.append(np.roll(depth, -shift, axis=1))  # unshift back

    # --- Scale-match all passes to pass 0 ------------------------------------
    # DAP is affine-invariant, so individual passes may differ in global scale.
    # Use the center band of pass 0 (reliable for shift=0) as the reference.
    half_band = max(blend_width, W // 8)
    cx = W // 2
    ref_mean = depth_passes[0][:, cx - half_band: cx + half_band].mean()
    for i in range(1, num_passes):
        src_mean = depth_passes[i][:, cx - half_band: cx + half_band].mean()
        if src_mean > 1e-6:
            depth_passes[i] *= ref_mean / src_mean

    # --- Cosine blend weights -------------------------------------------------
    # Each pass i has its seam at x = shifts[i].
    # Weight of pass i at column x: proportional to circular distance from seam,
    # with cosine rolloff over blend_width. Weights are then normalised to sum=1.
    x = np.arange(W, dtype=np.float32)
    weights = np.zeros((num_passes, W), dtype=np.float32)
    for i, shift in enumerate(shifts):
        # Circular distance from seam column (shift % W)
        dist = np.minimum(
            np.abs(x - shift) % W,
            W - np.abs(x - shift) % W,
        )
        alpha = np.clip(dist / blend_width, 0.0, 1.0)
        weights[i] = (1 - np.cos(alpha * np.pi)) / 2   # 0 at seam → 1 away

    weight_sum = weights.sum(axis=0, keepdims=True).clip(min=1e-6)
    weights /= weight_sum  # (num_passes, W), each column sums to 1

    depth_raw = sum(
        depth_passes[i] * weights[i][None, :] for i in range(num_passes)
    )
    depth_meters = depth_raw * depth_scale

    # --- Ray directions -------------------------------------------------------
    # Must match UniK3D Spherical.unproject convention so that convert_rgbd_to_gs
    # (built around UniK3D output) places gaussians correctly.
    # UniK3D convention: latitude increases downward in image space
    #   (top, v=0 → lat=-π/2 → y=-1;  bottom, v=H-1 → lat=+π/2 → y=+1)
    # convert_rgbd_to_gs treats y=+1 as world-up, so this is internally consistent
    # throughout the pipeline even though it differs from the standard Y-up panorama convention.
    device = next(model.parameters()).device

    i_coords = torch.arange(H, dtype=torch.float32, device=device)
    j_coords = torch.arange(W, dtype=torch.float32, device=device)

    lat = ((i_coords + 0.5) / H - 0.5) * torch.pi        # -π/2 (top) → +π/2 (bottom)
    lon = ((j_coords + 0.5) / W - 0.5) * 2 * torch.pi    # -π   (left) → +π  (right)

    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing='ij')  # (H, W)

    rx = torch.cos(lat_grid) * torch.sin(lon_grid)
    ry = torch.sin(lat_grid)
    rz = torch.cos(lat_grid) * torch.cos(lon_grid)
    rays = torch.stack([rx, ry, rz], dim=-1)  # (H, W, 3)

    rgb = torch.from_numpy(img_rgb).to(device)                    # (H, W, 3) uint8
    distance = torch.from_numpy(depth_meters).to(device).float()  # (H, W)
    depth = distance * rz                                          # Z-depth

    return {
        "rgb": rgb,
        "depth": depth,
        "distance": distance,
        "rays": rays,
    }
