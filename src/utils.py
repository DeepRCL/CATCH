import os
import uuid

import numpy as np
import torch
from PIL import Image

import torch
from monai.metrics import compute_hausdorff_distance, compute_dice, compute_iou
from monai.losses import DiceLoss

import wandb
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class AverageMeter:
    def __init__(self, name, fmt=":6.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def _token_mask_to_grid(token_ids, duration_tokens, grid_size):
    mask = torch.zeros((duration_tokens, grid_size, grid_size), dtype=torch.bool)
    if token_ids.numel() == 0:
        return mask
    flat = token_ids.to(torch.long).view(-1)
    spatial_area = grid_size * grid_size
    t = torch.div(flat, spatial_area, rounding_mode="floor")
    rem = flat % spatial_area
    y = torch.div(rem, grid_size, rounding_mode="floor")
    x = rem % grid_size
    valid = (t >= 0) & (t < duration_tokens)
    t = t[valid]
    y = y[valid]
    x = x[valid]
    mask[t, y, x] = True
    return mask


def _overlay_mask(image_chw, mask_hw, color_rgb, alpha=0.45):
    image = image_chw.permute(1, 2, 0).numpy().astype(np.float32)
    out = image.copy()
    mask = mask_hw.numpy()
    color = np.array(color_rgb, dtype=np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color
    return out.clip(0, 255).astype(np.uint8)


def save_mask_debug_panel(udata, masks_enc, masks_pred, modality, epoch, step, args):
    try:
        debug_dir = "./debug"
        debug_max_frames = 3
        os.makedirs(debug_dir, exist_ok=True)

        clip = udata[0][0][0].detach().cpu()
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=clip.dtype).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=clip.dtype).view(3, 1, 1, 1)
        clip_vis = (clip * (std * 255.0) + (mean * 255.0)).clamp(0, 255).to(torch.uint8)

        _, num_frames, _, _ = clip_vis.shape
        n_frames = min(debug_max_frames, num_frames)
        if modality == "image":
            n_frames = min(1, n_frames)

        grid_size = args.crop_size // args.patch_size
        duration_tokens = max(1, num_frames // args.tubelet_size)

        for mask_id, (enc_ids_bk, pred_ids_bk) in enumerate(zip(masks_enc, masks_pred)):
            enc_ids = enc_ids_bk[0].detach().cpu()
            pred_ids = pred_ids_bk[0].detach().cpu()
            enc_grid = _token_mask_to_grid(enc_ids, duration_tokens, grid_size)
            pred_grid = _token_mask_to_grid(pred_ids, duration_tokens, grid_size)

            rows = []
            for t in range(n_frames):
                token_t = min(t // args.tubelet_size, duration_tokens - 1)
                enc_hw = (
                    enc_grid[token_t]
                    .repeat_interleave(args.patch_size, dim=0)
                    .repeat_interleave(args.patch_size, dim=1)
                )
                pred_hw = (
                    pred_grid[token_t]
                    .repeat_interleave(args.patch_size, dim=0)
                    .repeat_interleave(args.patch_size, dim=1)
                )
                frame = clip_vis[:, t, :, :]
                enc_overlay = _overlay_mask(frame, enc_hw, color_rgb=(255, 0, 0))
                pred_overlay = _overlay_mask(frame, pred_hw, color_rgb=(0, 255, 0))
                raw = frame.permute(1, 2, 0).numpy()
                rows.append(np.concatenate([raw, enc_overlay, pred_overlay], axis=1))

            panel = np.concatenate(rows, axis=0)
            name = f"mask_debug_e{epoch + 1:03d}_i{step:05d}_{modality}_m{mask_id}_{uuid.uuid4().hex[:8]}.png"
            save_path = os.path.join(debug_dir, name)
            Image.fromarray(panel).save(save_path)
            print(f"[debug] saved mask panel: {save_path}")
    except Exception as exc:
        print(f"[debug] failed to save mask panel: {exc}")


def save_args_txt(args):
    args_txt_path = os.path.join(args.output_dir, "args.txt")
    with open(args_txt_path, "w", encoding="utf-8") as wf:
        for key, value in sorted(vars(args).items()):
            wf.write(f"{key}: {value}\n")


def log_gpu_usage(output_dir):
    used_gb, reserved_gb, total_gb = 0, 0, 0
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        used_gb = torch.cuda.memory_allocated(idx) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(idx) / (1024**3)
        total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
    with open(os.path.join(output_dir, "gpu_log.txt"), "a", encoding="utf-8") as wf:
        wf.write(
            f"GPU Memory: allocated={used_gb:.2f} GB, "
            f"reserved={reserved_gb:.2f} GB, "
            f"total={total_gb:.2f} GB "
            f"({used_gb / max(total_gb, 1e-6) * 100:.1f}%)\n"
        )


def calc_batch_seg_metrics(pred_labels, gt, spacing):
    """
    pred_labels: (bs, C, H, W) integer labels
    gt:  (bs, C, H, W)  ground truth
    """
    num_classes = gt.shape[1]

    # logger.info(f"input shapes :{pred_labels.shape}, {gt.shape}")
    # DICE
    dice_scores_per_class = compute_dice(
        y_pred=pred_labels,
        y=gt,
        # reduction="None",
        include_background=False,
        num_classes=num_classes,
    )

    # IOU (mean IoU)
    iou_per_class = compute_iou(
        y_pred=pred_labels,
        y=gt,
        # reduction="None",
        include_background=False,
    )

    gt_present = gt.sum(dim=(2, 3)) > 0   # shape (B, C)
    gt_present = gt_present[:, 1:]  
    
    absent_classes = ~gt_present  # shape (B,num_classes)
    dice_scores_per_class[absent_classes] = float('nan')
    iou_per_class[absent_classes] = float('nan')
    
    
    # spacing = torch.as_tensor(spacing)
    # spacing = spacing.tolist()
    # print(spacing)
    # if spacing.ndim == 1:
    #     spacing = spacing.unsqueeze(0)
    if (spacing == 0).any():
        hd_95_score_per_class = torch.full_like(
            dice_scores_per_class,
            float('nan')
        )
        
    else:  
        spacing = spacing.detach().cpu().squeeze()
        spacing = spacing.tolist()
        hd_95_score_per_class = compute_hausdorff_distance(
        y_pred=pred_labels,
        y=gt,
        include_background=False,
        distance_metric="euclidean",
        percentile=95,
        spacing=spacing,
        
        )
        hd_95_score_per_class[absent_classes] = float('nan')

  
    return dice_scores_per_class, iou_per_class, hd_95_score_per_class



dice_loss_fn = DiceLoss(
    include_background=True,
    to_onehot_y=False,
    softmax=True,
    reduction="none",
)


def compute_seg_loss(seg_logits, seg_gt):
    """
    seg_logits : (B, C=4, H, W)
    seg_gt     : (B, C=3, H, W)  one-hot
    """
    print(f"seg_logits: {seg_logits.shape}, seg_gt: {seg_gt.shape}")
    # build has_fg and gt_indices correctly
    has_fg = seg_gt.sum(dim=1) > 0  # (B, H, W)
    gt_indices = torch.argmax(seg_gt, dim=1) + 1  # (B, H, W) → indices 1,2,3
    gt_indices[~has_fg] = 0  # bg → 0

    # build 4-ch GT for Dice
    bg_ch = (~has_fg).unsqueeze(1).float()  # (B, 1, H, W)
    seg_gt_4ch = torch.cat([bg_ch, seg_gt], dim=1)  # (B, 4, H, W)

    # Dice Loss: per sample, per class
    label_present = (seg_gt_4ch.sum(dim=(2, 3)) > 0).float()  # (B, C=4) ← FIXED COMMENT
    dice_per_class = dice_loss_fn(seg_logits, seg_gt_4ch)  # (B, C=4)
    dice_per_class = dice_per_class.squeeze(-1).squeeze(-1)  # (B, C, 1, 1) → (B, C)
    # Mask out background class (index 0) for averaging
    dice_masked = dice_per_class[:, 1:] * label_present[:, 1:]  # (B, C=3)
    # Average over present foreground classes per sample, then over batch
    num_present = torch.clamp(label_present[:, 1:].sum(dim=1, keepdim=True), min=1)
    dice_loss = (dice_masked.sum(dim=1) / num_present.squeeze(1)).mean()

    return dice_loss


def log_overlay_masks(image, dense_pred, dense_gt_label, view_name):
    """
    Plot side-by-side overlays:
        Left: image + GT mask
        Right: image + Pred mask
    Mask alpha = 0.5 where mask > 0, else 0.
    """

    target_size = (128, 128)  # (width, height) for cv2

    if dense_gt_label.shape != target_size[::-1]:
        dense_gt_label = cv2.resize(
            dense_gt_label, target_size, interpolation=cv2.INTER_NEAREST
        )

    if dense_pred.shape != target_size[::-1]:
        dense_pred = cv2.resize(
            dense_pred, target_size, interpolation=cv2.INTER_NEAREST
        )
  
    if image.shape[:2] != target_size[::-1]:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
    alpha_dense_gt = 0.5 * (dense_gt_label > 0)
    alpha_dense_pred = 0.5 * (dense_pred > 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=100)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)

    # ===== LEFT: IMAGE + Dense_GT  =====
    axes[0].imshow(image)
    axes[0].imshow(dense_gt_label, cmap="viridis", vmin=0, vmax=4, alpha=alpha_dense_gt)
    axes[0].set_title(f"Image + Dense GT {view_name}")
    axes[0].axis("off")

    # ===== Middle: IMAGE + Dense_PRED =====
    axes[1].imshow(image)
    axes[1].imshow(dense_pred, cmap="viridis", vmin=0, vmax=4, alpha=alpha_dense_pred)
    axes[1].set_title(f"Image + Dense Pred {view_name}")
    axes[1].axis("off")

    wandb_img = wandb.Image(fig)
    plt.close(fig)
    return wandb_img



# def save_overlay_masks(image, dense_pred, dense_gt_label, save_path):
#     """
#     Plot side-by-side overlays:
#         Left: image + GT mask
#         Right: image + Pred mask
#     Mask alpha = 0.5 where mask > 0, else 0.
#     """
#     # Fixed colors for classes 0..3
#     colors = [
#         "black",      # background
#         "blue",
#         "green",
#         "yellow",
#     ]
#     cmap = mcolors.ListedColormap(colors)
#     norm = mcolors.BoundaryNorm(np.arange(-0.5, 4, 1), cmap.N)


#     target_size = (256, 256)  # (width, height) for cv2

#     if dense_gt_label.shape != target_size[::-1]:
#         dense_gt_label = cv2.resize(
#             dense_gt_label, target_size, interpolation=cv2.INTER_NEAREST
#         )

#     if dense_pred.shape != target_size[::-1]:
#         dense_pred = cv2.resize(
#             dense_pred, target_size, interpolation=cv2.INTER_NEAREST
#         )
  
#     if image.shape[:2] != target_size[::-1]:
#         image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
#     alpha_dense_gt = 0.2 * (dense_gt_label > 0)
#     alpha_dense_pred = 0.2 * (dense_pred > 0)

#     fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=100)

#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)

#     # ===== LEFT: IMAGE + Dense_GT  =====
#     axes[0].imshow(image, alpha=1.0)
#     axes[0].imshow(dense_gt_label, cmap=cmap, norm=norm, alpha=0.5)


#     axes[0].set_title("Image + GT")
#     axes[0].axis("off")

#     # ===== Middle: IMAGE + Dense_PRED =====
#     axes[1].imshow(image, alpha=1.0)
#     axes[1].imshow(dense_pred, cmap=cmap, norm=norm, alpha=0.5)
#     # axes[1].imshow(image)
#     # axes[1].imshow(dense_pred, cmap=cmap, norm=norm, alpha=alpha_dense_pred)
#     axes[1].set_title("Image + Pred")
#     axes[1].axis("off")

#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
#     plt.close(fig)

from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt


cmap = ListedColormap([
    [0, 0, 0, 0],        # background
    [1, 0.4, 0.4, 1],    # LV -> light red
    [0.4, 1, 0.4, 1],    # LA -> light green
    [0.4, 0.6, 1, 1],    # RA -> light blue
])

def get_mask_borders(mask, label):
    """Extract border pixels for a specific label in the mask."""
    binary = (mask == label)
    eroded = binary_erosion(binary)
    return binary & ~eroded

def save_overlay(img, gt_mask, pred_mask,  vis_save_path, alpha=0.3):
    """
    Save GT and prediction mask overlays side-by-side in a single figure.

    Args:
        img (np.ndarray): Grayscale shape (H, W) uint8 or float.
        gt_mask (np.ndarray): Ground truth segmentation mask, shape (H, W), integer labels in [0, 3].
        pred_mask (np.ndarray): Predicted segmentation mask, shape (H, W), integer labels in [0, 3].
        vis_save_path (str): Full path (including filename) to save the output figure.
        alpha (float): Overlay transparency for non-zero mask regions. Default 0.3.
    """
    alpha_dense_gt   = alpha * (gt_mask > 0)
    alpha_dense_pred = alpha * (pred_mask > 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # GT
    axes[0].imshow(img, cmap="gray")
    axes[0].imshow(gt_mask, cmap=cmap, alpha=alpha_dense_gt, vmin=0, vmax=3)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    # Prediction
    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(pred_mask, cmap=cmap, alpha=alpha_dense_pred, vmin=0, vmax=3)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    # Overlay GT borders on prediction panel
    label_colors = {
        1: (1.0, 0.0, 0.0),   # red
        2: (0.0, 1.0, 0.0),   # green
        3: (0.0, 0.0, 1.0),   # blue
    }
    H, W = gt_mask.shape
    for label, color in label_colors.items():
        border = get_mask_borders(gt_mask, label)
        if not border.any():
            continue
        # Build RGBA image: color on border pixels, transparent elsewhere
        rgba = np.zeros((H, W, 4), dtype=float)
        rgba[border, :3] = color
        rgba[border, 3]  = 1.0          # fully opaque border
        axes[1].imshow(rgba)

    axes[1].set_title("Prediction (GT borders)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(vis_save_path, dpi=200)
    plt.close()
    
    

def tensor_to_flat_dict(prefix, x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    if isinstance(x, (list, np.ndarray)):
        x = np.array(x).flatten()
        return {prefix: x}  

    return {prefix: x}

import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

N_DISCS = 20  # number of discs LV is divided into for Simpson's biplane method
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_lv_length_and_diameters(mask, n_discs=N_DISCS):
    """
    Given a binary LV mask (H, W), computes the LV long-axis length (px)
    and n_discs disc diameters (px), approximating the ASE/EACVI 2015
    Chamber Quantification guideline (Lang et al.) as closely as a raw
    predicted mask allows:

        "At the mitral valve level, the contour is closed by connecting
        the two opposite sections of the mitral ring with a straight
        line. LV length is defined as the distance between the bisector
        of this line and the apical point of the LV contour, which is
        most distant to it."

    A predicted mask has no annotated mitral annulus points, so the base
    ("mitral valve level") is approximated as the cross-section of
    maximum width along the long axis — the LV cavity tapers
    monotonically from base to apex, so the widest cross-section should
    sit at or near the annulus. Everything beyond that level (e.g. mask
    spillover into the left atrium, a known LV-segmentation failure mode
    near the base) is excluded from both the length and the disc
    measurements, mirroring the guideline's explicit closing-off step —
    this is the main thing a plain bounding-box span does NOT do.

    Long axis direction is still estimated via PCA on the foreground
    pixel cloud (no manually traced apex/base points exist on a predicted
    mask), so this remains an approximation of the guideline's
    landmark-based procedure, not a literal implementation of it.

    Discs are always ordered apex -> base (index 0 nearest apex), using
    an explicit apex/base identification below — NOT just
    proj_long.min()/max() — so that disc index i means the same
    anatomical depth in this mask as in whichever other view's mask it
    gets paired with in simpsons_biplane_volume.
    """
    if hasattr(mask, "numpy"):
        mask = mask.numpy()
    mask = np.asarray(mask) > 0.5  # classifying confidence of mask after converting it to a boolean array

    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        raise ValueError("LV mask has too few foreground pixels to compute discs")

    # below, we are computing the centroid and covariance matrix by treating the foreground pixels as a point cloud
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    centroid = points.mean(axis=0)
    centered = points - centroid
    # covariance matrix
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_axis_dir = eigvecs[:, np.argmax(eigvals)]
    long_axis_dir = long_axis_dir / np.linalg.norm(long_axis_dir)
    short_axis_dir = np.array([-long_axis_dir[1], long_axis_dir[0]])

    proj_long = centered @ long_axis_dir  # long axis projection
    proj_short = centered @ short_axis_dir

    lo, hi = proj_long.min(), proj_long.max()
    if hi <= lo:
        raise ValueError("Degenerate LV mask — zero-length long axis")

    # --- locate the base ("mitral valve level") as the widest cross-section ---
    # finer resolution than n_discs, just for finding where the base sits
    n_profile_bins = max(n_discs * 2, 40)
    bin_edges = np.linspace(lo, hi, n_profile_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    profile_widths = np.zeros(n_profile_bins)
    profile_extents = [None] * n_profile_bins  # (min_short, max_short) per bin

    for i in range(n_profile_bins):
        if i < n_profile_bins - 1:
            band = (proj_long >= bin_edges[i]) & (proj_long < bin_edges[i + 1])
        else:
            band = (proj_long >= bin_edges[i]) & (proj_long <= bin_edges[i + 1])
        if np.any(band):
            w = proj_short[band]
            profile_widths[i] = w.max() - w.min()
            profile_extents[i] = (w.min(), w.max())

    base_bin = int(np.argmax(profile_widths))
    if profile_extents[base_bin] is None:
        raise ValueError("Could not locate LV base — empty width profile")
    base_t = bin_centers[base_bin]
    base_short_min, base_short_max = profile_extents[base_bin]
    base_bisector_short = (base_short_min + base_short_max) / 2.0

    # --- apex = the long-axis extreme farther from the base ---
    apex_t = lo if abs(lo - base_t) >= abs(hi - base_t) else hi

    # apex point's short-axis position, so length is a true straight-line
    # distance (apex point -> bisector of the mitral-annulus line), not
    # just a projected difference
    tol = (hi - lo) / n_profile_bins
    apex_band = np.abs(proj_long - apex_t) < tol
    apex_short = proj_short[apex_band].mean() if np.any(apex_band) else 0.0

    long_axis_length_px = np.sqrt((apex_t - base_t) ** 2 + (apex_short - base_bisector_short) ** 2)
    if long_axis_length_px <= 0:
        raise ValueError("Degenerate LV mask — zero-length long axis")

    # --- disc bands span ONLY from the base level to the apex, ordered ---
    # --- apex -> base regardless of which end (lo/hi) is physically apex ---
    disc_height_proj = abs(base_t - apex_t) / n_discs
    direction = np.sign(base_t - apex_t) or 1.0
    level_ts = apex_t + direction * disc_height_proj * (np.arange(n_discs) + 0.5)

    diameters_px = np.zeros(n_discs)
    for i, t in enumerate(level_ts):
        band = np.abs(proj_long - t) < disc_height_proj / 2
        if not np.any(band):
            diameters_px[i] = 0.0
            continue
        widths = proj_short[band]
        diameters_px[i] = widths.max() - widths.min()

    return long_axis_length_px, diameters_px


def simpsons_biplane_volume(mask_a4c, mask_a2c, spacing_a4c, spacing_a2c, n_discs=N_DISCS):
    """
    LV volume (mL), biplane Method of Discs (Lang et al. 2015, ASE/EACVI
    Chamber Quantification guideline). spacing_* is (dX, dY) in CM/PIXEL
    — if yours are mm/pixel, divide the returned volume by 1000, or
    convert spacing to cm before calling this.
    """
    length_a4c_px, diam_a4c_px = get_lv_length_and_diameters(mask_a4c, n_discs=n_discs)
    length_a2c_px, diam_a2c_px = get_lv_length_and_diameters(mask_a2c, n_discs=n_discs)

    px_to_cm_a4c = 1
    px_to_cm_a2c = 1

    
    diam_a4c_cm = diam_a4c_px * px_to_cm_a4c
    diam_a2c_cm = diam_a2c_px * px_to_cm_a2c
    length_a4c_cm = length_a4c_px * px_to_cm_a4c
    length_a2c_cm = length_a2c_px * px_to_cm_a2c

    # ASE: "the use of the LONGER LV length between the apical two- and
    # four-chamber views is recommended" (Lang et al. 2015, p.4)
    L_cm = max(length_a4c_cm, length_a2c_cm)
    volume_cm3 = (np.pi / 4.0) * (L_cm / n_discs) * np.sum(diam_a4c_cm * diam_a2c_cm)
    return volume_cm3, length_a4c_px, length_a2c_px  # cm^3 == mL

