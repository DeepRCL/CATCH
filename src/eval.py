"""Evaluation script: segmentation metrics + biplane Simpson EF / inter-view consistency."""

import os

# Must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.makedirs("/data/tmp", exist_ok=True)
os.environ.setdefault("TMPDIR", "/data/tmp")

import argparse
import logging
import sys
from pathlib import Path

import cv2
import imageio
import lightning as L
import numpy as np
import pandas as pd
import torch
from monai.transforms import KeepLargestConnectedComponent
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.public_datasets import EchoNet_Dynamic, calc_ef_from_seg, camus_test
from datasets.test_echo_segmentation_dataset import EchoSegmentationDataset
from model import Model
from utils import calc_batch_seg_metrics, save_overlay, tensor_to_flat_dict, simpsons_biplane_volume

logger = logging.getLogger(__name__)

N_CLASSES = 4
SIGNIFICANT_DIFF_THRESHOLD = 0.10  # relative LV-axis difference considered "consistent"

keep_largest = KeepLargestConnectedComponent(
    applied_labels=[1, 2, 3],  # channel indices to process (skip channel 0 = background)
    is_onehot=True,
    independent=True,
    connectivity=1,
)


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────
def _flatten_frames(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten per-sample (num_frames, C, H, W) clips into one frame-major batch."""
    image_list, masks_list = [], []
    for b in batch:
        images = b["images"]
        masks = b.get("masks")
        for i in range(images.shape[0]):
            image_list.append(images[i])
            if masks is not None:
                masks_list.append(masks[i])
    return torch.stack(image_list), torch.stack(masks_list)


def collate_fn(batch: list) -> dict:
    """Collate for the private echo dataset (carries spacing / patient metadata)."""
    images, masks = _flatten_frames(batch)
    return {
        "images": images,
        "masks": masks,
        "spacing": [b["spacing"] for b in batch],
        "labels_name": [b["labels_name"] for b in batch],
        "pred_view": [b["pred_view"] for b in batch],
        "stream_id": [b["stream_id"] for b in batch],
        "patient_id": [b["patient_id"] for b in batch],
        "ef_visual": [b["ef_visual"] for b in batch],
    }


def collate_fn_public(batch: list) -> dict:
    """Collate for CAMUS / EchoNet."""
    images, masks = _flatten_frames(batch)
    return {
        "images": images,
        "masks": masks,
        "pred_view": [b["pred_view"] for b in batch],
        "ef_numeric": [b["ef_numeric"] for b in batch],
        "ef_visual": [b["ef_visual"] for b in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────
def _to_masks(images, labels, pred_masks, idx: int):
    img = images[idx].detach().cpu().squeeze().numpy()
    if labels.shape[1] == 1:
        gt_mask = (labels[idx][0].sigmoid() > 0.5).to(torch.uint8).cpu().numpy()
    else:
        gt_mask = labels[idx].argmax(0).cpu().numpy()
    pred_mask = pred_masks[idx].argmax(0).cpu().numpy()
    return img, gt_mask, pred_mask


def save_sample_overlay(images, labels, pred_masks, path: str, drop_label: int | None = None):
    img, gt_mask, pred_mask = _to_masks(images, labels, pred_masks, 0)
    if drop_label is not None:
        pred_mask = pred_mask.copy()
        pred_mask[pred_mask == drop_label] = 0
    save_overlay(img, gt_mask, pred_mask, vis_save_path=path)


def save_video_gif(images, labels, pred_masks, out_dir: str, tag: str, fps_duration: float = 0.5):
    frames = []
    for i in range(images.shape[0]):
        img, gt_mask, pred_mask = _to_masks(images, labels, pred_masks, i)
        frame_path = f"{out_dir}/{tag}_frame_{i}.png"
        save_overlay(img, gt_mask, pred_mask, vis_save_path=frame_path)
        frames.append(cv2.imread(frame_path))

    # savefig can produce slightly different dims per frame; pad to a common size.
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    frames = [
        cv2.copyMakeBorder(
            f, 0, max_h - f.shape[0], 0, max_w - f.shape[1],
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
        for f in frames
    ]

    gif_path = f"{out_dir}/{tag}.gif"
    imageio.mimsave(gif_path, frames, duration=fps_duration, loop=0)
    logger.info(f"Saved GIF as {gif_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Biplane Simpson consistency
# ─────────────────────────────────────────────────────────────────────────────
def _lv_axis_consistency(a4c_axis: float, a2c_axis: float) -> tuple[float, bool]:
    """Absolute LV-axis difference between views, and whether it is within threshold."""
    diff = abs(a4c_axis - a2c_axis)
    denom = max(a4c_axis, a2c_axis)
    is_consistent = bool(denom > 0 and diff / denom <= SIGNIFICANT_DIFF_THRESHOLD)
    return diff, is_consistent


def biplane_consistency(pair, gt_pair, pred_volume_pair, gt_volume_pair, sx=1.0, sy=1.0) -> dict:
    """Compare predicted vs. ground-truth biplane Simpson volumes across A4C/A2C."""
    edv_ml, ed_a4c, ed_a2c = simpsons_biplane_volume(pair["a4c"][0], pair["a2c"][0], sx, sy)
    esv_ml, es_a4c, es_a2c = simpsons_biplane_volume(pair["a4c"][1], pair["a2c"][1], sx, sy)
    gt_edv_ml, g_ed_a4c, g_ed_a2c = simpsons_biplane_volume(gt_pair["a4c"][0], gt_pair["a2c"][0], sx, sy)
    gt_esv_ml, g_es_a4c, g_es_a2c = simpsons_biplane_volume(gt_pair["a4c"][1], gt_pair["a2c"][1], sx, sy)

    ed_err, ed_label = _lv_axis_consistency(ed_a4c, ed_a2c)
    es_err, es_label = _lv_axis_consistency(es_a4c, es_a2c)
    gt_ed_err, gt_ed_label = _lv_axis_consistency(g_ed_a4c, g_ed_a2c)
    gt_es_err, gt_es_label = _lv_axis_consistency(g_es_a4c, g_es_a2c)

    ef_pred = (edv_ml - esv_ml) / edv_ml * 100.0 if edv_ml > 0 else float("nan")
    gt_ef = (gt_edv_ml - gt_esv_ml) / gt_edv_ml * 100.0 if gt_edv_ml > 0 else float("nan")

    # Inter-view volume disagreement, predicted vs. the irreducible ground-truth floor.
    def volume_gap(d, phase):
        return abs(d["a4c"][phase] - d["a2c"][phase])

    return {
        "ed_error": ed_err,
        "es_error": es_err,
        "consistency_error": abs((ed_err + es_err) - (gt_ed_err + gt_es_err)),
        "ed_label": ed_label,
        "es_label": es_label,
        "gt_ed_label": gt_ed_label,
        "gt_es_label": gt_es_label,
        "ef_pred": ef_pred,
        "gt_ef": gt_ef,
        "ef_abs_error": abs(ef_pred - gt_ef) if not np.isnan(gt_ef) else np.nan,
        "ed_volume_diff": abs(volume_gap(pred_volume_pair, 0) - volume_gap(gt_volume_pair, 0)),
        "es_volume_diff": abs(volume_gap(pred_volume_pair, 1) - volume_gap(gt_volume_pair, 1)),
    }


def _as_float(values: list) -> list:
    return [v.detach().cpu().item() if torch.is_tensor(v) else float(v) for v in values]


def mean_value(values: list) -> float:
    return float(np.mean(_as_float(values))) if values else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    log_dir: Path,
    dataset_name: str,
    video_visualization: bool = False,
    calc_ef: bool = True,
) -> dict:
    model.eval()
    mae_loss = nn.L1Loss(reduction="none")
    vis_output_dir = os.path.join(str(log_dir), "visualization")
    os.makedirs(vis_output_dir, exist_ok=True)

    all_dice, all_iou, all_hd95 = [], [], []
    all_mae, all_pred_ef, all_gt_ef = [], [], []
    ed_error, es_error, consistency_error = [], [], []
    ed_volume_diff, es_volume_diff, abs_error = [], [], []
    all_ed_label, all_es_label, consistency_classification = [], [], []

    # Per-patient A4C/A2C accumulators, flushed once both views have been seen.
    pair, gt_pair, pred_volume_pair, gt_volume_pair = {}, {}, {}, {}

    for batch_itr, batch in enumerate(tqdm(dataloader)):
        images = batch["images"].to(device)
        labels = batch["masks"].to(device).squeeze(1)
        pred_view = batch["pred_view"]

        if dataset_name == "private_echo":
            spacing = batch["spacing"][0].tolist() if torch.is_tensor(batch["spacing"][0]) else list(batch["spacing"][0])
            sx, sy = spacing[0], spacing[1]
            ef_visual = batch["ef_visual"][0]
            ef_numeric = ef_visual
        else:
            spacing = [0, 0]
            sx = sy = 1.0
            ef_visual = batch["ef_visual"][0]
            ef_numeric = batch["ef_numeric"][0]

        pred_logits = model.test(images=images, view_name=pred_view)

        # (B, C, H, W) one-hot prediction, then largest-connected-component cleanup.
        dense_pred_labels = torch.softmax(pred_logits, dim=1).argmax(1)
        pred_onehot = nn.functional.one_hot(dense_pred_labels, num_classes=N_CLASSES)
        pred_onehot = pred_onehot.permute(0, 3, 1, 2).float()
        pred_masks = torch.stack([keep_largest(p) for p in pred_onehot], dim=0)

        # ── Visualisation ────────────────────────────────────────────────
        tag = f"{dataset_name}_sample_{batch_itr}"
        if dataset_name == "camus" and batch_itr % 50 == 0:
            # Right atrium (label 3) is not annotated in CAMUS; drop it from the overlay.
            save_sample_overlay(images, labels, pred_masks, f"{vis_output_dir}/{tag}.png", drop_label=3)
        elif video_visualization and batch_itr < 100:
            save_video_gif(images, labels, pred_masks, vis_output_dir, f"{dataset_name}_video_{batch_itr}")
        elif batch_itr % 125 == 0:
            save_sample_overlay(images, labels, pred_masks, f"{vis_output_dir}/{tag}.png")

        # ── Segmentation metrics ─────────────────────────────────────────
        gt_onehot = labels
        if dataset_name == "echonet":
            pred_masks = pred_masks[:, :2, :, :]          # EchoNet annotates LV only
        elif dataset_name == "camus":
            pred_masks = pred_masks[[0, -1], :, :, :]     # ED / ES frames only
            gt_onehot = gt_onehot[[0, -1], :, :, :]

        dice_dense, iou_dense, hd_dense = calc_batch_seg_metrics(pred_masks, gt_onehot, spacing)
        all_dice.append(dice_dense)
        all_iou.append(iou_dense)
        all_hd95.append(hd_dense)

        # ── Ejection fraction ────────────────────────────────────────────
        if calc_ef:
            if dataset_name == "private_echo":
                # Annotated ED/ES frames are the two with a non-empty LV channel.
                idx = (gt_onehot[:, 1].sum(dim=(1, 2)) > 0).nonzero(as_tuple=True)[0]
                if len(idx) != 2:
                    torch.cuda.empty_cache()
                    continue
                ed_index, es_index = idx[0].item(), idx[1].item()
            else:
                ed_index, es_index = 0, -1

            pred_ef, pred_edv, pred_esv = calc_ef_from_seg(
                pred_masks[ed_index, 1:2, :, :], pred_masks[es_index, 1:2, :, :]
            )
            gt_ef = ef_visual.to(pred_ef.device) if torch.is_tensor(ef_visual) else torch.as_tensor(
                ef_visual, dtype=pred_ef.dtype, device=pred_ef.device
            )
            mae = mae_loss(pred_ef, gt_ef).squeeze(-1)
            all_mae.append(mae.detach().float())
            all_pred_ef.append(pred_ef.item())
            all_gt_ef.append(gt_ef.detach().cpu())
            logger.info(f"pred_ef={pred_ef}, gt_ef_visual={gt_ef}, numeric={ef_numeric}, mae={mae}")

            # Accumulate this view; run the biplane comparison once both views are in.
            view = "a4c" if str(pred_view[0]).lower().startswith("a4") else "a2c"
            pair[view] = (pred_masks[ed_index, 1:2].squeeze().cpu(), pred_masks[es_index, 1:2].squeeze().cpu())
            gt_pair[view] = (gt_onehot[ed_index, 1:2].squeeze().cpu(), gt_onehot[es_index, 1:2].squeeze().cpu())
            pred_volume_pair[view] = (pred_edv, pred_esv)
            _, gt_edv, gt_esv = calc_ef_from_seg(
                gt_onehot[ed_index, 1:2, :, :], gt_onehot[es_index, 1:2, :, :]
            )
            gt_volume_pair[view] = (gt_edv, gt_esv)

            if "a2c" in pair and "a4c" in pair:
                res = biplane_consistency(pair, gt_pair, pred_volume_pair, gt_volume_pair, sx, sy)
                ed_error.append(res["ed_error"])
                es_error.append(res["es_error"])
                consistency_error.append(res["consistency_error"])
                all_ed_label.append(res["ed_label"])
                all_es_label.append(res["es_label"])
                if res["gt_ed_label"]:
                    consistency_classification.append(res["ed_label"] == res["gt_ed_label"])
                if res["gt_es_label"]:
                    consistency_classification.append(res["es_label"] == res["gt_es_label"])
                abs_error.append(res["ef_abs_error"])
                ed_volume_diff.append(res["ed_volume_diff"])
                es_volume_diff.append(res["es_volume_diff"])
                logger.info(
                    f"ef_pred={res['ef_pred']:.2f}, gt_ef={res['gt_ef']:.2f}, "
                    f"ae={res['ef_abs_error']}"
                )
                for d in (pair, gt_pair, pred_volume_pair, gt_volume_pair):
                    d.clear()

        torch.cuda.empty_cache()

    # ── Aggregate ────────────────────────────────────────────────────────
    all_dense_dice = torch.cat(all_dice, dim=0).to(torch.float64)
    all_dense_iou = torch.cat(all_iou, dim=0).to(torch.float64)
    all_dense_hd95 = torch.cat(all_hd95, dim=0).to(torch.float64)

    torch.save(all_dense_dice, f"{log_dir}/all_dice.pt")
    torch.save(all_dense_hd95, f"{log_dir}/all_hd95.pt")
    torch.save(all_mae, f"{log_dir}/all_mae.pt")
    torch.save(all_pred_ef, f"{log_dir}/pred_ef.pt")
    torch.save(all_gt_ef, f"{log_dir}/gt_ef.pt")
    torch.save(abs_error, f"{log_dir}/abs_error.pt")

    logger.info(f"consistency_classification: {consistency_classification}")
    logger.info(
        f"ed consistency error: {mean_value(ed_error)}, "
        f"es: {mean_value(es_error)}, "
        f"consistency_error: {mean_value(consistency_error)}"
    )
    logger.info(
        f"abs_error mean: {np.nanmean(abs_error) if abs_error else float('nan')}, "
        f"ed_volume_diff mean: {mean_value(ed_volume_diff)}, "
        f"es_volume_diff mean: {mean_value(es_volume_diff)}"
    )

    overall_mae = (
        torch.stack(all_mae, dim=0).to(torch.float64).mean().item()
        if all_mae
        else float("nan")
    )

    dice_per_class = torch.nanmean(all_dense_dice, dim=0)
    iou_per_class = torch.nanmean(all_dense_iou, dim=0)
    hd95_per_class = torch.nanmean(all_dense_hd95, dim=0)
    logger.info(f"Testing results: dense_dice_per_class={dice_per_class}")

    final_metrics = {
        "dense_dice_per_class": dice_per_class,
        "dense_dice_overall": torch.nanmean(all_dense_dice),
        "dense_iou_per_class": iou_per_class,
        "dense_iou_overall": torch.nanmean(all_dense_iou),
        "dense_hd95_per_class": hd95_per_class,
        "dense_hd95_overall": torch.nanmean(all_dense_hd95),
        "dice": dice_per_class.mean(),
        "iou": iou_per_class.mean(),
        "hd95": hd95_per_class.mean(),
    }

    row = {"overall_mae": overall_mae}
    for key, value in final_metrics.items():
        row.update(tensor_to_flat_dict(key, value))
    pd.DataFrame([row]).to_csv(f"{log_dir}/overall_metrics.csv", index=False)

    return final_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-name", type=str, default="private_echo",
                        choices=["private_echo", "camus", "echonet"])
    parser.add_argument("--start-ckpt", type=str, required=True)
    parser.add_argument(
        "--base-config-path",
        type=str,
        default="configs/eval_echo_linear.yaml",
    )

    # Model hyperparameters (must match the checkpoint).
    parser.add_argument("--use-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-late-fusion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--output-dim", type=int, default=1024)
    parser.add_argument("--n-classes", type=int, default=N_CLASSES)

    parser.add_argument("--video-visualization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--calc-ef", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_dataloader(args) -> DataLoader:
    if args.dataset_name == "private_echo":
        dataset = EchoSegmentationDataset(split="test", visualize_video=args.video_visualization)
        collate = collate_fn
    elif args.dataset_name == "camus":
        dataset = camus_test(split="test")
        collate = collate_fn_public
    elif args.dataset_name == "echonet":
        dataset = EchoNet_Dynamic(split="test")
        collate = collate_fn_public
    else:
        raise NotImplementedError(f"Dataset {args.dataset_name} is not implemented")

    logger.info(f"test set size: {len(dataset)}")
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        collate_fn=collate,
        pin_memory=False,
        persistent_workers=False,
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler((log_dir / "test.log").as_posix()),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    L.seed_everything(args.seed)

    dataloader_test = build_dataloader(args)

    net = Model(
        base_config_path=args.base_config_path,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        n_seg_classes=args.n_classes,
        use_cross_attention=args.use_attention,
        use_late_fusion = args.use_late_fusion,
    )

    ckpt = torch.load(args.start_ckpt, map_location="cpu")
    msg = net.load_state_dict(ckpt["state_dict"], strict=False)
    logger.info(f"missing keys ({len(msg.missing_keys)}): {msg.missing_keys}")
    logger.info(f"unexpected keys ({len(msg.unexpected_keys)}): {msg.unexpected_keys}")

    net.to(device)
    net.eval()

    metrics = test(
        model=net,
        dataloader=dataloader_test,
        device=device,
        log_dir=log_dir,
        dataset_name=args.dataset_name,
        video_visualization=args.video_visualization,
        calc_ef=args.calc_ef,
    )
    logger.info(f"Test metrics: {metrics}")


if __name__ == "__main__":
    main()