# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from functools import partial
import logging
import numpy as np
import os
import random
import wandb

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, WeightedRandomSampler

from dinov3.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader, make_dataset
import dinov3.distributed as distributed
from dinov3.eval.segmentation.eval import evaluate_segmentation_model
from dinov3.eval.segmentation.loss import MultiSegmentationLoss
from dinov3.eval.segmentation.metrics import SEGMENTATION_METRICS
from dinov3.eval.segmentation.models import build_segmentation_decoder
from dinov3.eval.segmentation.schedulers import build_scheduler
from dinov3.eval.segmentation.transforms import make_segmentation_eval_transforms, make_segmentation_train_transforms
from dinov3.logging import MetricLogger, SmoothedValue

logger = logging.getLogger("dinov3")


import matplotlib.pyplot as plt
import numpy as np

def log_overlay_masks(image, dense_pred, sparse_pred,  gt, title=None):
    """
    Plot side-by-side overlays:
        Left: image + GT mask
        Right: image + Pred mask
    Mask alpha = 0.5 where mask > 0, else 0.
    """
    # Make alpha masks
    dense_gt_label = gt[:5].argmax(axis=0).detach().cpu().numpy()
    # gt_label = gt.argmax(axis=0)
    alpha_dense_gt = 0.5 * (dense_gt_label > 0)
    alpha_dense_pred = 0.5 * (dense_pred > 0)
    
    # sparse_gt = gt[5:]
    # sparse_gt_label = (gt[5:].sigmoid() > 0.5).float()
    # class_ids = torch.arange(5, 13, device=sparse_pred.device).view(-1, 1, 1)
    # sparse_gt_label = (sparse_gt_label * class_ids).max(dim=0).values # H, W
    # sparse_label_map = (sparse_pred * class_ids).max(dim=0).values # H, W
    # ===== Sparse GT processing =====
    sparse_gt_bin = gt[5:].float()   # [C, H, W]
    # sparse_gt_bin = (gt[5:].sigmoid() > 0.5).float()   # [C, H, W]
    class_ids = torch.arange(5, 13, device=sparse_pred.device).view(-1, 1, 1) # [C, 1, 1] with 

    # Sparse GT label map
    sparse_gt_label = (sparse_gt_bin * class_ids).max(dim=0).values  #   max on Dim=0 ([C, H, W] * [C, 1, 1] ) > [H, W]

    # ===== keep only GT-present classes =====
    # class presence: [C]
    gt_class_present = (sparse_gt_bin.sum(dim=(1, 2)) > 0).float()   # [C]

    # mask predictions by GT class presence
    sparse_pred_masked = sparse_pred * gt_class_present.view(-1, 1, 1)

    # Sparse Pred label map (GT-aware)
    sparse_label_map = (sparse_pred_masked * class_ids).max(dim=0).values
    
    # Alpha for sparse
    alpha_sparse_gt = 0.5 * (sparse_gt_label > 0)
    alpha_sparse_pred = 0.5 * (sparse_label_map > 0)

    fig, axes = plt.subplots(1, 4, figsize=(12, 4), dpi=150)

    # Remove whitespace
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)

    # ===== LEFT: IMAGE + Dense_GT  =====
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].imshow(dense_gt_label, cmap="viridis", vmin=0, vmax=4, alpha=alpha_dense_gt)
    axes[0].set_title("Image + Dense GT")
    axes[0].axis("off")

    # ===== Middle: IMAGE + Dense_PRED =====
    axes[1].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(dense_pred, cmap="viridis", vmin=0, vmax=4, alpha=alpha_dense_pred)
    axes[1].set_title("Image + Dense Pred")
    axes[1].axis("off")
    
    
    # ===== Middle: IMAGE + Sparse_GT =====
    axes[2].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(sparse_gt_label.detach().cpu().numpy(), cmap="viridis", vmin=0, vmax=12, alpha=alpha_sparse_gt.detach().cpu().numpy())
    axes[2].set_title("Image + Sparse GT")
    axes[2].axis("off")
    
    
    # ===== Right: IMAGE + Sparse_PRED =====
    axes[3].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(sparse_label_map.detach().cpu().numpy(), cmap="viridis", vmin=0, vmax=12, alpha=alpha_sparse_pred.detach().cpu().numpy())
    axes[3].set_title("Image + Sparse Pred")
    axes[3].axis("off")
    
    wandb_img = wandb.Image(fig, caption=title)
    plt.close(fig)
    return wandb_img

class InfiniteDataloader:
    def __init__(self, dataloader: torch.utils.data.DataLoader):
        self.dataloader = dataloader
        self.data_iterator = iter(dataloader)
        self.sampler = dataloader.sampler
        if not hasattr(self.sampler, "epoch"):
            self.sampler.epoch = 0  # type: ignore

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return len(self.dataloader)

    def __next__(self):
        try:
            data = next(self.data_iterator)
        except StopIteration:
            self.sampler.epoch += 1
            self.data_iterator = iter(self.dataloader)
            data = next(self.data_iterator)
        return data


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.
    The seed of each worker equals to num_worker * rank + worker_id + user_seed
    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def validate(
    segmentation_model: torch.nn.Module,
    val_dataloader,
    device,
    autocast_dtype,
    eval_res,
    eval_stride,
    decoder_head_type,
    num_classes,
    global_step,
    metric_to_save,
    current_best_metric_to_save_value,
    output_dir,
):
    new_metric_values_dict = evaluate_segmentation_model(
        segmentation_model,
        val_dataloader,
        device,
        eval_res,
        eval_stride,
        decoder_head_type,
        num_classes,
        autocast_dtype,
        output_dir,
    )
    logger.info(f"Step {global_step}: {new_metric_values_dict}")
    # `segmentation_model` is a module list of [backbone, decoder]
    # Only put the head in train mode
    segmentation_model.module.segmentation_model[1].train()
    is_better = False
    if new_metric_values_dict[metric_to_save] > current_best_metric_to_save_value:
        is_better = True
    return is_better, new_metric_values_dict


def train_step(
    segmentation_model: torch.nn.Module,
    batch,
    device,
    scaler,
    optimizer,
    optimizer_gradient_clip,
    scheduler,
    criterion,
    model_dtype,
    global_step,
):
    # a) load batch
    batch_img = batch["image"]        # (B, C, H, W)
    (_, gt) = batch["target"]      # (B, 13, 512, 512)
    gt_coords  = batch["points"]      # (B, 13, 2, 2)
    dx      = batch["dx"]          # (B,)
    dy      = batch["dy"]          # (B,)
    pred_view   = batch["pred_view"]   # list of length B
    # batch_img, (_, gt), _, gt_coords, dx, dy, pred_view  = batch
    # batch_img, gt = batch
    batch_img = batch_img.to(device)  # B x C x h x w
    gt = gt.to(device)  # B x (num_classes if multilabel) x h x w
    gt_coords = gt_coords.to(device)
    optimizer.zero_grad(set_to_none=True)

    # b) forward pass
    with torch.autocast("cuda", dtype=model_dtype, enabled=True if model_dtype is not None else False):
        pred = segmentation_model(batch_img)  # B x num_classes x h x w
        gt = torch.squeeze(gt).long()  # Adapt gt dimension to enable loss calculation
        # gt = gt.squeeze(0).long() 
        

    # c) compute loss
    if gt.shape[-2:] != pred.shape[-2:]:
        pred = torch.nn.functional.interpolate(input=pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    # print(pred.shape, gt.shape)
    loss_dict = criterion(pred, gt, gt_coords)
    loss = loss_dict["total_loss"]

    wandb.log(
        {'preds': log_overlay_masks(
            batch_img[0][0].detach().cpu().numpy(),
            pred[0, :5].argmax(dim=0, keepdim=False).detach().cpu().numpy(),
            (pred[0, 5:].sigmoid() > 0.5).float(),
            gt[0],
        )},
        step=global_step,
    )
    # d) optimization
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(segmentation_model.module.parameters(), optimizer_gradient_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(segmentation_model.module.parameters(), optimizer_gradient_clip)
        optimizer.step()

    if global_step > 0:  # inheritance from old mmcv code
        scheduler.step()

    return loss_dict



def train_segmentation(
    backbone,
    config,
):
    assert config.decoder_head.type == "linear", "Only linear head is supported for training"
    # 1- load the segmentation decoder
    logger.info("Initializing the segmentation model")
    segmentation_model = build_segmentation_decoder(
        backbone,
        config.decoder_head.backbone_out_layers,
        "linear",
        num_classes=config.decoder_head.num_classes,
        autocast_dtype=config.model_dtype.autocast_dtype,
        dropout=config.decoder_head.dropout,
    )
    # continue training from /data/project/users/bassant/code/DINO_LIN/dinov3-echo/outputs/train_with_validation_13_classes_outputs_weighted_dice_loss_upsample_RV/outputs_training_151999/best_model_900.pth
    # load_from = "/data/project/users/bassant/code/DINO_LIN/dinov3-echo/outputs/train_with_validation_13_classes_outputs_weighted_dice_loss_upsample_RV/outputs_training_151999/best_model_900.pth"
    # logger.info(f"Load model from {load_from}")

    global_device = distributed.get_rank()
    local_device = torch.cuda.current_device()
    # local_rank = 0
    device = torch.device(f"cuda:{local_device}")
    segmentation_model = torch.nn.parallel.DistributedDataParallel(
        segmentation_model.to(local_device), device_ids=[local_device]
    )  # should be local rank
    # state_dict = torch.load(load_from, map_location=device)["model"]
    # _, _ = segmentation_model.load_state_dict(state_dict, strict=False)
    
    model_parameters = filter(lambda p: p.requires_grad, segmentation_model.parameters())
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model_parameters)}")
    logger.info(f"Trained Model parameters: {model_parameters}")
    # 2- create data transforms + dataloaders
    train_transforms = make_segmentation_train_transforms(
        img_size=config.transforms.train.img_size,
        random_img_size_ratio_range=config.transforms.train.random_img_size_ratio_range,
        crop_size=config.transforms.train.crop_size,
        flip_prob=config.transforms.train.flip_prob,
    )
    val_transforms = make_segmentation_eval_transforms(
        img_size=config.transforms.eval.img_size,
        inference_mode=config.eval.mode,
    )
    
    train_dataset = DatasetWithEnumeratedTargets(
        make_dataset(
            dataset_str=f"{config.datasets.train}",
            transforms=train_transforms,
            split = 'train'
        )
    )
    
    
    # class_label_counts = torch.tensor(
    #     [3000, 1000, 1000, 23, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],

    #     # [3731, 2000, 999, 23, 996, 1000, 999, 1000, 999, 999, 1000, 1000],
    #     dtype=torch.float
    # )
    # class_weights = 1.0 / (class_label_counts + 1e-6)

    # sample_weights = []
    # min_weight = 1e-3  # small positive number

    # for i in range(len(train_dataset)):
    #     _, (_, mask) = train_dataset[i]          # mask: (13, H, W)
    #     mask = mask.squeeze(0)
    #     print(f"mask: {mask.shape}")
    #     mask_fg = mask[1:, :, :]          # (12, H, W)
    #     print(f"mask_fg:{mask_fg.shape}")
    #     # find classes present (excluding background if desired)
    #     present = (mask_fg.sum(dim=(1,2)) > 0)  # (12,)
        
    #     # image weight = sum of class weights of present classes
    #     present_class_weights = class_weights[present]

    #     # Sample weight = max weight among present classes (rarest class)
    #     if len(present_class_weights) > 0:
    #         weight = present_class_weights.max()
    #     else:
    #         weight = torch.tensor(min_weight)


    #     sample_weights.append(weight)

    # # control how often each data sample is drawn by the data loader based on the weights of the labels distributions
    # sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    # sample_weights_np = sample_weights.cpu().numpy()
    
    # # modified_sample_weights = sample_weights / sample_weights.mean()
    # # modified_sample_weights = torch.clamp(modified_sample_weights, max=10.0)
    # # modified_sample_weights_np = modified_sample_weights.cpu().numpy()

    # np.save("sample_weights_np_2.npy", sample_weights_np)
    def collate_fn(batch):
            indices = torch.tensor([b["target"][0] for b in batch])
            masks   = torch.stack([b["target"][1] for b in batch])
            return {
                "image": torch.stack([b["image"] for b in batch]),
                "target": (indices, masks),
                # "target": torch.stack([b["target"] for b in batch]),
                "points": torch.stack([b["points"] for b in batch]),
                "dx": torch.stack([b["dx"] for b in batch]),
                "dy": torch.stack([b["dy"] for b in batch]),
                "label_type_list": [b["label_type_list"] for b in batch],
                "pred_view": [b["pred_view"] for b in batch],
            }

    sample_weights = np.load("/data/project/users/bassant/code/DINO_LIN/dinov3-echo/sample_weights_np_2.npy")
    train_sampler_type = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),  # usually same as dataset size
        replacement=True                  # IMPORTANT for upsampling
    )

    init_fn = partial(
        worker_init_fn, num_workers=config.num_workers, rank=global_device, seed=config.seed + global_device
    )
    train_dataloader = InfiniteDataloader(
        make_data_loader(
            dataset=train_dataset,
            batch_size=config.bs,
            num_workers=config.num_workers,
            sampler_type=train_sampler_type,
            shuffle=False,
            collate_fn = collate_fn,
            persistent_workers=False,
            worker_init_fn=init_fn,
        )
    )

    val_dataset = DatasetWithEnumeratedTargets(
        make_dataset(
            dataset_str=f"{config.datasets.val}",
            transforms=val_transforms,
            split = 'val'
        )
    )
    val_sampler_type = None
    if distributed.is_enabled():
        val_sampler_type = SamplerType.DISTRIBUTED
    val_dataloader = make_data_loader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=config.num_workers,
        sampler_type=val_sampler_type,
        drop_last=False,
        shuffle=False,
        collate_fn = collate_fn,
        persistent_workers=True,
    )


    # 3- define and create scaler, optimizer, scheduler, loss
    scaler = None
    if config.model_dtype.autocast_dtype is not None:
        scaler = torch.amp.GradScaler("cuda")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": filter(lambda p: p.requires_grad, segmentation_model.parameters()),
                "lr": config.optimizer.lr,
                "betas": (config.optimizer.beta1, config.optimizer.beta2),
                "weight_decay": config.optimizer.weight_decay,
            }
        ]
    )
    scheduler = build_scheduler(
        config.scheduler.type,
        optimizer=optimizer,
        lr=config.optimizer.lr,
        total_iter=config.scheduler.total_iter,
        constructor_kwargs=config.scheduler.constructor_kwargs,
    )
    criterion = MultiSegmentationLoss()
    total_iter = config.scheduler.total_iter
    global_step = 0
    global_best_metric_values = {metric: 0.0 for metric in SEGMENTATION_METRICS}

    # 5- train the model
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", SmoothedValue(window_size=4, fmt="{value:.3f}"))
    for batch in metric_logger.log_every(
        train_dataloader,
        50,
        header="Train: ",
        start_iteration=global_step,
        n_iterations=total_iter,
    ):
        loss = train_step(
            segmentation_model,
            batch,
            local_device,
            scaler,
            optimizer,
            config.optimizer.gradient_clip,
            scheduler,
            criterion,
            config.model_dtype.autocast_dtype,
            global_step,
        )
        global_step += 1
        metric_logger.update(**loss)
        if global_step % config.eval.eval_interval == 0:
            dist.barrier()
            is_better, best_metric_values_dict = validate(
                segmentation_model,
                val_dataloader,
                local_device,
                config.model_dtype.autocast_dtype,
                config.eval.crop_size,
                config.eval.stride,
                config.decoder_head.type,
                config.decoder_head.num_classes,
                global_step,
                config.metric_to_save,
                global_best_metric_values[config.metric_to_save],
                config.output_dir,
            )
            if is_better:
                logger.info(f"New best metrics at Step {global_step}: {best_metric_values_dict}")
                global_best_metric_values = best_metric_values_dict
                torch.save(
                    {
                        "model": {k: v for k, v in segmentation_model.module.state_dict().items() if "segmentation_model.1" in k},
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(config.output_dir, f"best_model_{global_step}_after_900.pth"),
                )

        break

    logger.info("Finsjing one epoch and Saving embeddings is done!")

    return global_best_metric_values
