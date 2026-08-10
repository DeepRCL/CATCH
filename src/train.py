"""Fine-tuning script for multi-view echo segmentation."""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.makedirs("/data/tmp", exist_ok=True)
os.environ.setdefault("TMPDIR", "/data/tmp")

import argparse
import gc
import logging
import sys
import time
from logging import Logger
from pathlib import Path


import lightning as L
import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.echo_segmentation_dataset_paired_patient import MultiViewEchoSegmentationDataset
from model import Model, load_pretrained
from utils import AverageMeter, log_gpu_usage, log_overlay_masks


def move_to_device(batch: dict, device: torch.device) -> dict:
    """Recursively move tensors (and lists of tensors) in a batch to `device`."""
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif isinstance(value, list) and all(
            isinstance(x, torch.Tensor) for x in value if x is not None
        ):
            out[key] = [x.to(device) if x is not None else None for x in value]
        else:
            out[key] = value
    return out


@torch.no_grad()
def gradient_norms(model: nn.Module) -> dict:
    """Total and per-top-level-module gradient L2 norms."""
    norms = {}
    total_sq = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        param_norm = param.grad.detach().norm(2).item()
        total_sq += param_norm ** 2
        key = f"grad_norm/{name.split('.')[0]}"
        norms[key] = norms.get(key, 0.0) + param_norm
    norms["grad_norm/total"] = total_sq ** 0.5
    return norms


def train(
    model: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    optimizer: optim.Optimizer,
    logger: Logger,
    device: torch.device,
    log_dir: Path,
    accumulation_steps: int = 32,
    log_every: int = 100,
    ckpt_every: int = 1000,
) -> None:
    batch_time = AverageMeter("Time", ":6.3f")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    end = time.time()
    steps_per_epoch = len(dataloader)

    for i, batch in enumerate(tqdm(dataloader, desc=f"epoch {epoch}")):
        global_step = epoch * steps_per_epoch + i
        batch = move_to_device(batch, device)

        # Forward / backward
        total_loss, log, all_outputs = model(batch)
        (total_loss / accumulation_steps).backward()

        batch_time.update(time.time() - end)
        end = time.time()

        if i <= 3:
            log_gpu_usage(log_dir)

        if i % log_every == 0:
            wandb.log(
                {
                    **log,
                    **gradient_norms(model),
                    "lr": optimizer.param_groups[0]["lr"],
                    "batch_time": batch_time.avg,
                },
                step=global_step,
            )

            image = batch["images"][0][0][0][0]
            label_map = batch["masks"][0][0].argmax(dim=0) + 1
            view_name = str(batch["views"][0][0])
            wandb.log(
                {
                    "preds": log_overlay_masks(
                        image.detach().cpu().numpy(),
                        all_outputs[0].argmax(dim=0).detach().cpu().numpy(),
                        label_map.detach().cpu().numpy(),
                        view_name,
                    )
                },
                step=global_step,
            )

        if (i + 1) % accumulation_steps == 0 or (i + 1) == steps_per_epoch:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        del total_loss, log, batch, all_outputs

        if i % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        if (i + 1) % ckpt_every == 0:
            save_checkpoint(model, log_dir / f"ckpt_ep_{epoch}_itr_{i + 1}.pth")
            logger.info(f"Saved checkpoint at epoch {epoch}, iteration {i + 1}")


def save_checkpoint(model: nn.Module, path: Path) -> None:
    torch.save(
        {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
        path,
    )


def set_requires_grad(model: nn.Module, n_last_blocks: int = 1) -> list:
    """Freeze the encoder except the last N blocks and the norm layers."""
    for param in model.parameters():
        param.requires_grad = False

    unfrozen = list(model.feature_model.blocks[-n_last_blocks:])
    unfrozen += [model.feature_model.local_cls_norm, model.feature_model.norm]
    for module in unfrozen:
        for param in module.parameters():
            param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in trainable_params)
    print(f"Total parameters:         {total}")
    print(f"Trainable parameters:     {trainable}")
    print(f"Non-trainable parameters: {total - trainable}")
    return trainable_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="mv-echo-seg")

    # Model
    parser.add_argument("--use-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-late-fusion", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--output-dim", type=int, default=1024)
    parser.add_argument("--use-ce", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument(
        "--base-config-path",
        type=str,
        default="configs/eval_echo_linear.yaml",
    )

    # Optimisation
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--backbone-lr", type=float, default=1.0e-4)
    parser.add_argument("--classifier-lr", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dataset-name", type=str, default="private_echo")
    parser.add_argument(
        "--start-ckpt",
        type=str,
        default="pretrained/ckpt_1.pth",
    )
    return parser.parse_args()


def build_dataloaders(args) -> tuple[DataLoader, DataLoader]:
    if args.dataset_name != "private_echo":
        raise NotImplementedError(f"Dataset {args.dataset_name} is not implemented")

    views = ["AP4", "AP2"]
    dataset_train = MultiViewEchoSegmentationDataset(split="train", req_views=views)
    dataset_val = MultiViewEchoSegmentationDataset(split="val", req_views=views)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=False,
        pin_memory=False,
        prefetch_factor=1,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        persistent_workers=False,
    )
    return dataloader_train, dataloader_val


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
            logging.FileHandler((log_dir / "train.log").as_posix()),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    L.seed_everything(args.seed)

    dataloader_train, dataloader_val = build_dataloaders(args)
    logger.info(f"train batches: {len(dataloader_train)}, val batches: {len(dataloader_val)}")

    net = Model(
        base_config_path=args.base_config_path,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        n_seg_classes=args.n_classes,
        use_cross_attention=args.use_attention,
        use_late_fusion = args.use_late_fusion,
        use_ce=args.use_ce,
    )

    param_groups = [{"params": list(net.decoder.parameters()), "lr": args.backbone_lr}]
    if args.use_late_fusion:
        param_groups.append(
            {"params": list(net.token_mixers.parameters()), "lr": args.backbone_lr}
        )
    else:
        param_groups.append(
            {"params": list(net.token_mixer.parameters()), "lr": args.backbone_lr}
        )
    
    param_groups.append({"params": set_requires_grad(net.encoder), "lr": args.backbone_lr})
    param_groups.append({"params": [net.spatial_pos_embed, net.view_embed], "lr": args.backbone_lr})
    
    optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay)
    num_trainable = sum(
        p.numel() for group in param_groups for p in group["params"] if p.requires_grad
    )
    logger.info(f"num_trainable params: {num_trainable}")

    if args.start_ckpt:
        net = load_pretrained(net, args.start_ckpt)
        logger.info(f"Continuing training from: {args.start_ckpt}")

    net.to(device)

    wandb.init(project=args.run_name, name=f"dice_{args.dataset_name}", config=vars(args))
    logger.info("Start fine-tuning ...")
    try:
        for epoch in range(args.epochs):
            train(
                model=net,
                dataloader=dataloader_train,
                epoch=epoch,
                optimizer=optimizer,
                logger=logger,
                device=device,
                log_dir=log_dir,
                accumulation_steps=args.accumulation_steps,
            )
            save_checkpoint(net, log_dir / f"ckpt_ep_{epoch + 1}.pth")
            logger.info(f"Saved checkpoint at epoch {epoch + 1}")
            torch.cuda.empty_cache()
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()