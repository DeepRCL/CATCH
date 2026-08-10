"""Multi-view echo segmentation model: DINOv3 backbone + cross-view attention + decoder."""

import copy
from functools import partial

import torch
import torch.nn.functional as F
from monai.losses import DiceLoss
from omegaconf import OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint

import sys
REPO_DIR = "/home/bassant/code/asmus_26/"
sys.path.append(REPO_DIR)

from dinov3.eval.segmentation.config import SegmentationConfig
from dinov3.eval.segmentation.models import BackboneLayersSet, _get_backbone_out_indices
from dinov3.eval.setup import load_model_and_context
from dinov3.eval.utils import ModelWithIntermediateLayers

# change it to True when added another loss function to track gradients behaviour
DEBUG_CHECKS = False


def _check(tensor: torch.Tensor, name: str) -> None:
    if DEBUG_CHECKS:
        assert not torch.isnan(tensor).any(), f"{name} contains NaN"


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────────────────
class SmoothDecoder(nn.Module):
    """Upsamples token features 16x and predicts per-pixel segmentation logits."""

    def __init__(self, num_seg_classes: int, token_dim: int = 1024):
        super().__init__()

        def up(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                conv_block(in_ch, out_ch),
            )

        # Attribute names up1..up4 are load-bearing: they match the checkpoint keys.
        self.up1 = up(token_dim, token_dim)
        self.up2 = up(token_dim, 512)
        self.up3 = up(512, 256)
        self.up4 = up(256, 128)

        self.refine = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(128, num_seg_classes, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (N, S, D) sequence or (N, D, h, w) feature map -> (N, C, H, W)."""
        _check(tokens, "tokens")

        if tokens.dim() == 3:
            n, s, d = tokens.shape
            h = w = int(s ** 0.5)
            assert h * w == s, f"Spatial tokens ({s}) must be a perfect square"
            tokens = tokens.permute(0, 2, 1).reshape(n, d, h, w)

        x = self.up1(tokens)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.refine(x)
        seg_logits = self.classifier(x)
        _check(seg_logits, "seg_logits")
        return seg_logits


# ─────────────────────────────────────────────────────────────────────────────
# Cross-view attention
# ─────────────────────────────────────────────────────────────────────────────
class MultiViewCrossAttention(nn.Module):
    """Projects backbone tokens to `dim`, then mixes them across views/space."""

    def __init__(
        self,
        in_dim: int,
        dim: int,
        num_heads: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        ffn_expansion: int = 4,
    ):
        super().__init__()
        self.proj_q = nn.Linear(in_dim, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * ffn_expansion,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        q = self.proj_q(query)
        if self.training:
            return checkpoint(self.predictor, q, use_reentrant=False)
        return self.predictor(q)


# ─────────────────────────────────────────────────────────────────────────────
# Supervised loss
# ─────────────────────────────────────────────────────────────────────────────
class SegLoss(nn.Module):
    """Foreground-averaged Dice, optionally plus cross-entropy."""

    def __init__(self, use_ce: bool = False, ce_weight: float = 1.0):
        super().__init__()
        self.use_ce = use_ce
        self.ce_weight = ce_weight
        self.dice_loss_fn = DiceLoss(
            include_background=True,
            to_onehot_y=False,
            softmax=True,
            reduction="none",
        )
        self.ce_loss_fn = nn.CrossEntropyLoss()

    def forward(self, seg_logits: torch.Tensor, seg_gt: torch.Tensor):
        """seg_logits: (B, C, H, W) with background at index 0. seg_gt: (B, C-1, H, W) one-hot."""
        has_fg = seg_gt.sum(dim=1) > 0                              # (B, H, W)
        bg_ch = (~has_fg).unsqueeze(1).to(seg_gt.dtype)             # (B, 1, H, W)
        seg_gt_full = torch.cat([bg_ch, seg_gt], dim=1)             # (B, C, H, W)

        # Dice per sample per class, averaged over foreground classes actually present.
        label_present = (seg_gt_full.sum(dim=(2, 3)) > 0).float()   # (B, C)
        dice_per_class = self.dice_loss_fn(seg_logits, seg_gt_full).flatten(1)  # (B, C)
        dice_masked = dice_per_class[:, 1:] * label_present[:, 1:]
        num_present = label_present[:, 1:].sum(dim=1).clamp(min=1.0)
        dice_loss = (dice_masked.sum(dim=1) / num_present).mean()

        ce_loss = seg_logits.new_zeros(())
        if self.use_ce:
            gt_indices = torch.argmax(seg_gt, dim=1) + 1
            gt_indices[~has_fg] = 0
            ce_loss = self.ce_loss_fn(seg_logits, gt_indices.long())

        total = dice_loss + self.ce_weight * ce_loss
        return total, {"seg": total.item(), "dice": dice_loss.item(), "ce": ce_loss.item()}


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────
class Model(nn.Module):
    VIEWS = ("AP4", "AP2")
    VIEW_TO_INDEX = {name: i for i, name in enumerate(VIEWS)}

    def __init__(
        self,
        base_config_path: str,
        n_seg_classes: int = 4,
        output_dim: int = 1024,
        input_dim: int = 4096,
        num_views: int = 2,
        grid_size: int = 16,
        use_cross_attention: bool = True,
        use_late_fusion: bool = False,
        use_ce: bool = False,
    ):
        super().__init__()
        if num_views > len(self.VIEWS):
            raise ValueError(f"num_views={num_views} exceeds known views {self.VIEWS}")

        self.encoder = load_backbone(base_config_path)
        self.output_dim = output_dim
        self.num_views = num_views
        self.grid_size = grid_size
        self.num_spatial_tokens = grid_size * grid_size
        self.use_late_fusion = use_late_fusion

        if use_late_fusion:
            # One independent mixer per view; views never attend to each other.
            self.token_mixers = nn.ModuleDict(
                {
                    view: MultiViewCrossAttention(in_dim=input_dim, dim=output_dim)
                    for view in self.VIEWS[:num_views]
                }
            )
        elif use_cross_attention:
            # A single mixer sees all views' tokens jointly (early fusion).
            self.token_mixer = MultiViewCrossAttention(in_dim=input_dim, dim=output_dim)
        else:
            # Keeps the rest of the pipeline shape-compatible without cross-view mixing.
            self.token_mixer = nn.Linear(input_dim, output_dim)

        self.decoder = SmoothDecoder(num_seg_classes=n_seg_classes, token_dim=output_dim)
        self.supervised_loss = SegLoss(use_ce=use_ce)

        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, 1, 1, self.num_spatial_tokens, input_dim)
        )
        self.view_embed = nn.Parameter(torch.zeros(1, num_views, 1, 1, input_dim))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.view_embed, std=0.02)

    def _embed_tokens(
        self, images: torch.Tensor, b: int, n_v: int, t: int, view_indices: list[int] | None = None
    ) -> tuple[torch.Tensor, int, int]:
        """(B*N_V*T, C, H, W) -> (B, N_V, T, h*w, D) with spatial + view embeddings added."""
        view_embeddings = self.encoder(images)[0]              # (B*N_V*T, D, h, w)
        _, dim, h, w = view_embeddings.shape
        assert h * w == self.num_spatial_tokens, (
            f"Backbone produced a {h}x{w} grid but spatial_pos_embed expects "
            f"{self.grid_size}x{self.grid_size}"
        )

        tokens = view_embeddings.flatten(2).permute(0, 2, 1)   # (B*N_V*T, h*w, D)
        tokens = tokens.view(b, n_v, t, h * w, dim)

        if view_indices is None:
            view_indices = list(range(n_v))
        tokens = tokens + self.spatial_pos_embed + self.view_embed[:, view_indices]
        return tokens, h, w

    def _mix_joint(self, tokens: torch.Tensor, b: int, n_v: int, t: int) -> torch.Tensor:
        """Early fusion: all views' tokens go through one mixer. -> (B, N_V, T, h*w, D)."""
        s, dim = tokens.shape[3], tokens.shape[4]
        flat = tokens.permute(0, 2, 1, 3, 4).reshape(b * t, n_v * s, dim)
        mixed = self.token_mixer(flat)                          # (B*T, N_V*S, D_out)
        mixed = mixed.view(b, t, n_v, s, self.output_dim)
        return mixed.permute(0, 2, 1, 3, 4)                     # (B, N_V, T, S, D_out)

    def _mix_late(self, tokens: torch.Tensor, b: int, n_v: int, t: int) -> torch.Tensor:
        """Late fusion: each view gets its own mixer. -> (B, N_V, T, h*w, D)."""
        s, dim = tokens.shape[3], tokens.shape[4]
        per_view = []
        for i, view in enumerate(self.VIEWS[:n_v]):
            view_tokens = tokens[:, i].reshape(b * t, s, dim)   # (B*T, S, D)
            mixed = self.token_mixers[view](view_tokens)        # (B*T, S, D_out)
            per_view.append(mixed.view(b, t, s, self.output_dim))
        return torch.stack(per_view, dim=1)                     # (B, N_V, T, S, D_out)

    def _encode(self, images: torch.Tensor, b: int, n_v: int, t: int) -> tuple[torch.Tensor, int, int]:
        tokens, h, w = self._embed_tokens(images, b, n_v, t)
        mix = self._mix_late if self.use_late_fusion else self._mix_joint
        return mix(tokens, b, n_v, t), h, w

    def forward(self, batch_data: dict):
        images = batch_data["images"]
        masks = batch_data["masks"]
        loss_type = batch_data["loss_type"]

        b, n_v, t, c_in, height, width = images.shape
        gt_labels = masks.view(b * n_v, *masks.shape[2:])

        mixed, h, w = self._encode(images.view(b * n_v * t, c_in, height, width), b, n_v, t)

        # (B, N_V, T, h*w, D) -> (B*N_V*T, D, h, w)
        feats = mixed.view(b, n_v, t, h, w, self.output_dim)
        feats = feats.permute(0, 1, 2, 5, 3, 4).reshape(b * n_v * t, self.output_dim, h, w)
        _check(feats, "attn_out")

        seg_logits = self.decoder(feats)
        _, c_out, out_h, out_w = seg_logits.shape

        # Select, per (sample, view), the single annotated frame indexed by loss_type.
        seg_logits = seg_logits.view(b, n_v, t, c_out, out_h, out_w)
        gather_idx = loss_type.view(b, n_v, 1, 1, 1, 1).expand(-1, -1, -1, c_out, out_h, out_w)
        output = torch.gather(seg_logits, dim=2, index=gather_idx).squeeze(2)
        output = output.reshape(b * n_v, c_out, out_h, out_w)

        total_loss, s_log = self.supervised_loss(output, gt_labels)
        log = {
            "loss/total": total_loss.item(),
            "loss/supervised": total_loss.item(),
            "loss/seg": s_log["seg"],
            "loss/seg_dice": s_log["dice"],
            "loss/seg_ce": s_log["ce"],
        }
        return total_loss, log, output

    @torch.no_grad()
    def test(self, images: torch.Tensor, view_name: str) -> torch.Tensor:
        """images: (T, C, H, W) for a single view -> seg logits (T, n_classes, H', W')."""
        if isinstance(view_name, (list, tuple)):
            view_name = view_name[0]
        view_idx = self.VIEW_TO_INDEX[view_name]

        t = images.shape[0]
        b = n_v = 1
        tokens, h, w = self._embed_tokens(images, b, n_v, t, view_indices=[view_idx])

        if self.use_late_fusion:
            mixed = self.token_mixers[view_name](tokens.reshape(t, h * w, -1))
        else:
            # Single-view inference: the joint mixer sees only this view's tokens.
            mixed = self.token_mixer(tokens.reshape(t, h * w, -1))

        feats = mixed.view(t, h, w, self.output_dim).permute(0, 3, 1, 2)
        return self.decoder(feats)


# ─────────────────────────────────────────────────────────────────────────────
# Backbone / checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_backbone(
    base_config_path: str,
    n_intermediate_layers: tuple = (40,),
    n_blocks_to_duplicate: int = 1,
    verbose: bool = False,
) -> nn.Module:
    base_config = OmegaConf.load(base_config_path)
    dataclass_config: SegmentationConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(SegmentationConfig), base_config)
    )

    backbone_model, _ = load_model_and_context(dataclass_config.model, output_dir="")
    backbone_indices = _get_backbone_out_indices(backbone_model, BackboneLayersSet.LAST)
    if verbose:
        print(f"backbone_indices_to_use: {backbone_indices}")

    backbone_model = ModelWithIntermediateLayers(
        backbone_model,
        n=list(n_intermediate_layers),
        autocast_ctx=partial(
            torch.autocast, device_type="cuda", enabled=True, dtype=torch.float32
        ),
        reshape=True,
        return_class_token=False,
    )

    # Expand the last transformer blocks for adaptation to the segmentation task
    blocks = backbone_model.feature_model.blocks
    blocks.extend([copy.deepcopy(b) for b in blocks[-n_blocks_to_duplicate:]])
    print(f"Backbone depth after duplication: {len(blocks)}")

    backbone_model.requires_grad_(False)
    backbone_model.eval()
    return backbone_model


def strip_prefix(state_dict: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def truncate_dim0(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Trim a checkpoint tensor's output dim to match the current head."""
    return tensor[: target.shape[0]]


def load_compatible(module: nn.Module, state: dict, transforms: dict | None = None, name: str = "") -> None:
    """Load only the keys whose shapes match, applying optional per-key transforms."""
    transforms = transforms or {}
    target = module.state_dict()
    to_load, skipped = {}, []

    for key, value in state.items():
        if key not in target:
            skipped.append((key, "missing in module"))
            continue
        if key in transforms:
            value = transforms[key](value, target[key])
        if value.shape != target[key].shape:
            skipped.append((key, f"shape {tuple(value.shape)} != {tuple(target[key].shape)}"))
            continue
        to_load[key] = value

    module.load_state_dict(to_load, strict=False)
    print(f"[{name}] loaded {len(to_load)}/{len(target)} tensors; skipped {len(skipped)}")
    for key, reason in skipped:
        print(f"  skipped {key}: {reason}")


def load_pretrained(net: Model, path: str) -> Model:
    state = torch.load(path, map_location="cpu")["state_dict"]
    load_compatible(net.encoder, strip_prefix(state, "backbone."), name="encoder")
    load_compatible(
        net.decoder,
        strip_prefix(state, "decoder."),
        transforms={"classifier.weight": truncate_dim0, "classifier.bias": truncate_dim0},
        name="decoder",
    )
    return net