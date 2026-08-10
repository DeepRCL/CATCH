"""
Multi-view echo segmentation dataset (optimized).

Per-exam output:
    images     (N_VIEWS, N_SLOTS, 1, 256, 256)  float32 in [0, 1]
    masks      (N_VIEWS, N_SLOTS, 3, 256, 256)  float32 {0, 1}
    loss_type  (N_VIEWS,)  int64  1 = view has at least one segmentation label
    stream_ids (N_VIEWS,)  int64  -1 for a missing view
    spacing    (N_VIEWS, 2) float32
    views      list[str]   length N_VIEWS, fixed order

Missing views are zero-filled so every sample has identical shapes and the
default collate_fn works.
"""

import logging
import random
from typing import Any

import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rgb_to_grayscale
from torch.utils.data import DataLoader
from pathlib import Path

from rclstream.datasets.private import echo

log = logging.getLogger(__name__)

# OpenCV spawns its own thread pool; inside DataLoader workers that only causes
# oversubscription and context-switch thrash.
cv2.setNumThreads(0)

IMG_SIZE = 256
N_SLOTS = 2       # max labelled frames kept per view
N_CLASSES = 3     # 0 = LV, 1 = LA, 2 = RA

# label -> class id. Direct lookup, no substring scan.
LABEL_TO_CLASS: dict[str, int] = {
    "LV_vol_d_MOD_A2C_calc": 0,
    "LV_vol_d_MOD_A4C_calc": 0,
    "LV_vol_s_MOD_A2C_calc": 0,
    "LV_vol_s_MOD_A4C_calc": 0,
    "LA_Vol_MOD_A2C_calc": 1,
    "LA_Vol_MOD_A4C_calc": 1,
    "RA_Vol_MOD_A4C_calc": 2,
    "RA_Vol_MOD_A2C_calc": 2,
}
# Deterministic priority order (LV first, then LA, then RA).
LABEL_ORDER: tuple[str, ...] = tuple(LABEL_TO_CLASS)

DEFAULT_CSVS = {
    "train": "/home/bassant/code/Multi_view_seg_code/data_info/train_files_info.csv",
    "val": "/home/bassant/code/Multi_view_seg_code/data_info/val_files_info.csv",
    "test": "/home/bassant/code/Multi_view_seg_code/data_info/test_files_info_patient_level.csv",
}


def apply_transform(image: torch.Tensor, mask: torch.Tensor, max_angle: float = 15.0):
    """Paired flip + rotation. Both tensors must be (N, C, H, W)."""
    if random.random() < 0.5:
        image = T.hflip(image)
        mask = T.hflip(mask)

    angle = random.uniform(-max_angle, max_angle)
    if angle != 0.0:
        image = T.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
        mask = T.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
    return image, mask


class MultiViewEchoSegmentationDataset(torch.utils.data.Dataset):
    """
    Loads exams that carry LV/LA/RA segmentation labels and returns one stream
    per requested view. When an exam has several streams for the same view, the
    labelled one wins (higher view confidence).
    """

    def __init__(
        self,
        split: str = "train",
        req_views: tuple[str, ...] = ("AP4", "AP2"),
        ordered_views: tuple[str, ...] | None = None,
        aug_prob: float = 0.4,
        val_limit: int = 100,
        csv_paths: dict[str, str] | None = None,
        visualize_video: bool = False,
        **kwargs: Any,
    ):
        self.split = split
        self.req_views = tuple(req_views)
        self.req_view_set = frozenset(self.req_views)          # O(1) membership
        self.ordered_views = tuple(ordered_views or self.req_views)
        self.n_views = len(self.ordered_views)
        self.aug_prob = aug_prob if split == "train" else 0.0
        self.visualize_video = visualize_video
        self.csv_paths = csv_paths or DEFAULT_CSVS

        exam_ids = self._load_metadata()
        if split == "val" and val_limit:
            exam_ids = exam_ids[:val_limit]
        self.indices = exam_ids[:40000]

        # echo_exam_dataset = echo.EchoExamDataset()
        # print(f"ds keys: {echo_exam_dataset.keys()}")
        self.echo_exam_dataset = echo.EchoExamDataset(exam_ids = self.indices)
        # self.echo_exam_dataset = echo_exam_dataset[echo_exam_dataset["exam_id"].isin(exam_ids)]
        self._len = len(self.echo_exam_dataset)
        # print(f"loaded ds len: {self._len}")

        # Reusable zero-fill templates for missing views (never mutated in place).
        self._zero_image = torch.zeros(N_SLOTS, 1, IMG_SIZE, IMG_SIZE)
        self._zero_mask = torch.zeros(N_SLOTS, N_CLASSES, IMG_SIZE, IMG_SIZE)
        self._zero_spacing = torch.zeros(2)

        self.dataset_info = {
            "labels_names": [],
            "used_stream_ids": [],
            
        }
        self.json_path = Path(f"/home/bassant/code/Mod_code_view_pos_encod_mv_seg/data_analysis/data_info_{self.split}.json")
    # ------------------------------------------------------------------ setup

    def update_info(self, label_name, stream_id):
            # Store sample metadata
            self.dataset_info["labels_names"].append(label_name)

            # Only keep unique stream IDs
            if stream_id not in self.dataset_info["used_stream_ids"]:
                self.dataset_info["used_stream_ids"].append(stream_id)

            # Write to disk
            with open(self.json_path, "w") as f:
                json.dump(self.dataset_info, f, indent=4)
                
    def _load_metadata(self) -> list:
        """Read only the split we need, and only the column we need."""
        path = self.csv_paths[self.split]
        df = pd.read_csv(path, usecols=["exam_id"])
        return df["exam_id"].tolist()

    def __len__(self) -> int:
        return self._len

    # ----------------------------------------------------------------- target

    def _get_target(self, stream: dict):
        """
        Returns (mask, frame_ids, label_names).

        mask is uint8 (N_SLOTS, N_CLASSES, 256, 256), or None when the stream
        has no usable label -- so the 1.5 MB buffer is never allocated for the
        (common) unlabelled case.
        """
        labels = stream.get("labels") or {}
        mask = None
        frame_ids = np.zeros(N_SLOTS, dtype=np.int64)
        names = ["", ""]
        slot = 0

        for label_type in LABEL_ORDER:
            if slot >= N_SLOTS:
                break
            xml_label = labels.get(label_type)
            if xml_label is None:
                continue

            trace = xml_label.get("trace")
            if trace is None:
                continue  # nothing to segment; an all-zero slot adds no signal

            # Threshold in uint8 (4x less memory + faster resize than float32).
            m = (trace["mask_cropped"] > 0).astype(np.uint8, copy=False)
            if m.shape != (IMG_SIZE, IMG_SIZE):
                m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            if not m.any():
                continue

            if mask is None:
                mask = np.zeros((N_SLOTS, N_CLASSES, IMG_SIZE, IMG_SIZE), dtype=np.uint8)

            mask[slot, LABEL_TO_CLASS[label_type]] = m
            frame_ids[slot] = max(0, xml_label["frame_num"] - 1)
            names[slot] = label_type
            slot += 1

        return mask, frame_ids, names

    # ---------------------------------------------------------------- loading

    def _load_sample(self, index: int, max_attempts: int = 100):
        """Skip forward over corrupt exams instead of killing the epoch."""
        n = self._len
        for attempt in range(max_attempts):
            cur = (index + attempt) % n
            try:
                return self.echo_exam_dataset[cur]
            except Exception as exc:  
                log.warning("exam index %d failed: %s", cur, exc)
        raise RuntimeError(f"No valid sample within {max_attempts} indices of {index}")

    def _select_streams(self, sample: dict) -> dict[str, dict]:
        """
        One stream per requested view, labelled streams preferred.

        Mask decoding is skipped entirely once a view already has a labelled
        candidate, and `echo` is never touched here -- the video is only
        materialized for the streams that actually survive selection.
        """
        best: dict[str, dict] = {}
        stream_ids = sample.get("stream_id", ())

        for idx, stream in enumerate(sample["files"]):
            view = (stream.get("labels") or {}).get("pred_view")
            if view not in self.req_view_set:
                continue

            prev = best.get(view)
            if prev is not None and prev["has_label"]:
                continue  # can't do better than a labelled stream

            mask, frame_ids, names = self._get_target(stream)
            if prev is not None and mask is None:
                continue  # keep the incumbent

            best[view] = {
                "stream": stream,
                "stream_id": stream_ids[idx] if idx < len(stream_ids) else -1,
                "has_label": mask is not None,
                "mask": mask,
                "frame_ids": frame_ids,
                "labels_name": names,
            }
        return best

    @staticmethod
    def _frames_to_tensor(video: np.ndarray, frame_ids: np.ndarray) -> torch.Tensor:
        """
        (T, H, W) or (T, C, H, W) -> (N_SLOTS, 1, 256, 256), float in [0, 1].

        Only the requested frames are pulled out of the video, and RGB is
        collapsed to grayscale *before* resizing (1/3 of the interpolation work).
        """
        frames = np.ascontiguousarray(video[frame_ids])          # (N, H, W[, C])
        t = torch.from_numpy(frames)
        if t.ndim == 3:
            t = t.unsqueeze(1)                                   # (N, 1, H, W)
        if t.shape[1] == 3:
            t = rgb_to_grayscale(t, num_output_channels=1)
        t = t.float().div_(255.0)
        if t.shape[-2:] != (IMG_SIZE, IMG_SIZE):
            t = F.interpolate(t, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        return t

    # --------------------------------------------------------------- __getitem__

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._load_sample(index)
        # print(f"loaded sample: {sample['patient_id']}, {len(sample['files'])} streams")
        patient_id = sample["patient_id"]
        dx = float(sample.get("dX", 0.0))
        dy = float(sample.get("dY", 0.0))

        view_best = self._select_streams(sample)

        images, masks, stream_ids, loss_types, spacings = [], [], [], [], []

        for view in self.ordered_views:
            entry = view_best.get(view)
            if entry is None:
                images.append(self._zero_image)
                masks.append(self._zero_mask)
                stream_ids.append(-1)
                loss_types.append(0)
                spacings.append(self._zero_spacing)
                continue

            video = entry["stream"]["echo"]
            n_frames = video.shape[0]
            orig_h, orig_w = video.shape[-2:]
            
            self.update_info(entry["labels_name"], entry["stream_id"])

            frame_ids = np.minimum(entry["frame_ids"], n_frames - 1)
            image = self._frames_to_tensor(video, frame_ids)

            if entry["mask"] is not None:
                mask = torch.from_numpy(entry["mask"]).float()
            else:
                mask = self._zero_mask.clone()

            if self.aug_prob and random.random() < self.aug_prob:
                image, mask = apply_transform(image, mask)

            images.append(image)
            masks.append(mask)
            stream_ids.append(int(entry["stream_id"]))
            loss_types.append(int(entry["has_label"]))
            spacings.append(
                torch.tensor([dx * orig_w / IMG_SIZE, dy * orig_h / IMG_SIZE])
            )

            entry["stream"].pop("echo", None)  # release the decoded video early

        del sample, view_best
        images = torch.stack(images)   # (V, F, 1, 256, 256)
        masks  = torch.stack(masks)    # (V, F, 3, 256, 256)
        V, F = images.shape[:2]

        return {
            "patient_id": patient_id,
            "images": images,          # (V, F, 1, 256, 256)
            "masks":  masks,            # (V, F, 3, 256, 256)
            "stream_ids": torch.tensor(stream_ids, dtype=torch.int64),  # (V,)
            "loss_type": torch.tensor(loss_types, dtype=torch.int64),   # (V,)
            "spacing": torch.stack(spacings),                           # (V, 2)
            "views": list(self.ordered_views),
        }
    
if __name__=="__main__":
    multi_view_dataset = MultiViewEchoSegmentationDataset(split='train', req_views=['AP4', 'AP2'])
    print(f"train_data_info: {len(multi_view_dataset)}")
    sample = multi_view_dataset[0]
    print(sample)
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(key, value.shape)
            
    
    dataloader = DataLoader(
            multi_view_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=2,
            persistent_workers=False,
            pin_memory = False,
            prefetch_factor=1,
        )
    batch = next(iter(dataloader))

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(key, value.shape)
        else:
            print(key, value)