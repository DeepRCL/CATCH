import random
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rgb_to_grayscale

from rclstream.datasets.private import echo

DATA_INFO_DIR = "/home/bassant/code/Multi_view_seg_code/data_info"
IMG_SIZE = 256

SEGMENTATION_LABEL_ID_MAP = {
        'LV_vol_d_MOD_A2C_calc': 0,
        'LV_vol_d_MOD_A4C_calc': 0,
        'LV_vol_s_MOD_A2C_calc': 0,
        'LV_vol_s_MOD_A4C_calc': 0,
        'LA_Vol_MOD_A2C_calc': 1,
        'LA_Vol_MOD_A4C_calc': 1,
        'RA_Vol_MOD_A4C_calc': 2,
        'RA_Vol_MOD_A2C_calc': 2,
    }



def apply_transform(image, mask):
    """Apply hflip (50% chance) and a random small rotation in shuffled order."""
    ops = []
    if random.random() < 0.5:
        ops.append('hflip')
    ops.append('rotate')
    random.shuffle(ops)

    for op in ops:
        if op == 'hflip':
            image = T.hflip(image)
            mask = T.hflip(mask)
        elif op == 'rotate':
            angle = random.uniform(-15, 15)
            image = T.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = T.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

    return image, mask


class MultiViewEchoSegmentationDataset(torch.utils.data.Dataset):
    """
    Load metadata using predefined stream IDs and their corresponding exam IDs that contain
    segmentation labels (LV, LA, RA) in AP2 or AP3 views.

    Use this metadata to retrieve the corresponding exam IDs and initialize the exam dataset.

    Extract data from the exam dataset for the required views only. If multiple views exist per
    exam, prioritize those with segmentation labels, as they have higher view confidence.

    The data analysis assumption is that each exam definitely has a label in at least one of the
    views. This follows the same logic as the stream-level (video-level) data analysis used in the
    pipeline.

    For mask analysis, if a sample has multiple labels within the same frame, we keep all of them.
    If labels span multiple frames, we use only one frame during training, while all labeled frames
    are used during testing.
    """

    def __init__(
        self,
        split='train',
        visualize_video=False,
        req_views=(),
        next_frame=4,
        max_exams=40000,
        **kwargs,
    ):
        self.segmentation_labels = list(SEGMENTATION_LABEL_ID_MAP)
        self.segmentation_labels_id_map = SEGMENTATION_LABEL_ID_MAP
        self.req_views = list(req_views)
        self.split = split
        self.visualize_video = visualize_video
        self.next_frame = next_frame

        self.df_split = self._load_metadata()
        self.indices = self.df_split['exam_id'].to_list()
        if self.split == "val":
            self.indices = self.indices[:100]
        
        self.indices = self.indices[:max_exams]

        self.echo_exam_dataset = echo.EchoExamDataset(exam_ids=self.indices)

        self.empty_mask_count = defaultdict(int)
        self.total_mask_count = defaultdict(int)
        self.ORDERED_VIEWS = ['AP4', 'AP2']

    def _load_metadata(self):
        files = {
            'train': f"{DATA_INFO_DIR}/train_files_info.csv",
            'val': f"{DATA_INFO_DIR}/val_files_info.csv",
            'test': f"{DATA_INFO_DIR}/test_files_info_patient_level.csv",
        }
        return pd.read_csv(files[self.split], index_col=0)

    def _get_target(self, sample):
        """Return (mask_ch [3, H, W], frame_id, labels_name)."""
        mask_ch = np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        frame_id = None
        labels_name = [""] * 3

        for label_type in self.segmentation_labels:
            if label_type not in sample['labels']:
                continue
            xml_label = sample['labels'][label_type]
            frame_id = max(0, xml_label['frame_num'] - 1)

            class_id = None
            class_name = None
            for k, cls_id in self.segmentation_labels_id_map.items():
                if k in label_type:
                    class_id = cls_id
                    class_name = k

            if 'trace' in xml_label:
                target = xml_label['trace']['mask_cropped']
                mask = (target > 0).astype(np.float32)
                mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            else:
                mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

            mask_ch[class_id] = mask
            labels_name[class_id] = class_name

        return mask_ch, frame_id, labels_name

    def __len__(self):
        return len(self.echo_exam_dataset)

    def _load_sample(self, index, max_attempts=100):
        """Return the first loadable sample at or after `index`, wrapping around."""
        current_index = index
        for attempt in range(max_attempts):
            try:
                return self.echo_exam_dataset[current_index]
            except Exception as e:
                print(f"Index {current_index} failed: {e}")

            current_index = (current_index + 1) % len(self)
            if current_index == index and attempt > 0:
                raise RuntimeError("All indices are bad or unreachable")

        raise RuntimeError(f"Could not find valid sample after {max_attempts} attempts")

    def __getitem__(self, index):
        sample = self._load_sample(index)
        patient_id = sample["patient_id"]

        # Keep one stream per view, prioritizing streams that carry a segmentation label.
        view_best = {}
        for idx, stream_data in enumerate(sample["files"]):
            view = stream_data['labels'].get('pred_view', None)
            if view not in self.req_views:
                continue

            mask_ch, frame_id, labels_name = self._get_target(stream_data)
            has_label = int(mask_ch.sum() > 0)

            orig_height, orig_width = stream_data['echo'].shape[-2:]
            scaled_dX = sample.get('dX', 0.0) * orig_width / IMG_SIZE
            scaled_dY = sample.get('dY', 0.0) * orig_height / IMG_SIZE

            candidate = {
                "view": view,
                "stream_id": sample["stream_id"][idx],
                "echo": stream_data["echo"],
                "spacing": torch.tensor([scaled_dX, scaled_dY]),
                "loss_type": has_label,
                "seg_info": {
                    "mask": mask_ch,
                    "frame_id": frame_id,
                    "labels_name": labels_name,
                } if has_label else None,
            }
            # Add view data only if it has not already been added, or if it has a label and the previously added data does not.            
            if view not in view_best or (view_best[view]["loss_type"] == 0 and has_label == 1):
                view_best[view] = candidate

        del sample

        all_images = []
        all_masks = []
        all_stream_ids = []
        all_loss_types = []
        all_spacing = []

        for v_name in self.ORDERED_VIEWS:
            if v_name not in view_best:
                # View is missing -> append dummy zero tensors so shapes match.
                all_images.append(torch.zeros((2, 1, IMG_SIZE, IMG_SIZE)))
                all_masks.append(torch.zeros((3, IMG_SIZE, IMG_SIZE)))
                all_stream_ids.append(torch.tensor(-1))
                all_loss_types.append(torch.tensor(0))
                all_spacing.append(torch.zeros(2))
                continue

            v = view_best[v_name]
            num_frames = v["echo"].shape[0]

            if v["loss_type"] == 0:
                max_start = max(num_frames - 3, 1)
                start = torch.randint(0, max_start, (1,)).item()
                idxs = [min(i, num_frames - 1) for i in (start, start + self.next_frame)]
                image = torch.from_numpy(v["echo"][idxs])
                mask = torch.zeros((3, IMG_SIZE, IMG_SIZE))
                image = F.interpolate(
                    image, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False
                )
                if image.shape[1] == 3:
                    image = rgb_to_grayscale(image, num_output_channels=1)
                image = image.float() / 255.0
            else:
                frame = min(v["seg_info"]["frame_id"], num_frames - 1)
                image = torch.from_numpy(v["echo"][frame]).unsqueeze(0)
                mask = torch.from_numpy(v["seg_info"]["mask"]).clone()
                mask = F.interpolate(
                    mask.unsqueeze(0).float(), (IMG_SIZE, IMG_SIZE), mode="nearest"
                ).squeeze(0)
                if self.split == "train" and random.random() < 0.4:
                    image, mask = apply_transform(image, mask)
                image = F.interpolate(
                    image, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False
                )
                if image.shape[1] == 3:
                    image = rgb_to_grayscale(image, num_output_channels=1)
                image = image.float() / 255.0
                image = torch.cat([image, image], dim=0)

            del v["echo"]
            all_images.append(image)
            all_masks.append(mask)
            all_stream_ids.append(torch.tensor(v["stream_id"]))
            all_loss_types.append(torch.tensor(v["loss_type"]))
            all_spacing.append(v["spacing"])
            del v

        return {
            "patient_id": patient_id,
            "images": torch.stack(all_images),        # (V, 2, 1, 256, 256)
            "masks": torch.stack(all_masks),          # (V, 3, 256, 256)
            "stream_ids": torch.stack(all_stream_ids),
            "loss_type": torch.stack(all_loss_types),
            "spacing": torch.stack(all_spacing),
            "views": self.ORDERED_VIEWS,
        }


if __name__ == "__main__":
    multi_view_dataset = MultiViewEchoSegmentationDataset(split='train', req_views=['AP4', 'AP2'])
    sample = multi_view_dataset[0]
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(key, value.shape)
            if key =="loss_type" or key =="stream_ids":
                print(value)
        else:
            print(key, value)

    dataloader = DataLoader(
        multi_view_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        persistent_workers=False,
    )
    batch = next(iter(dataloader))

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(key, value.shape)
        else:
            print(key, value)