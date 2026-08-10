import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from rclstream.datasets.public import camus
from rclstream.datasets.public import echonet

CAMUS_SPLIT_FILES = {
    "train": "data/camus_files/subgroup_training.txt",
    "val": "data/camus_files/subgroup_validation.txt",
    "test": "data/camus_files/subgroup_testing.txt",
}

IMG_SIZE = (256, 256)


def calc_ef_from_seg(pred_mask_1, pred_mask_2):
    volume_1 = torch.sum(pred_mask_1, dim=(1, 2))
    volume_2 = torch.sum(pred_mask_2, dim=(1, 2))

    stacked = torch.stack([volume_1, volume_2], dim=1)  # (B, 2)
    EDV = stacked.max(dim=1).values                     # (B,)
    ESV = stacked.min(dim=1).values                     # (B,)

    pred_ef = 100 * (EDV - ESV) / EDV                   # (B,)
    return pred_ef, EDV, ESV


def _camus_filtered_indices(split, view_list):
    """Indices of CAMUS metadata rows matching the given split and cardiac views."""
    if split not in CAMUS_SPLIT_FILES:
        raise ValueError(f"Unknown split: {split!r}")
    patients = np.loadtxt(CAMUS_SPLIT_FILES[split], dtype=str)

    metadata = camus.get_metadata()
    return metadata[
        (metadata["cardiac_view"].isin(view_list))
        & (
            metadata["file_path"]
            .apply(lambda x: os.path.basename(os.path.dirname(x)))
            .isin(patients)
        )
    ].index.tolist()


def _resize_frames(image, mask, indices):
    """Resize the selected frames of a (B, H, W) video and its mask to IMG_SIZE."""
    resized_images, resized_masks = [], []
    for idx in indices:
        resized_images.append(
            torch.from_numpy(cv2.resize(image[idx], IMG_SIZE, interpolation=cv2.INTER_LINEAR))
        )
        resized_masks.append(
            torch.from_numpy(
                cv2.resize(mask[idx].numpy(), IMG_SIZE, interpolation=cv2.INTER_NEAREST)
            )
        )
    return resized_images, resized_masks


class camus_test(camus.CAMUSDataset):
    """CAMUS clip from ED to ES (padded to 16 frames) with one-hot masks and EF targets."""

    def __init__(self, split="train", transforms=None):
        super().__init__()
        self.transforms = transforms
        self.view_list = ["AP4", "AP2"]
        self.num_classes = 4
        self.split = split
        self.filtered_indices = _camus_filtered_indices(split, self.view_list)

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, index):
        sample = super().__getitem__(self.filtered_indices[index])

        # Convert to float32 and normalize to [0,1]
        image = sample["video"].astype("float32") / 255.0
        mask = torch.from_numpy(sample["mask"]).long()

        es_index = int(sample["end_systolic_frame_index"])
        ed_index = int(sample["end_diastolic_frame_index"])
        indices = list(range(min(es_index, ed_index), max(es_index, ed_index) + 1))

        ef = torch.tensor(sample["ejection_fraction"], dtype=torch.float32)

        resized_images, resized_masks = _resize_frames(image, mask, indices)

        # Pad if fewer than 16 frames
        while len(resized_images) < 16:
            resized_images.append(resized_images[-1].clone())
            resized_masks.append(resized_masks[-1].clone())

        image = torch.stack(resized_images).unsqueeze(1)  # (B, 1, 256, 256)
        mask_resized = torch.stack(resized_masks).long()

        # Convert to one-hot: (B, H, W, C) -> (B, C, H, W)
        target = F.one_hot(mask_resized, num_classes=self.num_classes)
        target = target.permute(0, 3, 1, 2).float()

        # swap channel 2 and 3, then zero out myo channel
        target[:, [2, 3]] = target[:, [3, 2]]
        target[:, 3, :, :] = 0

        ef_seg, edv, esv = calc_ef_from_seg(target[0, 1:2], target[-1, 1:2])

        if self.transforms is not None:
            image = self.transforms(image)

        return {
            "images": image,
            "masks": target,
            "pred_view": sample["cardiac_view"],
            "edv": edv,
            "esv": esv,
            "ef_numeric": ef,
            "ef_visual": ef_seg,
        }


class EchoNet_Dynamic(echonet.EchoNetDataset):
    """
    Returns resized AP4 video frames (values normalized 0-1), 2 frames per sample (ED, ES),
    with corresponding LV labels.

    Output shapes:
    - Video frames: (T, C=1, H, W)
    - Masks: (T, C=2, H, W)
    """

    def __init__(self, split="test"):
        super().__init__()
        self.split = split
        metadata = echonet.get_metadata()
        self.filtered_indices = metadata[metadata["split"] == self.split].index.tolist()

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, index):
        sample = super().__getitem__(self.filtered_indices[index])
        video = sample["video"]  # (T, C, H, W)
        ed_index, es_index = sample["end_diastolic_frame_index"], sample["end_systolic_frame_index"]

        masks = np.stack(
            [
                cv2.resize(sample["end_diastolic_mask"], IMG_SIZE, interpolation=cv2.INTER_NEAREST),
                cv2.resize(sample["end_systolic_mask"], IMG_SIZE, interpolation=cv2.INTER_NEAREST),
            ],
            axis=0,
        )
        masks = torch.from_numpy(masks).long()
        masks = F.one_hot(masks, num_classes=2).permute(0, 3, 1, 2).float()  # (2, 2, 256, 256)

        selected_video = []
        for frame_index in [ed_index, es_index]:
            frame = video[frame_index]
            if frame.ndim == 3 and frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0))  # (3, H, W) -> (H, W, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_LINEAR) / 255.0
            selected_video.append(np.expand_dims(frame, axis=0))

        selected_video = torch.from_numpy(np.stack(selected_video, axis=0)).float()

        edv = torch.tensor(sample["end_diastolic_volume"], dtype=torch.float32)
        esv = torch.tensor(sample["end_systolic_volume"], dtype=torch.float32)
        ef = torch.tensor(sample["ejection_fraction"], dtype=torch.float32)
        ef_seg, _, _ = calc_ef_from_seg(masks[0, 1:2], masks[-1, 1:2])

        return {
            "images": selected_video,
            "masks": masks,
            "pred_view": "AP4",
            "edv": edv,
            "esv": esv,
            "ef_numeric": ef,
            "ef_visual": ef_seg,
        }


class Video_Camus(camus.CAMUSDataset):
    """Full CAMUS video with one-hot masks and pixel spacing rescaled to IMG_SIZE."""

    LABELS_NAME = ["LV", "LA", "MYO"]

    def __init__(self, split="train", transforms=None):
        super().__init__()
        self.transforms = transforms
        self.view_list = ["AP4", "AP2"]
        self.num_classes = 4
        self.split = split
        self.filtered_indices = _camus_filtered_indices(split, self.view_list)

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, index):
        sample = super().__getitem__(self.filtered_indices[index])

        # Convert to float32 and normalize to [0,1]
        image = sample["video"].astype("float32") / 255.0
        mask = torch.from_numpy(sample["mask"]).long()

        B, H, W = image.shape
        resized_images, resized_masks = _resize_frames(image, mask, range(B))

        image = torch.stack(resized_images).unsqueeze(1)  # (B, 1, 256, 256)
        mask_resized = torch.stack(resized_masks).long()

        # Convert to one-hot: (B, H, W, C) -> (B, C, H, W)
        target = F.one_hot(mask_resized, num_classes=self.num_classes)
        target = target.permute(0, 3, 1, 2).float()

        # swap channel 2 and 3
        target[:, [2, 3]] = target[:, [3, 2]]

        if self.transforms is not None:
            image = self.transforms(image)

        new_width, new_height = IMG_SIZE
        scaled_dx = sample["spacing"][0] * W / new_width
        scaled_dy = sample["spacing"][1] * H / new_height

        return {
            "images": image,
            "masks": target,
            "spacing": torch.tensor([scaled_dx, scaled_dy]),
            "pred_view": sample["cardiac_view"],
            "labels_name": self.LABELS_NAME,
            "stream_id": torch.tensor(-1),
        }



if __name__ == "__main__":
    camus_test_dataset = camus_test()
    sample = camus_test_dataset[0]
    print(sample)