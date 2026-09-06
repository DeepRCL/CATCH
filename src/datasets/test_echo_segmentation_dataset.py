import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms.functional as T
import matplotlib.pyplot as plt
import pymysql
from torch.utils.data import DataLoader 

from rclstream.config import get_database_options
from rclstream.datasets.private import echo
from rclstream.datasets.public import echonet
import cv2


def ensure_rgb(image):
    if image.ndim == 2:
        return np.repeat(image[None], 3, axis=0)

    if image.shape[0] == 1:
        return np.repeat(image, 3, axis=0)

    return image

def ensure_grey(image):
    if image.ndim == 2:
        return np.repeat(image[None], 1, axis=0)
    if image.shape[0] == 3:
        image = T.rgb_to_grayscale(image, num_output_channels=1)  
        return image
    
class VideoSegTransform(object):
    def __init__(
        self,
        crop_size=256,
        random_horizontal_flip=True,
        perform_aug= False,
    ):
        self.crop_size = crop_size
        self.random_horizontal_flip = random_horizontal_flip
        self.perform_aug = perform_aug

    def __call__(self, video, mask):
        """
        video: Tensor (T, C, H, W)
        mask : (T, C, H, W)
        """

        # Resize video
        video = F.interpolate(
            video,
            size=(self.crop_size, self.crop_size),
            mode="bilinear",
            align_corners=False,
        )

        # Resize mask
        mask = mask.float()
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(0)  # (1, C, H, W)
        
        mask = F.interpolate(
            mask,
            size=(self.crop_size, self.crop_size),
            mode="nearest",
        )
        
        if self.perform_aug and self.random_horizontal_flip and random.random() < 0.5:
                    video = torch.flip(video, dims=[3])  # flip W dimension
                    mask = torch.flip(mask, dims=[3])

        # Normalize video to [0, 1]
        video = video.float()
        min_val = video.min()
        max_val = video.max()

        if max_val > min_val:
            video = (video - min_val) / (max_val - min_val)


        return video, mask.long()


class EchoSegmentationDataset(echo.EchoDataset):
    """
    Dataset for 3-Chamber Echocardiogram Segmentation (LV, LA, RA).
    
    Data Split Behavior:
    --------------------
    - Train / Val: Returns labels for exactly ONE frame per video clip.
    - Test:        Returns labels for ALL annotated frames in the video clip.
    
    Returned Dictionary Format:
    ----------------------------
    - "images":            Tensor of shape (T, C, 256, 256) 
                           [T = frames, C = channels (1 or 3)]
    - "masks":             Tensor of shape (T, 3, 256, 256) 
                           [One-hot encoded: c=0: LV, c=1: LA, c=2: RA]
    - "spacing":          Tensor of shape (T, 2) [Rescaled dX, dY pixel spacing]
    - "pred_view":         String indicating the cardiac view (e.g., "A2C", "A4C")
    - "stream_id":         String ID for dataset tracking
    - "gt_mask_indicies":  Tensor of shape (T,) [Original frame index locations]
    """
    def __init__(
        self,
        split="train",
        res=(256, 256),
        transform=None,
        image_channel_num=3,
        **kwargs,
    ):
        super().__init__()

        self.segmentation_labels_id_map = {
            "LV_vol_d_MOD_A2C_calc": 0,
            "LV_vol_d_MOD_A4C_calc": 0,
            "LV_vol_s_MOD_A2C_calc": 0,
            "LV_vol_s_MOD_A4C_calc": 0,
            "LA_Vol_MOD_A2C_calc": 1,
            "LA_Vol_MOD_A4C_calc": 1,
            "RA_Vol_MOD_A4C_calc": 2,
            "RA_Vol_MOD_A2C_calc": 2,
        }

        self.split = split
    
        self.df_split = self._load_metadata()
        self.indices = self.df_split.index.to_list()
        self.indices = self.indices[:2]
        self.segmentation_labels = self.df_split["label_type"].to_list()

        self.res = res
        self.transform = transform
        self.image_channel_num = image_channel_num
        

    def aggregate_labels(self, df):
        """
        Aggregate all label_type values per cardiac_file_id into a list.
        Each stream_id remains unique in the index.
        """
        df_sampled = df.groupby("label_type", group_keys=False).apply(
            lambda x: x.assign(label_type=x.name).sample(
                n=len(x), random_state=42, replace=False
            )
        )
        df_out = (
            df_sampled.groupby("cardiac_file_id", as_index=False)
            .agg(label_type=("label_type", list), stream_id=("stream_id", "first"))
            .set_index("stream_id")
        )
        return df_out

    def _load_metadata(self):
        conn = pymysql.connect(database="echo_inventory", **get_database_options())
        query = f"SELECT cardiac_file_id, label_type_id FROM label;"
        df_label = pd.read_sql(query, conn)
        query = f"SELECT id, name FROM label_type;"
        df_label_type = pd.read_sql(query, conn)
        df_label["label_type"] = df_label["label_type_id"].map(
            df_label_type.set_index("id")["name"]
        )


        df_segmentation_labels = df_label[df_label['label_type'].isin(self.segmentation_labels_id_map.keys())]
            
        df_file = echo.get_metadata(columns=["id", "processed_file_address"])
        df_files_with_segmentation = pd.merge(
            df_file.reset_index(drop=False),
            df_segmentation_labels,
            left_on="id",
            right_on="cardiac_file_id",
            how="inner",
        )
        
        stream_ids_train = pd.read_csv(
            "data/private_data_files/train_files_info.csv",
            index_col=0
        ).index

        stream_ids_val = pd.read_csv(
            "data/private_data_files/val_files_info.csv",
            index_col=0
        ).index

        stream_ids_test = pd.read_csv(
            "data/private_data_files/test_files_info_patient_level.csv",
            index_col=0
        ).index


        df_train = df_files_with_segmentation[
            df_files_with_segmentation.stream_id.isin(stream_ids_train)
        ]
        df_val = df_files_with_segmentation[
            df_files_with_segmentation.stream_id.isin(stream_ids_val)
        ]
        df_test = df_files_with_segmentation[
            df_files_with_segmentation.stream_id.isin(stream_ids_test)
        ]
        df_val = df_val[:300]    
        df_test = df_test[:6000]
        
        df_split = {
                    "train": self.aggregate_labels(df_train),
                    "val": self.aggregate_labels(df_val),
                    "test": self.aggregate_labels(df_test)
                }

        print(f"Loaded {len(df_split[self.split])} samples for {self.split} split.")
        return df_split[self.split]

    def build_mask(self, xml_label, class_id):
        H, W = self.res
        mask_ch = np.zeros((3, H, W), dtype=np.float32)

        if "trace" not in xml_label:
            return mask_ch

        mask = (
            xml_label["trace"]["mask_cropped"] > 0
        ).astype(np.float32)

        mask = cv2.resize(
            mask,
            (W, H), 
            interpolation=cv2.INTER_NEAREST,
        )

        mask_ch[class_id] = mask

        return mask_ch

    def _get_target(self, sample, index):
        """
        get all labels values for each sample
        """
        label_type_list = self.segmentation_labels[index]
        frame_ids = []
        all_frames_masks = []
        all_frames_labels_name = []
        H, W = self.res
        
        for label_type in label_type_list:
            mask_ch = np.zeros((3, H, W), dtype=np.float32)
            labels_name = [""] * 3
            xml_label = sample['labels'][label_type]
            frame_id = xml_label['frame_num'] - 1
            if frame_id <0:
                frame_id = 0
                frame_ids.append(frame_id) 
                all_frames_masks.append(mask_ch)
                all_frames_labels_name.append(labels_name)
                continue
            
            class_id = self.segmentation_labels_id_map[label_type]      
            mask = self.build_mask(xml_label, class_id)      
            
            frame_ids.append(frame_id) 
            all_frames_masks.append(mask)
            all_frames_labels_name.append(labels_name)
            
            
        return all_frames_masks, frame_ids, all_frames_labels_name   
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample = super().__getitem__(self.indices[index])
        video = sample['echo']
        total_frames = video.shape[0]
        pred_view = sample['labels'].get('pred_view', None)
        stream_id_list = self.indices[index]
        dx = sample.get('dX', 0.0)
        dy = sample.get('dY', 0.0)
        image_stack = []
        mask_stack = []
        clipped_frame_ids = []
        new_height, new_width = self.res
        orig_height, orig_width = video.shape[-2:]
        scaled_dX =  dx * orig_width / new_width
        scaled_dY =  dy * orig_height / new_height
        target_list, frame_id_list, labels_name_list = self._get_target(sample, index)
        
        items = (
            zip(frame_id_list, target_list)
            if self.split == "test"
            else [(frame_id_list[0], target_list[0])]
        )
        
        for frame_id, target_data in items:
            new_frame_id = np.clip(frame_id, 0, total_frames - 1)
            clipped_frame_ids.append(new_frame_id)
            image_frame = video[new_frame_id]
            image_frame = image_frame.astype("float32")

            if self.image_channel_num == 3:
                image_frame = ensure_rgb(image_frame)

            else:
                image_frame = ensure_grey(image_frame)
                
            if image_frame is None or image_frame.size == 0:
                raise ValueError(
                    f"Invalid image at frame {frame_id} at index: {index ,self.indices[index]}"
                )

            
            image_stack.append(image_frame)
            mask_stack.append(target_data)
            
                        
        images = np.stack(image_stack, axis=0)   # (T, C, H, W)
        masks = np.stack(mask_stack, axis=0)

        images = torch.from_numpy(images).float()
        masks = torch.from_numpy(masks).float()
        # (T, C, H, W)
        print(f"images: {images.shape}, masks:{ masks.shape}")


        
        if self.transform is not None:
            images, masks = self.transform(images, masks)
        
        
        
        N_F = masks.shape[0]

        spacing = torch.tensor([scaled_dX, scaled_dY], dtype=torch.float32).repeat(N_F, 1)
        print(images.shape, masks.shape, spacing.shape, N_F)
        return {
            "images": images,
            "masks": masks,
            "spacing": spacing,
            "pred_view": pred_view,
            "labels_name": labels_name_list,
            "stream_id": stream_id_list,
            "gt_mask_indicies": torch.tensor(clipped_frame_ids),
        }


if __name__ =="__main__":
    
    # Instantiate the dataset
    train_dataset = EchoSegmentationDataset(
        split="train",
        res=(256, 256),
        transform=transform,
        image_channel_num=3,  # 3 for RGB, 1 for Grayscale
    )    
    first_sample = train_dataset[0]
    print(torch.unique(first_sample["masks"]))
    print(torch.min(first_sample["images"]), torch.max(first_sample["images"]))
    for key, value in first_sample.items():
        if hasattr(value, "shape"):
            print(f"{key}: shape={value.shape}")
        elif isinstance(value, list):
            print(f"{key}: list of length {len(value)}")

            # inspect elements
            for i, item in enumerate(value[:3]):  # first 3 items
                if hasattr(item, "shape"):
                    print(f"    [{i}] shape={item.shape}")
                else:
                    print(f"    [{i}] type={type(item)}")
        else:
            print(f"{key}: type={type(value)}, value={value}")
            
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    