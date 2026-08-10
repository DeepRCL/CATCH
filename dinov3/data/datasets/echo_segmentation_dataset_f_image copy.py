import pymysql
import numpy as np
import pandas as pd
from rclstream.datasets.private import echo
from rclstream.config import get_database_options
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale  

import torch
import cv2
from collections import defaultdict
import sys

from torchvision import transforms
import random
import torchvision.transforms.functional as T

import os
from torch.utils.data import DataLoader

 
from torch.utils.data import ConcatDataset
 
from torchvision.transforms import InterpolationMode
 
import matplotlib.pyplot as plt 
from collections import defaultdict


def apply_transform(image, mask):
    transforms = []
    # print(f"image, mask, :{image.shape, mask.shape}")
    # Add horizontal flip with 50% probability
    if random.random() < 0.5:
        transforms.append('hflip')

    transforms.append('rotate')

    # Shuffle order
    random.shuffle(transforms)

    # if len(mask.shape) == 2: 
    #     mask = mask.unsqueeze(0).unsqueeze(0) 
    for t in transforms:
        if t == 'hflip':
            image = T.hflip(image)
            mask = T.hflip(mask)
        elif t == 'rotate':
            angle = random.uniform(-15, 15)
            image = T.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = T.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

    # if len(mask.shape) == 4:
    #     mask = mask.squeeze(0).squeeze(0)      # [H,W]
    return image, mask


class MultiViewEchoSegmentationDataset(torch.utils.data.Dataset):
    
    """
    Load metadata using predefined stream IDs and their corresponding exam IDs that contain segmentation labels (LV, LA, RA) in AP2 or AP3 views.

    Use this metadata to retrieve the corresponding exam IDs and initialize the exam dataset.

    Extract data from the exam dataset for the required views only. If multiple views exist per exam, prioritize those with segmentation labels, as they have higher view confidence.

    The data analysis assumption is that each exam definitely has a label in at least one of the views. This follows the same logic as the stream-level (video-level) data analysis used in the pipeline.

    For mask analysis, if a sample has multiple labels within the same frame, we keep all of them. If labels span multiple frames, we use only one frame during training, while all labeled frames are used during testing.
    """
    def __init__(
        self,
        split='train',
        visualize_video =False,
        req_views = [],
        next_frame = 4,
        **kwargs
    ):
        self.segmentation_labels = [
            'LV_vol_d_MOD_A2C_calc',
            'LV_vol_d_MOD_A4C_calc',
            'LV_vol_s_MOD_A2C_calc',
            'LV_vol_s_MOD_A4C_calc',
            'LA_Vol_MOD_A2C_calc',
            'LA_Vol_MOD_A4C_calc',
            'RA_Vol_MOD_A4C_calc',
            'RA_Vol_MOD_A2C_calc',

        ]
        self.segmentation_labels_id_map ={
            'LV_vol_d_MOD_A2C_calc': 0,
            'LV_vol_d_MOD_A4C_calc': 0,
            'LV_vol_s_MOD_A2C_calc': 0,
            'LV_vol_s_MOD_A4C_calc': 0,
            'LA_Vol_MOD_A2C_calc': 1,
            'LA_Vol_MOD_A4C_calc': 1,
            'RA_Vol_MOD_A4C_calc': 2,
            'RA_Vol_MOD_A2C_calc': 2,

            
        }
        # self.req_views = ['AP3', 'AP4', 'AP2']
        self.req_views = req_views
        
        self.split = split
        
        self.df_split = self._load_metadata()
        self.indices = self.df_split['exam_id'].to_list()
        
        if self.split =="val":
            self.indices = self.indices[:100]

        # self.indices = self.indices[:1]
        self.echo_exam_dataset = echo.EchoExamDataset(exam_ids = self.indices)
        self.visualize_video = visualize_video

        self.empty_mask_count = defaultdict(int)
        self.total_mask_count = defaultdict(int)
        self.next_frame = next_frame

        
    def _load_metadata(self):
        
        df_split = {}
        df_split['train']= pd.read_csv("/home/bassant/code/Multi_view_seg_code/data_info/train_files_info.csv", index_col=0)
        df_split['val']= pd.read_csv("/home/bassant/code/Multi_view_seg_code/data_info/val_files_info.csv", index_col=0)
        df_split['test'] =pd.read_csv("/home/bassant/code/Multi_view_seg_code/data_info/test_files_info_patient_level.csv", index_col=0)

        return df_split[self.split]
    

    
    def _get_target(self, sample):
            label_type_list = self.segmentation_labels
  
            mask_ch = np.zeros((3, 256, 256), dtype=np.float32)
            frame_id = None 
            labels_name = [""] * 3
            
            for label_type in label_type_list:
                if label_type not in sample['labels'].keys():
                    continue
                xml_label = sample['labels'][label_type]
                frame_id = max(0, xml_label['frame_num'] - 1)
                assert frame_id >= 0, f"Frame ID is less than 0: {frame_id}"
                
                v = None
                class_name = None
                for k, cls_id in self.segmentation_labels_id_map.items():
                    if k in label_type:
                        v = cls_id
                        class_name = k
                        
                if 'trace' in xml_label:
                    target = xml_label['trace']['mask_cropped']
                    mask = (target > 0).astype(np.float32)
                    mask = cv2.resize(
                                mask,
                                (256, 256),
                                interpolation=cv2.INTER_NEAREST
                            )
                else:
                    mask = np.zeros((256, 256), dtype=np.float32)
                    
                mask_ch[v] = mask   
                labels_name[v] = class_name  
            return mask_ch, frame_id, labels_name
    def __len__(self):
        return len(self.echo_exam_dataset)
    
    
    def __getitem__(self, index):
        max_attempts = 100
        current_index = index
        original_index = index
        
        attempts = 0
        sample = None
        final_dict = {
            "patient_id":[],
            "images":[],
            "masks":[],
            "views":[],
            "stream_ids":[],
            "loss_type":[],
            "spacing": []
        }
        while attempts < max_attempts:
            try:
                sample = self.echo_exam_dataset[current_index]
                break  # Successfully loaded, exit loop
            except Exception as e:
                print(f"Index {current_index} failed: {e}")

            current_index = (current_index + 1) % len(self)
            attempts += 1
            
            # Safety check to avoid infinite loop
            if current_index == original_index and attempts > 1:
                raise RuntimeError(f"All indices are bad or unreachable")
        
        if sample is None:
            raise RuntimeError(f"Could not find valid sample after {max_attempts} attempts")
        patient_id = sample["patient_id"]
        sample_dict = {
            "sample_id": sample["patient_id"],
            "views":[],
            "stream_ids":[],
            "loss_type":[],
            "echo":[],
            "seg_labels_info":[],
            "spacing": []
        }
        view_best = {}

        for idx, stream_data in enumerate(sample["files"]):
            view = stream_data['labels'].get('pred_view', None)
            if view in self.req_views:
                mask_ch, frame_id, labels_name = self._get_target(stream_data)
                has_label = int(mask_ch.sum() > 0)
                dx = sample.get('dX', 0.0)
                dy = sample.get('dY', 0.0)
                new_height, new_width = 256, 256
                video = stream_data['echo']
                orig_height, orig_width = video.shape[-2:]
                scaled_dX = dx * orig_width / 256
                scaled_dY = dy * orig_height / 256
                candidate = {
                    "view": view,
                    "stream_id": sample["stream_id"][idx],
                    "echo": stream_data["echo"],
                    "spacing": torch.tensor([scaled_dX, scaled_dY]),
                    "loss_type": has_label,
                    "seg_info": {
                        "mask": mask_ch,
                        "frame_id": frame_id,
                        "labels_name": labels_name
                    } if has_label else None
                }
                
                # Keep only one per view, prioritize loss_type == 1
                if view not in view_best:
                    view_best[view] = candidate
                elif view_best[view]["loss_type"] == 0 and has_label == 1:
                    view_best[view] = candidate
            else:
                continue

        # for v in view_best.values():
        #     sample_dict["views"].append(v["view"])
        #     sample_dict["stream_ids"].append(v["stream_id"])
        #     sample_dict["echo"].append(v["echo"])
        #     sample_dict["loss_type"].append(v["loss_type"])
        #     sample_dict["spacing"].append(v["spacing"])
        #     if v["loss_type"] == 1:
        #         sample_dict["seg_labels_info"].append(v["seg_info"])
        #     else:
        #         sample_dict["seg_labels_info"].append(None)
                
        
        # modify returned dictonary to be 
        # patient id, video data (N_VIEWS, T, CIN, H,W), SEG INFO (N_VIEWS, T, CIN, H,W), LOSS LABEL LABELS (N_VIEWS, VALUE), STREAM_ID (N_VIEWS, VALUE), Spacing (N_VIEWS, VALUE)
        # final_dict["patient_id"] = sample["patient_id"]
        final_dict["patient_id"] = patient_id
        # Define the exact order of views you expect
        ordered_views = ['AP4', 'AP2']
        
        del sample
        all_images = []
        all_masks = []
        all_stream_ids = []
        all_loss_types = []
        all_spacing = []

        for v_name in ordered_views:
            if v_name in view_best:
                # View exists for this patient -> Process normally
                v = view_best[v_name]
                # echo_data_video = torch.from_numpy(v["echo"]).clone()
                num_frames = v["echo"].shape[0]

                if v["loss_type"] == 0:
                    max_start = max(num_frames - 3, 1)
                    start = torch.randint(0, max_start, (1,)).item()
                    idxs = [start, start + self.next_frame]
                    idxs = [min(i, num_frames - 1) for i in idxs]
                    # idxs = [min(i, len(echo_data_video) - 1) for i in idxs]
                    # image = echo_data_video[idxs]
                    image = torch.from_numpy(v["echo"][idxs])
                    mask = torch.zeros((3, 256, 256))
                    image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)
                    if image.shape[1] == 3:
                        image = rgb_to_grayscale(image, num_output_channels=1)         
                    image = image.float() / 255.0          
                else:
                    # frame = min(v["seg_info"]["frame_id"], len(echo_data_video) - 1)
                    frame = min(v["seg_info"]["frame_id"], num_frames - 1)
                    # image = echo_data_video[frame].unsqueeze(0)
                    image = torch.from_numpy(v["echo"][frame]).unsqueeze(0)
                    mask = torch.from_numpy(v["seg_info"]["mask"]).clone()
                    mask = F.interpolate(mask.unsqueeze(0).float(), (256, 256), mode="nearest").squeeze(0)
                    if random.random() < 0.4 and self.split == "train":
                        image, mask = apply_transform(image, mask)
                    image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)
                    if image.shape[1] == 3:
                        image = rgb_to_grayscale(image, num_output_channels=1)
                    image = image.float() / 255.0   
                    image = torch.cat([image, image], dim=0)

                del v["echo"]  
                all_images.append(image)
                all_masks.append(mask)
                all_stream_ids.append(torch.tensor(v["stream_id"]))
                all_loss_types.append(torch.tensor(v["loss_type"])) # e.g., 0 or 1
                all_spacing.append(torch.tensor(v["spacing"]))
                del v
            else:
                # View is MISSING -> Append dummy zero tensors so shapes match!
                all_images.append(torch.zeros((2, 1, 256, 256))) # Same shape as valid image
                all_masks.append(torch.zeros((3, 256, 256)))
                all_stream_ids.append(torch.tensor(-1))          # Dummy ID
                all_loss_types.append(torch.tensor(0))        
                all_spacing.append(torch.zeros(2))

        final_dict = {
            "patient_id": patient_id,
            "images": torch.stack(all_images),      # Configured exactly to: (3, 2, 1, 256, 256)
            "masks": torch.stack(all_masks),        # Configured exactly to: (3, 3, 256, 256)
            "stream_ids": torch.stack(all_stream_ids),
            "loss_type": torch.stack(all_loss_types),
            "spacing": torch.stack(all_spacing),
            "views": ordered_views
        }        
        return final_dict        
        # all_images = []
        # all_masks = []
        # all_stream_ids = []
        # all_loss_types = []
        # all_spacing = []
        # all_views = []
        # for v in view_best.values():
        #     echo_data_video = torch.from_numpy(v["echo"]).clone()

        #     if v["loss_type"] == 0:
        #         max_start = max(len(echo_data_video) - 3, 1)
        #         start = torch.randint(0, max_start, (1,)).item()

        #         idxs = [start, start + self.next_frame]
        #         idxs = [min(i, len(echo_data_video) - 1) for i in idxs]

        #         image = echo_data_video[idxs]   # [T, C, H, W]
        #         mask = torch.zeros((3, 256, 256))
        #         image = F.interpolate(
        #                 image,
        #                 size=(256, 256),
        #                 mode='bilinear',
        #                 align_corners=False
        #             )
        #         if image.shape[1] ==3:
        #             image = rgb_to_grayscale(image, num_output_channels=1)         
        #         image = image.float() / 255.0          
        #     else:
        #         frame = min(v["seg_info"]["frame_id"], len(echo_data_video) - 1)

        #         image = echo_data_video[frame].unsqueeze(0)

        #         mask = torch.from_numpy(v["seg_info"]["mask"]).clone()
        #         mask = F.interpolate(
        #             mask.unsqueeze(0).float(),
        #             (256, 256),
        #             mode="nearest"
        #         ).squeeze(0)
        #         if random.random() < 0.4  and self.split =="train":
        #             # APPLY TRANSFORMS 
        #             image, mask = apply_transform(image ,mask)
        #         print(image.shape,mask.shape )
        #         # take 2 frames in supervised views as in the un_supervised views 
        #         # resize
        #         image = F.interpolate(
        #                 image,
        #                 size=(256, 256),
        #                 mode='bilinear',
        #                 align_corners=False
        #             )
        #         if image.shape[1] ==3:
        #             image = rgb_to_grayscale(image, num_output_channels=1)
        #         image = image.float() / 255.0   
        #         image = torch.cat([image, image], dim=0)
                


        #     print(image.shape,mask.shape )       
        #     all_images.append(image)
        #     all_masks.append(mask)
        #     all_stream_ids.append(torch.tensor(v["stream_id"]))
        #     all_loss_types.append(torch.tensor(v["loss_type"]))
        #     all_spacing.append(torch.tensor(v["spacing"]))
        #     all_views.append(v["view"])
                
        # final_dict = {
        #     "patient_id": sample["patient_id"],
        #     "images": torch.stack(all_images),      # (N_VIEWS,T=2, C, H, W) list of views 
        #     "masks": torch.stack(all_masks),  # (N_views,1, C, H, W) 1 label per view (zero for no view with no label)
        #     "stream_ids": torch.stack(all_stream_ids),
        #     "loss_type": torch.stack(all_loss_types),
        #     "spacing": torch.stack(all_spacing),
        #     "views": all_views
        # }        
                
        
        # return final_dict



  
if __name__=="__main__":
    multi_view_dataset = MultiViewEchoSegmentationDataset(split='test', req_views=['AP4','AP3','AP2'])
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
            persistent_workers=False
        )
    batch = next(iter(dataloader))

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(key, value.shape)
        else:
            print(key, value)