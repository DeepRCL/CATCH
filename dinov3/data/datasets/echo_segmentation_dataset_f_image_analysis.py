import pymysql
import numpy as np
import pandas as pd
from rclstream.datasets.private import echo
from rclstream.config import get_database_options
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale  
from pathlib import Path
import torch
import cv2
from collections import defaultdict
import sys

from torchvision import transforms
import random
import torchvision.transforms.functional as T
import json
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

        # self.indices = self.indices[:2]
        self.echo_exam_dataset = echo.EchoExamDataset(exam_ids = self.indices)
        print(f"full dataset size is: {len(self.echo_exam_dataset)}")
        self.visualize_video = visualize_video

        self.empty_mask_count = defaultdict(int)
        self.total_mask_count = defaultdict(int)
        self.next_frame = next_frame
        
        self.dataset_info = {
            "labels_names": [],
            "used_stream_ids": [],
            
        }
        self.json_path = Path(f"data_info_{self.split}.json")
        
    def update_info(self, label_name, stream_id):
            # Store sample metadata
            self.dataset_info["labels_names"].append(label_name)

            # Only keep unique stream IDs
            if stream_id not in self.dataset_info["used_stream_ids"]:
                self.dataset_info["used_stream_ids"].append(stream_id)

            # Write to disk
            with open(self.json_path, "w") as f:
                json.dump(self.dataset_info, f, indent=4)
        
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
                candidate = {
                    "view": view,
                    "stream_id": sample["stream_id"][idx],
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

        ordered_views = ['AP4', 'AP2']
        
        del sample

        for v_name in ordered_views:
            if v_name in view_best:
                # View exists for this patient -> Process normally
                v = view_best[v_name]                
                if v["loss_type"] == 1:

                    self.update_info(v["seg_info"]["labels_name"], v["stream_id"])
                    
                    self.update_info(v["labels_name"], v["stream_id"])




  
if __name__=="__main__":
    multi_view_dataset = MultiViewEchoSegmentationDataset(split='train', req_views=['AP4', 'AP2'])
    start_idx = 13564
    for idx in range(start_idx, len(multi_view_dataset)):
        multi_view_dataset[idx]
        print(f"Processing {idx + 1}/{len(multi_view_dataset)}")
