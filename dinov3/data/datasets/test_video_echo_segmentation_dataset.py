import pymysql
import numpy as np
import pandas as pd
from rclstream.datasets.private import echo
from rclstream.config import get_database_options
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
# import torch.nn.functional as F
import torch
import cv2
from collections import defaultdict
import sys
REPO_DIR = "/data/project/users/bassant/code/ablation_multi_view/multi_view_seg_proj/"
sys.path.append(REPO_DIR)
from torchvision import transforms
import random
import torchvision.transforms.functional as T
# import imgaug.augmenters as iaa
from dinov3.data.datasets.rwma_dataset import PrivateRWMADataset
import numpy as np
import pandas as pd
from rclstream.datasets.private import echo
import torch
import cv2
from collections import defaultdict
import random
import torchvision.transforms.functional as T 
from torchvision.transforms import InterpolationMode
from collections import defaultdict
import ast
from collections import Counter, defaultdict
import os
import torch.nn.functional as F


import ast
import pandas as pd
from collections import Counter, defaultdict


from torch.utils.data import ConcatDataset

from torchvision.transforms import InterpolationMode

import matplotlib.pyplot as plt 
from collections import defaultdict
import json
import os
import torch

JSON_PATH = "test_data.json"

def save_sample_to_json(sample_stream_id, scaled_dX, scaled_dY, path=JSON_PATH):
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[sample_stream_id] = [scaled_dX, scaled_dY]  # int key, list value

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        
        
class VideoSegTransform(object):
    def __init__(
        self,
        crop_size=256,
        random_horizontal_flip=True,
        normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ):
        self.crop_size = crop_size
        self.random_horizontal_flip = random_horizontal_flip

        self.mean = (
            torch.tensor(normalize[0], dtype=torch.float32).view(3, 1, 1, 1) * 255.0
        )
        self.std = (
            torch.tensor(normalize[1], dtype=torch.float32).view(3, 1, 1, 1) * 255.0
        )

    def __call__(self, video):
        """
        video: Tensor (C, T, H, W)
        """

        # Resize
        C, T, H, W = video.shape

        # Resize video
        video = F.interpolate(
            video,
            size=(self.crop_size, self.crop_size),
            mode="bilinear",
            align_corners=False,
        )


        # Normalize video

        if video.dtype == torch.uint8:
            video = video.float()

        video = (video - self.mean) / self.std

        return video


class EchoSegmentationDataset(echo.EchoDataset):
    def __init__(
        self,
        split='train',
        visualize_video =False,
        transform = None,
        **kwargs
    ):
        super().__init__()

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
            'LV_vol_d_MOD_A2C_calc': 1,
            'LV_vol_d_MOD_A4C_calc': 1,
            'LV_vol_s_MOD_A2C_calc': 1,
            'LV_vol_s_MOD_A4C_calc': 1,
            'LA_Vol_MOD_A2C_calc': 2,
            'LA_Vol_MOD_A4C_calc': 2,
            'RA_Vol_MOD_A4C_calc': 3,
            'RA_Vol_MOD_A2C_calc': 3,

            
        }
        
        self.split = split
        self.df_split = self._load_metadata()
        self.indices = self.df_split.index.to_list()
        
        self.segmentation_labels = self.df_split['label_type'].to_list()
        self.visualize_video = visualize_video

        self.empty_mask_count = defaultdict(int)
        self.total_mask_count = defaultdict(int)

        self.transform = transform
        
    def aggregate_labels(self, df):
        """get
        Aggregate all label_type values per cardiac_file_id into a list.
        Each stream_id remains unique in the index.
        """
        df_sampled = (
            df
            .groupby("label_type", group_keys=False)
            .apply(lambda x: x.assign(label_type=x.name).sample(
                n=len(x),
                random_state=42,
                replace=False
            ))
        )
        df_out = (
            df_sampled
            .groupby('cardiac_file_id', as_index=False)
            .agg(
                label_type=('label_type', list),
                stream_id=('stream_id', 'first'),
                exam_id=('exam_id', 'first')
            )
        .set_index('stream_id')
        
        )
        return df_out
    def _load_metadata(self):
        conn = pymysql.connect(database="echo_inventory", **get_database_options())
        query = f"SELECT cardiac_file_id, label_type_id FROM label;"
        df_label = pd.read_sql(query, conn)
        query = f"SELECT id, name FROM label_type;"
        df_label_type = pd.read_sql(query, conn)
        df_label['label_type'] = df_label['label_type_id'].map(df_label_type.set_index('id')['name'])
        

        df_segmentation_labels = df_label[df_label['label_type'].isin(self.segmentation_labels)]
        df_file = echo.get_metadata(columns=['id', 'processed_file_address', 'exam_id'])
        df_files_with_segmentation = pd.merge(df_file.reset_index(drop=False), df_segmentation_labels, left_on='id', right_on='cardiac_file_id', how='inner')     
        print(f"before filtering: {len(df_files_with_segmentation)}")

        # stream_ids_train = torch.load("/home/bassant/code/Multi_view_seg_code/data_info/stream_ids_train.pt",  weights_only=False)
        # stream_ids_val = torch.load("/home/bassant/code/Multi_view_seg_code/data_info/stream_ids_val.pt",  weights_only=False)
        stream_ids_test = torch.load("/home/bassant/code/Multi_view_seg_code/data_info/updated_stream_ids_test_patient_level.pt",  weights_only=False)

        # df_train = df_files_with_segmentation[df_files_with_segmentation.stream_id.isin(stream_ids_train)]
        # df_val = df_files_with_segmentation[df_files_with_segmentation.stream_id.isin(stream_ids_val)]
        
        df_test = df_files_with_segmentation[df_files_with_segmentation.stream_id.isin(stream_ids_test)]
        df_test = df_test[:3000]
        # df_test = df_test[:2]
        print(f"new test data size: {len(df_test)}")


        # df_train_agg = self.aggregate_labels(df_train)
        # df_val_agg = self.aggregate_labels(df_val)
        df_test_agg = self.aggregate_labels(df_test)
        df_split = {}
        # df_split['train']= df_train_agg
        # df_split['val']= df_val_agg
        df_split['test'] =df_test_agg
        # all_splits = {}
        # for split in ['train', 'val', 'test']:
        #     unique_labels = (
        #         df_split[split]['label_type']
        #         .dropna()
        #         .explode()   # turns lists into rows
        #         .value_counts()
        #     )

        #     label_df = unique_labels.reset_index().rename(
        #             columns={'index': 'label_name', 'label_type': 'count'}
        #         )

        print(f"len of {self.split} echo data: {len(df_split[self.split])}")
        return df_split[self.split]

 
    def _get_target(self, sample, index):
        label_type_list = self.segmentation_labels[index]
        frame_ids = []
        all_frames_masks = []
        all_frames_labels_name = []
            
        if self.split =="test":
            # collect frame ids
            for label_type in label_type_list:
                mask_ch = np.zeros((4, 256, 256), dtype=np.float32)
                labels_name = [""] * 4
                xml_label = sample['labels'][label_type]
                frame_id = xml_label['frame_num'] - 1
                if frame_id <0:
                    frame_id = 0
                    frame_ids.append(frame_id) 
                    all_frames_masks.append(mask_ch)
                    all_frames_labels_name.append(labels_name)
                    continue
                # assert frame_id >= 0, f"Frame ID is less than 0: {frame_id}"
                
                v = None
                class_name = None
                for k, cls_id in self.segmentation_labels_id_map.items():
                    if k in label_type:
                        v = cls_id
                        class_name = k
                        # break
                        
                if 'trace' in xml_label:
                    target = xml_label['trace']['mask_cropped']
                    mask = (target > 0).astype(np.float32)
                    self.total_mask_count[label_type] += 1
                    mask = cv2.resize(
                                mask,
                                (256, 256),
                                interpolation=cv2.INTER_NEAREST
                            )
                else:
                # if mask_ch is None:
                    H, W = mask.shape
                    mask = np.zeros((256, 256), dtype=np.float32)
                    self.empty_mask_count[label_type] += 1
                    
                mask_ch[v] = mask   
                labels_name[v] = class_name
                
                frame_ids.append(frame_id) 
                all_frames_masks.append(mask_ch)
                all_frames_labels_name.append(labels_name)
            return all_frames_masks, frame_ids, all_frames_labels_name
        
        else:
        
            mask_ch = np.zeros((4, 256, 256), dtype=np.float32)
            frame_id = None 
            labels_name = [""] * 4
            
            for label_type in label_type_list:
                xml_label = sample['labels'][label_type]
                frame_id = xml_label['frame_num'] - 1
                assert frame_id >= 0, f"Frame ID is less than 0: {frame_id}"
                
                v = None
                class_name = None
                for k, cls_id in self.segmentation_labels_id_map.items():
                    if k in label_type:
                        v = cls_id
                        class_name = k
                        # break
                        
                if 'trace' in xml_label:
                    target = xml_label['trace']['mask_cropped']
                    mask = (target > 0).astype(np.float32)
                    self.total_mask_count[label_type] += 1
                    mask = cv2.resize(
                                mask,
                                (256, 256),
                                interpolation=cv2.INTER_NEAREST
                            )
                else:
                # if mask_ch is None:
                    H, W = mask.shape
                    mask = np.zeros((256, 256), dtype=np.float32)
                    self.empty_mask_count[label_type] += 1
                    
                mask_ch[v] = mask   
                labels_name[v] = class_name  
            return mask_ch, frame_id, labels_name
    
    def __len__(self):
        return len(self.indices)
    
    
    def __getitem__(self, index):
        sample = super().__getitem__(self.indices[index])
        # print(sample.keys())
        video = sample['echo'].astype("float32")
        frames = torch.from_numpy(video).float()
        frames = frames.permute(1, 0, 2, 3)
        new_height, new_width = 256, 256
        orig_height, orig_width = video.shape[-2:]
        dx = sample.get('dX', 0.0)
        dy = sample.get('dY', 0.0)
        # print(f"dx, dY: {dx, dy}")
        scaled_dX =  dx * orig_width / new_width
        scaled_dY =  dy * orig_height / new_height
        sample_stream_id = int(sample['stream_id'])
        save_sample_to_json(sample_stream_id, scaled_dX, scaled_dY)
        
        if self.transform is not None:
            images = self.transform(frames)
        pred_view = sample['labels'].get('pred_view', "None")
        stream_id_list = self.indices[index]

        mask_stack = []

        if self.split =="test":
            target_list, frame_id_list, labels_name_list = self._get_target(sample, index)
            
            for frame_id, target_data in zip(frame_id_list,target_list):
                if frame_id == 0 and len(np.unique(target_data)) == 1:
                    # mask_stack.append(target_data)
                    continue
                mask_stack.append(target_data)
                
                            
            masks = np.stack(mask_stack, axis=0)     # (T, 4, 256, 256)
            # print(f"frame_id_list: {frame_id_list}, masks: {masks.shape},images: {images.shape} ")
            # convert to torch
            masks = torch.from_numpy(masks).float()
            if images.shape[1] > 64:
                start = max(0, min(frame_id_list))
                pad =64
                end = min(images.shape[1], max(frame_id_list) + pad +1)

                images = images[:, start:end]
                 

                final_frame_id_list = [f - start for f in frame_id_list if start <= f < end]
                masks = masks[:len(final_frame_id_list), ...]
                # Sanity check — all frames should be covered
                assert len(final_frame_id_list) == len(frame_id_list), \
                    f"Some frame_ids were out of range: {frame_id_list}, start={start}, end={end}"
            else:
                final_frame_id_list = frame_id_list
                
            # print(f"final_frame_id_list: {final_frame_id_list}, masks: {masks.shape},images: {images.shape} ")
            # if images.shape[1]> 100:
            #     min_frame_idx = min(frame_id_list)
            #     max_frame_idx = max(frame_id_list)
            #     center = (min_frame_idx + max_frame_idx) // 2
            #     half_window = 100 // 2
                
            #     new_start = max(0,  center - half_window)
            #     new_end= new_start + 100
                
            #     T = images.shape[1]
                
            #     if new_end > T:
            #         new_end = T
            #         new_start = max(0,  T - 100)
            #     images = images[:, new_start:new_end, :, :]
            #     frame_id_list = [f - new_start for f in frame_id_list if new_start <= f < new_end]
            # print(images.shape, masks.shape)
            
            return {
                "images": images,
                "masks": masks,
                "spacing": torch.tensor([scaled_dX, scaled_dY]),
                "pred_view": pred_view,
                "labels_name": labels_name_list,
                "stream_id": stream_id_list,
                "exam_id": self.df_split.iloc[index]["exam_id"],
                "frame_id_list": final_frame_id_list
            } 

if __name__=="__main__":
    dataset_test = EchoSegmentationDataset(split='test', visualize_video= False)
    for sample in dataset_test:
        
        print(counter)
    # sample = dataset_test[0]
    # print(sample)
    # for key, value in sample.items():
    #     if isinstance(value, torch.Tensor):
    #         print(key, value.shape)
            