import pymysql
import numpy as np
import pandas as pd
from rclstream.datasets.private import echo
from rclstream.config import get_database_options
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
import torch
import cv2
from collections import defaultdict
import sys

from torchvision import transforms
import random
import torchvision.transforms.functional as T

from torch.utils.data import ConcatDataset

from torchvision.transforms import InterpolationMode

import matplotlib.pyplot as plt 
from collections import defaultdict

def apply_transform(image, mask):
    transforms = []

    # Add horizontal flip with 50% probability
    if random.random() < 0.5:
        transforms.append('hflip')

    # Always add rotation
    transforms.append('rotate')

    # Shuffle order
    random.shuffle(transforms)

    for t in transforms:
        if t == 'hflip':
            image = T.hflip(image)
            mask = T.hflip(mask)
        elif t == 'rotate':
            angle = random.uniform(-15, 15)
            image = T.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = T.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

    return image, mask

class EchoSegmentationDataset(echo.EchoDataset):
    def __init__(
        self,
        split='train',
        visualize_video =False,
        **kwargs
    ):
        super().__init__()

        self.segmentation_labels = [
            'LV_vol_d_MOD_A2C_calc',
            'LV_vol_d_MOD_A4C_calc',
            'LV_vol_s_MOD_A2C_calc',
            'LV_vol_s_MOD_A4C_calc',
            # 'LA_Vol_MOD_A2C_calc',
            # 'LA_Vol_MOD_A4C_calc',
            # 'RA_Vol_MOD_A4C_calc',
            # 'RA_Vol_MOD_A2C_calc',

        ]
        self.segmentation_labels_id_map ={
            'LV_vol_d_MOD_A2C_calc': 1,
            'LV_vol_d_MOD_A4C_calc': 1,
            'LV_vol_s_MOD_A2C_calc': 1,
            'LV_vol_s_MOD_A4C_calc': 1,
            # 'LA_Vol_MOD_A2C_calc': 2,
            # 'LA_Vol_MOD_A4C_calc': 2,
            # 'RA_Vol_MOD_A4C_calc': 3,
            # 'RA_Vol_MOD_A2C_calc': 3,

            
        }
        
        self.split = split
        self.df_split = self._load_metadata()
        self.indices = self.df_split.index.to_list()
        self.segmentation_labels = self.df_split['label_type'].to_list()
        self.visualize_video = visualize_video

        self.empty_mask_count = defaultdict(int)
        self.total_mask_count = defaultdict(int)
        self.exam_ds = echo.EchoExamDataset()

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
        
        
        # self.df_ef = pd.read_pickle("/home/bassant/code/Mod_code_view_pos_encod_mv_seg/ef_results_full_df.pkl")
        # df_file = df_file[df_file["exam_id"].isin(self.df_ef["exam_id"])]
        
        df_files_with_segmentation = pd.merge(df_file.reset_index(drop=False), df_segmentation_labels, left_on='id', right_on='cardiac_file_id', how='inner')     
        print(f"before filtering: {len(df_files_with_segmentation)}")

        stream_ids_test = torch.load("/home/bassant/code/Multi_view_seg_code/data_info/updated_stream_ids_test_patient_level.pt",  weights_only=False)
        # stream_ids_test = stream_ids_test[:1]

        df_test = df_files_with_segmentation[df_files_with_segmentation.stream_id.isin(stream_ids_test)]
        df_test = df_test[:6000]
        # df_test = df_test[:2]
        print(f"new test data size: {len(df_test)}")


        # df_train_agg = self.aggregate_labels(df_train)
        # df_val_agg = self.aggregate_labels(df_val)
        df_test_agg = self.aggregate_labels(df_test)
        df_split = {}
        # df_split['train']= df_train_agg
        # df_split['val']= df_val_agg
        df_split['test'] =df_test_agg
        all_splits = {}
        for split in ['test']:
            unique_labels = (
                df_split[split]['label_type']
                .dropna()
                .explode()   # turns lists into rows
                .value_counts()
            )

            label_df = unique_labels.reset_index().rename(
                    columns={'index': 'label_name', 'label_type': 'count'}
                )
            label_df.to_csv(f"/home/bassant/code/Multi_view_seg_code/data_info/{split}_label_distribution.csv", index=False)
      
            print(f"Saved unique label types to '{split}_label_distribution.csv'")

        df_split["test"].to_csv("/home/bassant/code/Multi_view_seg_code/data_info/test_files_info_patient_level.csv", index=True)   
        print(f"Saved exam and stream ids")

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
        video = sample['echo']
        pred_view = sample['labels'].get('pred_view', None)
        stream_id_list = self.indices[index]
        dx = sample.get('dX', 0.0)
        dy = sample.get('dY', 0.0)
        image_stack = []
        # if self.visualize_video == True:
        #     mask_stack = np.zeros((video.shape[0], 4, 256, 256), dtype=np.float32)
        # else:
        mask_stack = []
        new_height, new_width = 256, 256
        orig_height, orig_width = video.shape[-2:]
        print(f"orig_height, orig_width: {orig_height, orig_width}")
        scaled_dX =  dx * orig_width / new_width
        scaled_dY =  dy * orig_height / new_height
        target_list, frame_id_list, labels_name_list = self._get_target(sample, index)
        print(f"video shape: {video.shape}, frame_id_list:{frame_id_list}")

        # if self.visualize_video == True:
        #     for image_id in range(video.shape[0]):
        #         frame = video[image_id, :, :]
        #         # print(f"frame shape: {frame.shape}")
        #         if frame.shape[0] ==3:
        #             frame = np.transpose(frame, (1, 2, 0))  # → (H,W,C)
        #             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    
        #         if len(frame.shape) == 3 and frame.shape[0] ==1:
        #             frame = frame.squeeze(0) 
        #         print(f"frame shape: {frame.shape}")
        #         frame = frame.astype('float32') / 255.0
        #         frame = cv2.resize(
        #             frame,
        #             (256, 256),
        #             interpolation=cv2.INTER_NEAREST
        #         )
        #         frame = np.expand_dims(frame, axis=(0))# add channel dimension 
        #         image_stack.append(frame)
        #     for frame_id, target_data in zip(frame_id_list,target_list):
        #         mask_stack[frame_id] = target_data
        
        # else:
        for frame_id, target_data in zip(frame_id_list,target_list):
            # if frame_id == 0 and len(np.unique(target_data)) == 1:
            #     frames = np.zeros((1, 256, 256), dtype=np.float32)
            #     image_stack.append(frames)
            #     mask_stack.append(target_data)
            #     continue
            if len(np.unique(target_data)) == 1:
                frames = np.zeros((1, 256, 256), dtype=np.float32)
                image_stack.append(frames)
                mask_stack.append(target_data)
                
                print("No label values in this target")
                continue
            if video.shape[1] == 3:
                frames = video[frame_id,:,:,:]
                frames = np.transpose(frames, (1, 2, 0))  # → (H,W,C)
                frames = cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY)
            

            else:
                frames = video[frame_id, : , :]
            frames = frames.astype('float32') / 255.0

            if frames is None or frames.size == 0:
                raise ValueError(f"Invalid image at frame {frame_id} at index: {index ,self.indices[index]}")

            # image = np.transpose(image, (1, 2, 0))  # (C,H,W) → (H,W,C)
            # print(f"frames shape: {frames.shape}")
            if len(frames.shape) == 3 and frames.shape[0] ==1:
                frames = frames.squeeze(0) 

            # print(f"image shape: {image.shape}")
            frames = cv2.resize(
                    frames,
                    (256, 256),
                    interpolation=cv2.INTER_NEAREST
                )
            
            # cv2.resize(frame, (new_width, new_height))[None, ...]
            frames = np.expand_dims(frames, axis=(0))# add channel dimension 
            image_stack.append(frames)
            mask_stack.append(target_data)
        
                        
        images = np.stack(image_stack, axis=0)   # (T, 1, 256, 256)
        masks = np.stack(mask_stack, axis=0)     # (T, 4, 256, 256)

        # convert to torch
        images = torch.from_numpy(images).float()
        masks = torch.from_numpy(masks).float()
        exam_id = self.df_split.iloc[index]["exam_id"]
        
        exam_data = self.exam_ds["exam_id"] == exam_id
        patient_id = exam_data["patient_id"]
        ef_visual = -1 
        
        # row = self.df_ef.loc[self.df_ef["exam_id"] == exam_id, "ef_value"]
        # if row.empty or pd.isna(row.iloc[0]):
        #     ef_visual = -1  # or None / skip sample
        # else:
        #     ef_visual = int(row.iloc[0])
        return {
            "images": images,
            "masks": masks,
            "spacing": torch.tensor([scaled_dX, scaled_dY]),
            "pred_view": pred_view,
            "labels_name": labels_name_list,
            "stream_id": stream_id_list,
            "patient_id": patient_id,
            "ef_visual": torch.tensor(ef_visual)
        } 


if __name__=="__main__":
    
    # dataset_train = EchoSegmentationDataset(split='train')
    # dataset_val= EchoSegmentationDataset(split='val')
    # dataset_test = EchoSegmentationDataset(split='test')

    dataset_test = MultiViewEchoSegmentationDataset(split='val')

    first_sample = dataset_test[0]
    print(first_sample)
    print(f"labeled_image: {first_sample['labeled_image'].shape}, "
        f"{torch.max(first_sample['labeled_image'])}, {torch.min(first_sample['labeled_image'])}")

    print(f"unlabeled_image: {first_sample['unlabeled_image'].shape}, "
        f"{torch.max(first_sample['unlabeled_image'])}, {torch.min(first_sample['unlabeled_image'])}")

    print(f"labeled_mask: {first_sample['labeled_mask'].shape}, "
        f"{torch.unique(first_sample['labeled_mask'])}")


# import pymysql
# import numpy as np
# import pandas as pd
# from rclstream.datasets.private import echo
# from rclstream.config import get_database_options
# from sklearn.model_selection import train_test_split
# from scipy.ndimage import gaussian_filter
# # import torch.nn.functional as F
# import torch
# import cv2
# from collections import defaultdict
# import sys
# # REPO_DIR = "/data/project/users/bassant/code/ablation_multi_view/multi_view_seg_proj/"
# # sys.path.append(REPO_DIR)
# from torchvision import transforms
# import random
# import torchvision.transforms.functional as T
# # import imgaug.augmenters as iaa
# # from dinov3.data.datasets.rwma_dataset import PrivateRWMADataset



# from torch.utils.data import ConcatDataset

# from torchvision.transforms import InterpolationMode

# import matplotlib.pyplot as plt 
# from collections import defaultdict

# import json
# import os
# import torch

# JSON_PATH = "test_data_dino.json"

# def save_sample_to_json(sample_stream_id, scaled_dX, scaled_dY, path=JSON_PATH):
#     if os.path.exists(path):
#         with open(path, "r") as f:
#             data = json.load(f)
#     else:
#         data = {}

#     data[sample_stream_id] = [scaled_dX, scaled_dY]  # int key, list value

#     with open(path, "w") as f:
#         json.dump(data, f, indent=2)
        
        
# class VideoSegTransform(object):
#     def __init__(
#         self,
#         crop_size=256,
#         random_horizontal_flip=True,
#         normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ):
#         self.crop_size = crop_size
#         self.random_horizontal_flip = random_horizontal_flip

#         self.mean = (
#             torch.tensor(normalize[0], dtype=torch.float32).view(3, 1, 1, 1) * 255.0
#         )
#         self.std = (
#             torch.tensor(normalize[1], dtype=torch.float32).view(3, 1, 1, 1) * 255.0
#         )

#     def __call__(self, video):
#         """
#         video: Tensor (C, T, H, W)
#         """

#         # Resize
#         C, T, H, W = video.shape

#         # Resize video
#         video = F.interpolate(
#             video,
#             size=(self.crop_size, self.crop_size),
#             mode="bilinear",
#             align_corners=False,
#         )


#         # Normalize video

#         if video.dtype == torch.uint8:
#             video = video.float()

#         video = (video - self.mean) / self.std

#         return video


# def apply_transform(image, mask):
#     transforms = []

#     # Add horizontal flip with 50% probability
#     if random.random() < 0.5:
#         transforms.append('hflip')

#     # Always add rotation
#     transforms.append('rotate')

#     # Shuffle order
#     random.shuffle(transforms)

#     for t in transforms:
#         if t == 'hflip':
#             image = T.hflip(image)
#             mask = T.hflip(mask)
#         elif t == 'rotate':
#             angle = random.uniform(-15, 15)
#             image = T.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
#             mask = T.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

#     return image, mask

# class EchoSegmentationDataset(echo.EchoDataset):
#     def __init__(
#         self,
#         split='train',
#         visualize_video =False,
#         **kwargs
#     ):
#         super().__init__()

#         self.segmentation_labels = [
#             'LV_vol_d_MOD_A2C_calc',
#             'LV_vol_d_MOD_A4C_calc',
#             'LV_vol_s_MOD_A2C_calc',
#             'LV_vol_s_MOD_A4C_calc',
#             'LA_Vol_MOD_A2C_calc',
#             'LA_Vol_MOD_A4C_calc',
#             'RA_Vol_MOD_A4C_calc',
#             'RA_Vol_MOD_A2C_calc',

#         ]
#         self.segmentation_labels_id_map ={
#             'LV_vol_d_MOD_A2C_calc': 1,
#             'LV_vol_d_MOD_A4C_calc': 1,
#             'LV_vol_s_MOD_A2C_calc': 1,
#             'LV_vol_s_MOD_A4C_calc': 1,
#             'LA_Vol_MOD_A2C_calc': 2,
#             'LA_Vol_MOD_A4C_calc': 2,
#             'RA_Vol_MOD_A4C_calc': 3,
#             'RA_Vol_MOD_A2C_calc': 3,

            
#         }
        
#         self.split = split
#         self.df_split = self._load_metadata()
#         self.indices = self.df_split.index.to_list()
#         self.segmentation_labels = self.df_split['label_type'].to_list()
#         self.visualize_video = visualize_video

#         self.empty_mask_count = defaultdict(int)
#         self.total_mask_count = defaultdict(int)

#     def aggregate_labels(self, df):
#         """get
#         Aggregate all label_type values per cardiac_file_id into a list.
#         Each stream_id remains unique in the index.
#         """
#         df_sampled = (
#             df
#             .groupby("label_type", group_keys=False)
#             .apply(lambda x: x.assign(label_type=x.name).sample(
#                 n=len(x),
#                 random_state=42,
#                 replace=False
#             ))
#         )
#         df_out = (
#             df_sampled
#             .groupby('cardiac_file_id', as_index=False)
#             .agg(
#                 label_type=('label_type', list),
#                 stream_id=('stream_id', 'first'),
#                 exam_id=('exam_id', 'first')
#             )
#         .set_index('stream_id')
        
#         )
#         return df_out
#     def _load_metadata(self):
#         conn = pymysql.connect(database="echo_inventory", **get_database_options())
#         query = f"SELECT cardiac_file_id, label_type_id FROM label;"
#         df_label = pd.read_sql(query, conn)
#         query = f"SELECT id, name FROM label_type;"
#         df_label_type = pd.read_sql(query, conn)
#         df_label['label_type'] = df_label['label_type_id'].map(df_label_type.set_index('id')['name'])
        

#         df_segmentation_labels = df_label[df_label['label_type'].isin(self.segmentation_labels)]
#         df_file = echo.get_metadata(columns=['id', 'processed_file_address', 'exam_id'])
#         # self.df_ef = pd.read_pickle("/home/bassant/code/Mod_code_view_pos_encod_mv_seg/ef_results_full_df.pkl")
#         # df_file = df_file[df_file["exam_id"].isin(self.df_ef["exam_id"])]

#         df_files_with_segmentation = pd.merge(df_file.reset_index(drop=False), df_segmentation_labels, left_on='id', right_on='cardiac_file_id', how='inner')     
#         print(f"before filtering: {len(df_files_with_segmentation)}")
#         # df_files_with_segmentation = df_files_with_segmentation.loc[
#         #         df_files_with_segmentation["stream_id"] < 8256628
#         # ]
        
#         # unique_labels = df_files_with_segmentation['label_type'].value_counts()
#         # print(unique_labels)
#         # df_unique_labels = unique_labels.reset_index()
#         # df_unique_labels.columns = ['label_type', 'count']
#         # df_unique_labels.to_csv('unique_label_types.csv', index=False)
#         # print("Saved unique label types to 'unique_label_types.csv'")

        
#         # ids = df_files_with_segmentation.stream_id.unique()
#         # # exam_ids = meta_data['exam_id'].to_numpy() 
#         # print(f"Total ids: {len(ids)}")
        
#         # stream_ids_train, rest_ids = train_test_split(ids, test_size=0.2, random_state=42)
#         # stream_ids_val, stream_ids_test = train_test_split(rest_ids, test_size=0.5, random_state=42)
        
#         stream_ids_train = torch.load("/home/bassant/code/Multi_view_seg_code/data_info/stream_ids_train.pt",  weights_only=False)
#         stream_ids_val = torch.load("/home/bassant/code/Multi_view_seg_code/data_info/stream_ids_val.pt",  weights_only=False)
#         stream_ids_test = torch.load("/home/bassant/code/Multi_view_seg_code/data_info/updated_stream_ids_test_patient_level.pt",  weights_only=False)

#         df_train = df_files_with_segmentation[df_files_with_segmentation.stream_id.isin(stream_ids_train)]
#         df_val = df_files_with_segmentation[df_files_with_segmentation.stream_id.isin(stream_ids_val)]
        
#         # df_test = df_files_with_segmentation[
#         #     (~df_files_with_segmentation.exam_id.isin(df_train.exam_id)) &
#         #     (~df_files_with_segmentation.exam_id.isin(df_val.exam_id))
#         # ]
        
#         # df_test = df_files_with_segmentation[
#         #     (~df_files_with_segmentation.patient_id.isin(df_train.patient_id)) &
#         #     (~df_files_with_segmentation.patient_id.isin(df_val.patient_id))
#         # ]
#         # print(f" test sub sample : {len(df_test)}")
#         # df_test = df_test[:3000]
#         # print(f" final test sub sample : {len(df_test)}")

#         # torch.save(
#         #     df_files_with_segmentation.stream_id.tolist(),
#         #     "/home/bassant/code/Multi_view_seg_code/data_info/updated_stream_ids_test_patient_level.pt"
#         # )

        
#         # stream_ids_test = torch.load("/home/bassant/code/Multi_view_seg_code/data_info/updated_stream_ids_test.pt",  weights_only=False)

#         # uploaded_df_test = df_files_with_segmentation[df_files_with_segmentation.stream_id.isin(stream_ids_test)]
#         # uploaded_df_test.equals(df_test)
#         df_test = df_files_with_segmentation[df_files_with_segmentation.stream_id.isin(stream_ids_test)]
#         df_test = df_test[:6000]
#         # df_test = df_test[:100]
#         print(f"new test data size: {len(df_test)}")


#         df_train_agg = self.aggregate_labels(df_train)
#         df_val_agg = self.aggregate_labels(df_val)
#         df_test_agg = self.aggregate_labels(df_test)
#         df_split = {}
#         df_split['train']= df_train_agg
#         df_split['val']= df_val_agg
#         df_split['test'] =df_test_agg
#         # all_splits = {}
#         # for split in ['train', 'val', 'test']:
#         #     unique_labels = (
#         #         df_split[split]['label_type']
#         #         .dropna()
#         #         .explode()   # turns lists into rows
#         #         .value_counts()
#         #     )
#         #     # all_splits[split] = unique_labels.reset_index().rename(
#         #     #     columns={'index': 'label_name', 'label_type': 'count'}
#         #     # )
#         #     label_df = unique_labels.reset_index().rename(
#         #             columns={'index': 'label_name', 'label_type': 'count'}
#         #         )
#         #     label_df.to_csv(f"/home/bassant/code/Multi_view_seg_code/data_info/{split}_label_distribution.csv", index=False)
#         #     # unique_labels = df_split[split]['label_type'].dropna().unique()
#         #     # pd.Series(unique_labels).to_csv(f'/home/bassant/code/Multi_view_seg_code/data_info/{split}_label_types.csv', index=False)
#         #     print(f"Saved unique label types to '{split}_label_distribution.csv'")
#         # # self.df_split.iloc[index]["exam_id"] 
#         # # df_split["train"].to_csv("/home/bassant/code/Multi_view_seg_code/data_info/train_files_info.csv", index=True)   
#         # # df_split["val"].to_csv("/home/bassant/code/Multi_view_seg_code/data_info/val_files_info.csv", index=True)   
#         # df_split["test"].to_csv("/home/bassant/code/Multi_view_seg_code/data_info/test_files_info_patient_level.csv", index=True)   
#         # print(f"Saved exam and stream ids")
#         # # torch.save(list(stream_ids_train), "/home/bassant/code/Multi_view_seg_code/data_info/stream_ids_train.pt")
#         # # torch.save(list(stream_ids_val), "/home/bassant/code/Multi_view_seg_code/data_info/stream_ids_val.pt")
#         # # torch.save(list(stream_ids_test), "/home/bassant/code/Multi_view_seg_code/data_info/stream_ids_test.pt")
        
#         print(f"len of {self.split} echo data: {len(df_split[self.split])}")
#         return df_split[self.split]
    
#     # def _remove_used_samples(self, df_file):

#     #     exclude_df = pd.concat([
#     #         pd.read_csv('/data/project/users/bassant/code/DINOV3/df_train.csv').set_index('stream_id'),
#     #         pd.read_csv('/data/project/users/bassant/code/DINOV3/df_test.csv').set_index('stream_id'),
#     #     ], axis=0)
#     #     df_file = df_file[~df_file.index.isin(exclude_df.index)]

#     #     return df_file

 
#     def _get_target(self, sample, index):
#         label_type_list = self.segmentation_labels[index]
#         frame_ids = []
#         all_frames_masks = []
#         all_frames_labels_name = []
            
#         if self.split =="test":
#             # collect frame ids
#             for label_type in label_type_list:
#                 mask_ch = np.zeros((4, 256, 256), dtype=np.float32)
#                 labels_name = [""] * 4
#                 xml_label = sample['labels'][label_type]
#                 frame_id = xml_label['frame_num'] - 1
#                 if frame_id <0:
#                     frame_id = 0
#                     frame_ids.append(frame_id) 
#                     all_frames_masks.append(mask_ch)
#                     all_frames_labels_name.append(labels_name)
#                     continue
#                 # assert frame_id >= 0, f"Frame ID is less than 0: {frame_id}"
                
#                 v = None
#                 class_name = None
#                 for k, cls_id in self.segmentation_labels_id_map.items():
#                     if k in label_type:
#                         v = cls_id
#                         class_name = k
#                         # break
                        
#                 if 'trace' in xml_label:
#                     target = xml_label['trace']['mask_cropped']
#                     mask = (target > 0).astype(np.float32)
#                     self.total_mask_count[label_type] += 1
#                     mask = cv2.resize(
#                                 mask,
#                                 (256, 256),
#                                 interpolation=cv2.INTER_NEAREST
#                             )
#                 else:
#                 # if mask_ch is None:
#                     H, W = mask.shape
#                     mask = np.zeros((256, 256), dtype=np.float32)
#                     self.empty_mask_count[label_type] += 1
                    
#                 mask_ch[v] = mask   
#                 labels_name[v] = class_name
                
#                 frame_ids.append(frame_id) 
#                 all_frames_masks.append(mask_ch)
#                 all_frames_labels_name.append(labels_name)
#             return all_frames_masks, frame_ids, all_frames_labels_name
        
#         else:
        
#             mask_ch = np.zeros((4, 256, 256), dtype=np.float32)
#             frame_id = None 
#             labels_name = [""] * 4
            
#             for label_type in label_type_list:
#                 xml_label = sample['labels'][label_type]
#                 frame_id = xml_label['frame_num'] - 1
#                 assert frame_id >= 0, f"Frame ID is less than 0: {frame_id}"
                
#                 v = None
#                 class_name = None
#                 for k, cls_id in self.segmentation_labels_id_map.items():
#                     if k in label_type:
#                         v = cls_id
#                         class_name = k
#                         # break
                        
#                 if 'trace' in xml_label:
#                     target = xml_label['trace']['mask_cropped']
#                     mask = (target > 0).astype(np.float32)
#                     self.total_mask_count[label_type] += 1
#                     mask = cv2.resize(
#                                 mask,
#                                 (256, 256),
#                                 interpolation=cv2.INTER_NEAREST
#                             )
#                 else:
#                 # if mask_ch is None:
#                     H, W = mask.shape
#                     mask = np.zeros((256, 256), dtype=np.float32)
#                     self.empty_mask_count[label_type] += 1
                    
#                 mask_ch[v] = mask   
#                 labels_name[v] = class_name  
#             return mask_ch, frame_id, labels_name
    
#     def __len__(self):
#         return len(self.indices)
    
    
#     def __getitem__(self, index):
#         sample = super().__getitem__(self.indices[index])
#         # print(sample.keys())
#         video = sample['echo']
#         pred_view = sample['labels'].get('pred_view', "")
#         stream_id_list = self.indices[index]
#         dx = sample.get('dX', 0.0)
#         dy = sample.get('dY', 0.0)
#         image_stack = []
#         mask_stack = []
#         new_height, new_width = 256, 256
#         orig_height, orig_width = video.shape[-2:]
#         # print(f"orig_height, orig_width: {orig_height, orig_width}")
#         scaled_dX =  dx * (orig_width / new_width)
#         scaled_dY =  dy * (orig_height / new_height)
        
#         sample_stream_id = int(sample['stream_id'])
#         # save_sample_to_json(sample_stream_id, scaled_dX, scaled_dY)
#         if self.split =="test":
#             target_list, frame_id_list, labels_name_list = self._get_target(sample, index)
            
#             for frame_id, target_data in zip(frame_id_list,target_list):
#                 if frame_id == 0 and len(np.unique(target_data)) == 1:
#                     frames = np.zeros((1, 256, 256), dtype=np.float32)
#                     image_stack.append(frames)
#                     mask_stack.append(target_data)
#                     continue
#                 if video.shape[1] == 3:
#                     frames = video[frame_id,:,:,:]
#                     frames = np.transpose(frames, (1, 2, 0))  # → (H,W,C)
#                     frames = cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY)
#                 elif self.visualize_video == True:
#                     frames = []
#                     masks = np.zeros((video.shape[0], 4, 256, 256), dtype=np.float32)
#                     for image_id in range(video.shape[0]):
#                         image_frame = video[image_id, : , :]
#                         image_frame = image_frame.astype('float32') / 255.0
                        
#                         if frame_id == image_id:
#                             masks[image_id] = target_data

#                         # if image_frame is None or image_frame.size == 0:
#                         #     raise ValueError(f"Invalid image at frame {frame_id} at index: {index ,self.indices[index]}")

                        
#                         # # image = np.transpose(image, (1, 2, 0))  # (C,H,W) → (H,W,C)
#                         # image_frame = image_frame.squeeze(0) 
#                         # # print(f"image shape: {image_frame.shape}")
#                         # image_frame = cv2.resize(
#                         #         image_frame,
#                         #         (256, 256),
#                         #         interpolation=cv2.INTER_NEAREST
                
#                         #     )
#                         frames.append(image_frame)
#                     frames = np.stack(frames, axis=0)
#                     # frames = np.expand_dims(frames, axis=(1))# add channel dimension 
#                 else:
#                     frames = video[frame_id, : , :]
#                     # print(f"video shape: {video.shape}, frame_id: {frame_id}")
#                 frames = frames.astype('float32') / 255.0
#                 # frame: (np.float32(0.0), np.float32(1.0))
#                 # print(f"frame: {np.min(image), np.max(image)}")

#                 # if frames is None or frames.size == 0:
#                 #     raise ValueError(f"Invalid image at frame {frame_id} at index: {index ,self.indices[index]}")
   
#                 # image = np.transpose(image, (1, 2, 0))  # (C,H,W) → (H,W,C)
#                 print(f"frames shape: {frames.shape}")
#                 if len(frames.shape) == 3 and frames.shape[0] ==1 :
#                     frames = frames.squeeze(0) 
#                 if len(frames.shape) == 4 and frames.shape[1] ==1 : 
#                     frames = frames.squeeze(1)   
                    

#                 # print(f"image shape: {image.shape}")

                

#                 if self.visualize_video == True: 
#                     frames = np.stack([cv2.resize(f, (256, 256),
#                         interpolation=cv2.INTER_LINEAR) for f in frames])
#                     frames = frames[:, None]      
#                     images = frames
                    
#                 else:
#                     mask_stack.append(target_data)
#                     frames = cv2.resize(
#                         frames,
#                         (256, 256),
#                         interpolation=cv2.INTER_NEAREST
#                     )
#                     frames = np.expand_dims(frames, axis=(0))# add channel dimension 
#                     image_stack.append(frames)
                
                            
#                     images = np.stack(image_stack, axis=0)   # (T, 1, 256, 256)
#                     masks = np.stack(mask_stack, axis=0)     # (T, 4, 256, 256)

#             # convert to torch
#             images = torch.from_numpy(images).float()
#             masks = torch.from_numpy(masks).float()
#             print(f" in test data set: images {images.shape}, masks: {masks.shape}")
#             exam_id = self.df_split.iloc[index]["exam_id"]
#             ef_visual = 0
#             # ef_visual = int(self.df_ef.loc[self.df_ef["exam_id"] == exam_id, "ef_value"].iloc[0])
#             return {
#                 "images": images,
#                 "masks": masks,
#                 "spacing": torch.tensor([scaled_dX, scaled_dY]),
#                 "pred_view": pred_view,
#                 "labels_name": labels_name_list,
#                 "stream_id": stream_id_list,
#                 "exam_id": self.df_split.iloc[index]["exam_id"],
#                 "ef_visual": torch.tensor(ef_visual)
#             } 

#         else:
#             target, frame_id, labels_name = self._get_target(sample, index)
#             print(f" after mask processing: {target.shape}")
#             print(f"labels_name: {labels_name}, frame_id: {frame_id}")
#             new_frame_id = max(0, min(frame_id, video.shape[0] - 1))
#             orig_height, orig_width = video.shape[-2:]
#             print(f"orig_height, orig_width: {orig_height, orig_width}")
#             new_height, new_width = 256, 256

#             if self.visualize_video:
#                 frames = []
#                 for frame_id in range(video.shape[0]):
#                     image_frame = video[frame_id, : , :]
#                     image_frame = image_frame.astype('float32') / 255.0

#                     if image_frame is None or image_frame.size == 0:
#                         raise ValueError(f"Invalid image at frame {frame_id} at index: {index ,self.indices[index]}")

                    
#                     # image = np.transpose(image, (1, 2, 0))  # (C,H,W) → (H,W,C)
#                     image_frame = image_frame.squeeze(0) 
#                     # print(f"image shape: {image_frame.shape}")
#                     image_frame = cv2.resize(
#                             image_frame,
#                             (256, 256),
#                             interpolation=cv2.INTER_NEAREST
            
#                         )
#                     frames.append(image_frame)
#                 frames = np.stack(frames, axis=0)
#                 frames = np.expand_dims(frames, axis=(1))# add channel dimension 
#             else:

#                 frames = video[new_frame_id, : , :]
#                 # print(f"video shape: {video.shape}, new_frame_id: {new_frame_id}")
#                 frames = frames.astype('float32') / 255.0
#                 # frame: (np.float32(0.0), np.float32(1.0))
#                 # print(f"frame: {np.min(image), np.max(image)}")

#                 if frames is None or frames.size == 0:
#                     raise ValueError(f"Invalid image at frame {new_frame_id} at index: {index ,self.indices[index]}")

                
#                 # image = np.transpose(image, (1, 2, 0))  # (C,H,W) → (H,W,C)
#                 frames = frames.squeeze(0) 
#                 # print(f"image shape: {image.shape}")
#                 frames = cv2.resize(
#                         frames,
#                         (256, 256),
#                         interpolation=cv2.INTER_NEAREST
#                     )
                
#                 # cv2.resize(frame, (new_width, new_height))[None, ...]
#                 frames = np.expand_dims(frames, axis=(0))# add channel dimension 
            
            
#             frames = torch.from_numpy(frames).float() 
#             target = torch.from_numpy(target).float()

#             if self.split =="train" and random.random() < 0.3:
#                 frames, target = apply_transform(frames, target)


#             scaled_dX =  dx * orig_width / new_width
#             scaled_dY =  dy * orig_height / new_height
        
            
#             frames =frames.unsqueeze(0)
#             print(f" in base data set: images {frames.shape}, masks: {target.shape}")
#             return {
#                 "images": frames,
#                 "masks": target,
#                 "spacing": torch.tensor([scaled_dX, scaled_dY]),
#                 "pred_view": pred_view, 
#                 "labels_name": labels_name,
#                 "stream_id": stream_id_list,
#                 "exam_id": self.df_split.iloc[index]["exam_id"]
#                 }   


# class MultiViewEchoSegmentationDataset(EchoSegmentationDataset):
#     def __init__(
#         self,
#         split='train',
#         visualize_video =False,
#         **kwargs
#     ):
#         super().__init__(split=split, visualize_video = visualize_video)
#         self.meta = echo.get_metadata(columns=["exam_id", "predicted_cardiac_view_id", "processed_file_address"])

#         # self.exam_dataset = 
#         self.views_dict ={
#             "AP2": 3,
#             "AP4": 2,
#             "AP5": 3,
#             "AP3": 2,
#         }  
#     def __getitem__(self, index):
#         # return {
#         #         "images": frames,
#         #         "masks": target,
#         #         "spacing": torch.tensor([scaled_dX, scaled_dY]),
#         #         "pred_view": pred_view, 
#         #         "labels_name": labels_name,
#         #         "stream_id": stream_id_list,
#         #         "exam_id": sample["exam_id"]
#         #         } 
#         # targets = []
#         sample_dict = super().__getitem__(index)
#         first_image = sample_dict["images"]
#         # frames.append(first_image)
#         first_target = sample_dict["masks"]
#         print(f"first_target: {first_target.shape}")

#         # second_target = torch.zeros(1, 4, 256, 256)
#         # targets.append(first_target)
#         # targets.append(second_target)
#         sample_exam_id = sample_dict["exam_id"]
#         # exam_data = exam_dataset[exam_dataset['stream_id']== sample_stream_id]
#         # sample_view "AP4" , "AP2"
#         sample_view = sample_dict["pred_view"]
#         print(f"sample_view: {sample_view}")
#         sample_meta_data = self.meta[
#             (self.meta["exam_id"] == sample_exam_id) &
#             (self.meta["predicted_cardiac_view_id"] == self.views_dict[sample_view])
#         ]
#         # sample_meta_data = self.meta[self.meta["exam_id"] == sample_exam_id and self.meta["predicted_cardiac_view_id"] in views_dict[sample_view]]
#         other_view_data_stream_id = sample_meta_data.index.tolist()[0]
#         filtered_dataset = echo.EchoDataset(columns=["exam_id", "predicted_cardiac_view_id"], stream_ids = [other_view_data_stream_id])
#         other_view_sample = filtered_dataset[0]
#         # print(f"other_view_sample: {other_view_sample}")
#         other_view_sample_video = other_view_sample['echo']
#         other_view_sample_view =  other_view_sample["predicted_cardiac_view_id"]
#         selected_frame_idx = np.random.randint(other_view_sample_video.shape[0])
#         # CHECK CHANNELS INPUT SHAPE 
#         if other_view_sample_video.shape[1] == 3:
#             second_frame = other_view_sample_video[selected_frame_idx,:,:,:]
#             if second_frame.shape[0] == 3:  # (C,H,W)
#                 second_frame = np.transpose(second_frame, (1, 2, 0))  # → (H,W,C)
#             if second_frame.shape[2] == 3:  # RGB → Gray
#                 second_frame = cv2.cvtColor(second_frame, cv2.COLOR_RGB2GRAY)
#         elif other_view_sample_video.shape[1] == 1:
#             second_frame = other_view_sample_video[selected_frame_idx,:,:]
            
#         print(f"video shape: {other_view_sample_video.shape}, selected_frame: {selected_frame_idx} of shape :{second_frame.shape}")
#         second_frame = second_frame.astype('float32') / 255.0
#         # frame: (np.float32(0.0), np.float32(1.0))
#         # print(f"frame: {np.min(image), np.max(image)}")

#         if second_frame is None or second_frame.size == 0:
#             raise ValueError(f"Invalid image at frame {selected_frame_idx}")

#         if second_frame.shape[0] == 1 and len(second_frame.shape) ==3:
#             second_frame = np.squeeze(second_frame)
#         # image = np.transpose(image, (1, 2, 0))  # (C,H,W) → (H,W,C)
#         # second_frame = second_frame.squeeze(0) 
#         # print(f"image shape: {image.shape}")
#         second_frame = cv2.resize(
#                 second_frame,
#                 (256, 256),
#                 interpolation=cv2.INTER_NEAREST
#             )
        
#         # cv2.resize(frame, (new_width, new_height))[None, ...]
#         second_frame = np.expand_dims(second_frame, axis=(0))# add channel dimension  
            
#         unlabeled_image = torch.from_numpy(second_frame).float()

#         unlabeled_image =unlabeled_image.unsqueeze(0)
#         first_target = first_target.unsqueeze(0)
#         print(f"first_target: {first_target.shape}")
#         return {
#             "labeled_image": first_image,
#             "labeled_mask": first_target,
#             "unlabeled_image": unlabeled_image,
#             "spacing": sample_dict["spacing"],
#             "pred_view": sample_dict["pred_view"], 
#             "labels_name": sample_dict["labels_name"],
#             "stream_id": sample_dict["stream_id"],
#         }


# if __name__=="__main__":
    
#     # dataset_train = EchoSegmentationDataset(split='train')
#     # dataset_val= EchoSegmentationDataset(split='val')
#     # dataset_test = EchoSegmentationDataset(split='test')

#     dataset_test = MultiViewEchoSegmentationDataset(split='val')

#     first_sample = dataset_test[0]
#     print(first_sample)
#     print(f"labeled_image: {first_sample['labeled_image'].shape}, "
#         f"{torch.max(first_sample['labeled_image'])}, {torch.min(first_sample['labeled_image'])}")

#     print(f"unlabeled_image: {first_sample['unlabeled_image'].shape}, "
#         f"{torch.max(first_sample['unlabeled_image'])}, {torch.min(first_sample['unlabeled_image'])}")

#     print(f"labeled_mask: {first_sample['labeled_mask'].shape}, "
#         f"{torch.unique(first_sample['labeled_mask'])}")
