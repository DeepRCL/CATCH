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

    def __call__(self, video, mask):
        """
        video: Tensor (C, T, H, W)
        mask : Tensor (T, H, W) OR (1, T, H, W)
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

        # Resize mask
        mask = mask.float()

        mask = F.interpolate(
            mask.unsqueeze(0).float(),  # (1, C, H, W)
            size=(self.crop_size, self.crop_size),
            mode="nearest",
        ).squeeze(0)

        # Random horizontal flip

        if self.random_horizontal_flip and random.random() < 0.5:
            video = torch.flip(video, dims=[3])  # flip W dimension
            mask = torch.flip(mask, dims=[2])

        # Normalize video

        if video.dtype == torch.uint8:
            video = video.float()

        video = (video - self.mean) / self.std

        return video, mask.long()







class MultiViewEchoSegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split='train',
        visualize_video =False,
        req_views = [],
        transform= None,
        use_video_input = True,
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
        # self.req_views = ['AP3', 'AP4', 'AP2','AP5', 'PLAX']
        self.req_views = req_views
        self.clip_len = 4
        # self.req_views = ['AP4', 'AP2']
        self.split = split
        self.transform = transform
        self.use_video_input = use_video_input
                
        self.df_split = self._load_metadata()
        

            
        self.indices = self.df_split['exam_id'].to_list()
        
        if self.split =="val":
            self.indices = self.indices[:100]
            
        # self.indices = self.indices[:1]
        
        self.echo_exam_dataset = echo.EchoExamDataset(exam_ids = self.indices)
        self.visualize_video = visualize_video

        self.empty_mask_count = defaultdict(int)
        self.total_mask_count = defaultdict(int)

    #     self.bad_indices_file = "/data/project/users/bassant/code/code_from_b300/MV/data_info/bad_indices_vjepa.txt"
    #     self.bad_indices = self._load_bad_indices()
    
    # def _load_bad_indices(self):
    #     """Load previously recorded bad indices from file"""
    #     bad_indices = set()
    #     if os.path.exists(self.bad_indices_file):
    #         try:
    #             with open(self.bad_indices_file, 'r') as f:
    #                 bad_indices = set(int(line.strip()) for line in f if line.strip())
    #             print(f"Loaded {len(bad_indices)} bad indices from {self.bad_indices_file}")
    #         except Exception as e:
    #             print(f"Error loading bad indices: {e}")
    #     return bad_indices
    
    # def _save_bad_index(self, index):
    #     """Append a new bad index to the file"""
    #     with open(self.bad_indices_file, 'a') as f:
    #         f.write(f"{index}\n")
    #     self.bad_indices.add(index)
    #     print(f"Saved bad index {index} to {self.bad_indices_file}")
        
    def _load_metadata(self):
        
        df_split = {}
        df_split['train']= pd.read_csv("/home/bassant/code/Multi_view_seg_code/data_info/train_files_info.csv", index_col=0)
        df_split['val']= pd.read_csv("/home/bassant/code/Multi_view_seg_code/data_info/val_files_info.csv", index_col=0)
        df_split['test'] =pd.read_csv("/home/bassant/code/Multi_view_seg_code/data_info/test_files_info_patient_level.csv", index_col=0)

        return df_split[self.split]
    

    
    def _get_target(self, sample):
            label_type_list = self.segmentation_labels
            orig_height, orig_width = sample["echo"].shape[-2:]
            mask_ch = np.zeros((3, orig_height, orig_width), dtype=np.float32)
            frame_id = None 
            labels_name = [""] * 3
            
            for label_type in label_type_list:
                if label_type not in sample['labels'].keys():
                    continue
                xml_label = sample['labels'][label_type]
                frame_id = xml_label['frame_num'] - 1
                if frame_id < 0:
                    frame_id = 0
                # assert frame_id >= 0, f"Frame ID is less than 0: {frame_id}"
                
                v = None
                class_name = None
                for k, cls_id in self.segmentation_labels_id_map.items():
                    if k in label_type:
                        v = cls_id
                        class_name = k
                        
                if 'trace' in xml_label:
                    target = xml_label['trace']['mask_cropped']
                    mask = (target > 0).astype(np.float32)
                    # mask = cv2.resize(
                    #             mask,
                    #             (256, 256),
                    #             interpolation=cv2.INTER_NEAREST
                    #         )
                else:
                    mask = np.zeros((orig_height, orig_width), dtype=np.float32)
                    
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
            # Skip already-known bad indices
            # if current_index not in self.bad_indices:
            try:
                sample = self.echo_exam_dataset[current_index]
                break  # Successfully loaded, exit loop
            except Exception as e:
                print(f"Index {current_index} failed: {e}")
                # self._save_bad_index(current_index)  # Save to file
            # else:
            #     print(f"Skipping known bad index: {current_index}")
            
            # Move to next index
            current_index = (current_index + 1) % len(self)
            attempts += 1
            
            # Safety check to avoid infinite loop
            if current_index == original_index and attempts > 1:
                raise RuntimeError(f"All indices are bad or unreachable")
        
        # Check if we successfully loaded a sample
        if sample is None:
            raise RuntimeError(f"Could not find valid sample after {max_attempts} attempts")
    
        # sample = self.echo_exam_dataset[index]
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
                if self.use_video_input: 
                    half_clip = 2
                    total_frames = video.shape[0]
                    frame_id = 0 if frame_id is None else frame_id
                    new_frame_id = np.clip(frame_id, 0, total_frames - 1)
                    # define temporal window around target frame
                    start = frame_id - half_clip
                    end = frame_id + half_clip
                    if start < 0:
                        start = 0
                        end = self.clip_len

                    if end > total_frames:
                        end = total_frames
                        start = total_frames - self.clip_len

                    if total_frames < self.clip_len:
                        start = 0
                        end = total_frames

                    sampled_frame_ids = np.arange(start, end)
                    # Pad if needed
                    if len(sampled_frame_ids) < self.clip_len:
                        pad_count = self.clip_len - len(sampled_frame_ids)
                        sampled_frame_ids = np.concatenate(
                            [sampled_frame_ids, np.full(pad_count, sampled_frame_ids[-1])]
                        )

                    modified_index = new_frame_id - start
                    orig_height, orig_width = video.shape[-2:]
                    if self.use_video_input:
                        frames = []
                        for frame_id in sampled_frame_ids:
                            image_frame = video[frame_id]
                            image_frame = image_frame.astype("float32")

                            if image_frame.shape[0] == 1:
                                image_frame = image_frame.squeeze(0)
                                image_frame = np.repeat(image_frame[None, :, :], 3, axis=0)

                            elif len(image_frame.shape) == 2:
                                image_frame = np.repeat(image_frame[None, :, :], 3, axis=0)

                            if image_frame is None or image_frame.size == 0:
                                raise ValueError(
                                    f"Invalid image at frame {frame_id} at index: {index ,self.indices[index]}"
                                )

                            frames.append(image_frame)
                        frames = np.stack(frames, axis=0)
                    
                
                frames = torch.from_numpy(frames).float()  # (T, C H, W)
                target = torch.from_numpy(mask_ch).float()

                #  convert  (T,C H, W) > (C, T, H,W)
                frames = frames.permute(1, 0, 2, 3)
                if self.transform is not None:
                    frames, target = self.transform(frames, target)

                scaled_dX = dx * orig_width / 256
                scaled_dY = dy * orig_height / 256
                
                candidate = {
                    "view": view,
                    "stream_id": sample["stream_id"][idx],
                    "echo": frames,  # (C, T, H,W)
                    "spacing:": torch.tensor([scaled_dX, scaled_dY]),
                    "loss_type": has_label,
                    "seg_info": {
                        "mask": target,
                        "frame_id": modified_index,
                        "labels_name": labels_name,
                        
                    } if has_label else None
                }
                
                # Keep only one per view, prioritize loss_type == 1
                if view not in view_best:
                    view_best[view] = candidate
                elif view_best[view]["loss_type"] == 0 and has_label == 1:
                    view_best[view] = candidate
            else:
                continue
            
        final_dict["patient_id"] = sample["patient_id"]
        # Define the exact order of views you expect
        ordered_views = ['AP4', 'AP2']
        
        all_images = []
        all_masks = []
        all_stream_ids = []
        all_loss_types = []
        all_spacing = []
        all_masks_indicies = []

        # for v_name in ordered_views:
        #     if v_name in view_best:
        #         v = view_best[v_name]
        #         print(v.keys())
        #         all_images.append(v["echo"])
        #         if v["loss_type"] == 1:
        #             all_masks.append(v["seg_info"]["mask"], dtype=torch.float32)
        #             all_spacing.append(torch.zeros(2), dtype=torch.float32)
        #         #     all_spacing.append(torch.tensor(v["spacing"]))
        #             # all_masks_indicies.append(torch.tensor(v["seg_info"]["frame_id"]))
        #             all_masks_indicies.append(torch.tensor(v["seg_info"]["frame_id"], dtype=torch.long))
        #         else:
        #             all_masks.append(torch.zeros((3, 256, 256), dtype=torch.float32))
        #             all_spacing.append(torch.zeros(2), dtype=torch.float32)
        #             all_masks_indicies.append(torch.tensor(-1), dtype=torch.long)  
        #         all_stream_ids.append(torch.tensor(v["stream_id"], dtype=torch.long))
        #         all_loss_types.append(torch.tensor(v["loss_type"], dtype=torch.long))
                
        #         # if v["loss_type"] == 1:


        #     else:
        #         # View is MISSING -> Append dummy zero tensors so shapes match!
        #         all_images.append(torch.zeros((3, 4,  256, 256), dtype=torch.float32)) # Same shape as valid image
        #         all_masks.append(torch.zeros((3, 256, 256), dtype=torch.float32))
        #         all_stream_ids.append(torch.tensor(-1), dtype=torch.long)          # Dummy ID
        #         all_loss_types.append(torch.tensor(0), dtype=torch.long)        
        #         all_spacing.append(torch.zeros(2), dtype=torch.long)
        #         all_masks_indicies.append(torch.tensor(-1), dtype=torch.long)
        for v_name in ordered_views:
            if v_name in view_best:
                v = view_best[v_name]
                # print(v.keys())

                # images
                all_images.append(v["echo"])

                if v["loss_type"] == 1:
                    all_masks.append(torch.tensor(v["seg_info"]["mask"], dtype=torch.float32))
                    all_spacing.append(torch.zeros(2, dtype=torch.float32))

                    all_masks_indicies.append(
                        torch.tensor(v["seg_info"]["frame_id"], dtype=torch.long)
                    )
                else:
                    all_masks.append(torch.zeros((3, 256, 256), dtype=torch.float32))
                    all_spacing.append(torch.zeros(2, dtype=torch.float32))

                    all_masks_indicies.append(torch.tensor(-1, dtype=torch.long))

                all_stream_ids.append(torch.tensor(v["stream_id"], dtype=torch.long))
                all_loss_types.append(torch.tensor(v["loss_type"], dtype=torch.long))

            else:
                # Missing view → dummy tensors (MUST match real ones)

                all_images.append(torch.zeros((3, 4, 256, 256), dtype=torch.float32))
                all_masks.append(torch.zeros((3, 256, 256), dtype=torch.float32))

                all_stream_ids.append(torch.tensor(-1, dtype=torch.long))
                all_loss_types.append(torch.tensor(0, dtype=torch.long))

                all_spacing.append(torch.zeros(2, dtype=torch.float32))
                all_masks_indicies.append(torch.tensor(-1, dtype=torch.long))
        
        final_dict = {
            "patient_id": sample["patient_id"],
            "images": torch.stack(all_images),      # Configured exactly to: (3, 2, 1, 256, 256)
            "masks": torch.stack(all_masks),        # Configured exactly to: (3, 3, 256, 256)
            "masks_indices": torch.stack(all_masks_indicies),
            "stream_ids": torch.stack(all_stream_ids),
            "loss_type": torch.stack(all_loss_types),
            "spacing": torch.stack(all_spacing),
            "views": ordered_views
            
            
        }        
        return final_dict  
        # for v in view_best.values():
        #     sample_dict["views"].append(v["view"])
        #     sample_dict["stream_ids"].append(int(v["stream_id"]))
        #     sample_dict["echo"].append(v["echo"])
        #     sample_dict["loss_type"].append(v["loss_type"])

        #     if v["loss_type"] == 1:
        #         sample_dict[":1"].append(v["seg_info"])
        #     else:
        #         sample_dict["seg_labels_info"].append(None)
        

        # return sample_dict



# if __name__=="__main__":
    # patients_per_split =  Counter()
    # files_list = ["/data/project/users/bassant/code/code_from_b300/MV/data_info/train_files_info.csv", "/data/project/users/bassant/code/code_from_b300/MV/data_info/val_files_info.csv", "/data/project/users/bassant/code/code_from_b300/MV/data_info/test_files_info_patient_level.csv"]
    # splits_name = ["train", "val", "test"]
    # for data_f_path, data_split in zip (files_list, splits_name):
    #     df = pd.read_csv(data_f_path)
    #     patients = analysis_dataset(df, data_split)
    #     patients_per_split[data_split] = patients
    # from itertools import combinations

    # for s1, s2 in combinations(splits_name, 2):
    #     overlap = set(patients_per_split[s1]) & set(patients_per_split[s2])
        
    #     print(f"{s1} ∩ {s2}: {len(overlap)} patients")
    # multi_view_dataset = MultiViewEchoSegmentationDataset(split='train')
    # sample = multi_view_dataset[0]
    # print(sample)