from rclstream.datasets.public import camus
from rclstream.datasets.public import echonet

import numpy as np 
from torch.utils.data import Subset
import os 
import torch
import torch.nn.functional as F
import cv2
from scipy.io import loadmat


import torch
def calc_ef_from_seg(pred_mask_1, pred_mask_2):
    volume_1 = torch.sum(pred_mask_1, dim=(1, 2))
    volume_2 = torch.sum(pred_mask_2, dim=(1, 2))

    stacked = torch.stack([volume_1, volume_2], dim=1)  # (B, 2)
    EDV = stacked.max(dim=1).values                     # (B,)
    ESV = stacked.min(dim=1).values                     # (B,)

    print(f"EDV:{EDV.shape}, {EDV}, ESV: {ESV.shape}, {ESV}")
    pred_ef = 100 * (EDV - ESV) / EDV                  # (B,)
    return pred_ef, EDV, ESV

class camus_test(camus.CAMUSDataset):
    
    def __init__(self, split = "train", transforms=None):
        super().__init__()
        
        self.transforms = transforms
        self.view_list=["AP4", "AP2"]
        # self.view = view
        self.num_classes = 4 
        self.split = split
        metadata = camus.get_metadata()
        if self.split  == "test":
            patients = np.loadtxt(
                "/home/bassant/data/camus/camus_test/subgroup_testing.txt",
                # "/home/bassant/data/camus/camus_test/test_val.txt",
                dtype=str
            )
        elif self.split  =="train":
            patients = np.loadtxt(
                "/home/bassant/data/camus/camus_test/subgroup_training.txt",
                dtype=str
            )
        elif self.split  =="val":
            patients = np.loadtxt(
                "/home/bassant/data/camus/camus_test/subgroup_validation.txt",
                dtype=str
            )
        # Filter for specific cardiac views (e.g., 4CH views)
        self.filtered_indices = metadata[
            (metadata["cardiac_view"].isin(self.view_list)) &
            (metadata["file_path"].apply(lambda x: os.path.basename(os.path.dirname(x))).isin(patients))
        ].index.tolist()
        
        ###### here #######
        # self.filtered_indices = self.filtered_indices[:2]

    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, index):
        sample = super().__getitem__(self.filtered_indices[index])
        image = sample["video"]
        # Convert to float32 and normalize to [0,1]
        image = image.astype('float32') / 255.0
        mask = torch.from_numpy(sample['mask']).long()
        
        mask_clipped = mask.clone()
        
        es_index = int(sample["end_systolic_frame_index"])
        ed_index = int(sample["end_diastolic_frame_index"])
        
        start_frame = min(es_index, ed_index)
        end_frame = max(es_index, ed_index)

        indices = list(range(start_frame, end_frame + 1))
        
        ef = torch.tensor(sample["ejection_fraction"], dtype=torch.float32)
        
        
        # Resize each frame to 256x256
        B, H, W = image.shape
        resized_images = []
        resized_masks = []
        # if B >= 32:
        #     indices  = list(range(0, 32, 2))  # [0, 2, 4, ..., 30]
        # else:
        #     indices = list(range(0, B, 1))  
        #     indices = indices[:16]
        for idx in indices:
            resized_img = cv2.resize(image[idx], (256, 256), interpolation=cv2.INTER_LINEAR)
            resized_images.append(torch.from_numpy(resized_img))
            
            resized_mask = cv2.resize(
                mask_clipped[idx].numpy(),  # Convert tensor to numpy
                (256, 256),
                interpolation=cv2.INTER_NEAREST
            )
            resized_masks.append(torch.from_numpy(resized_mask))

        # Pad if fewer than 16 frames
        while len(resized_images) < 16:
            resized_images.append(resized_images[-1].clone())
            resized_masks.append(resized_masks[-1].clone())
        # Stack and add channel dim: (B, 1, 256, 256)
        
        image = torch.stack(resized_images).unsqueeze(1) 
        mask_resized = torch.stack(resized_masks).long()

        # Convert to one-hot: shape (B, C, H, W)
        target = F.one_hot(mask_resized, num_classes=self.num_classes)  # shape (B, H, W, C)
        target = target.permute(0, 3, 1, 2).float()   
        
        # swap channel 2 and 3
        target[:, [2, 3]] = target[:, [3, 2]]
        # zero out myo channel 
        target[:, 3, :, :] = 0
        
        # dx = sample["spacing"][0] # spacing along x-axis
        # dy = sample["spacing"][1]  # spacing along y-axis
        
        ef_seg, edv, esv = calc_ef_from_seg(target[0,1:2,:,:], target[-1,1:2,:,:])
        
        orig_height, orig_width = H, W
        new_height, new_width = 256, 256
        # scaled_dX =  dx * orig_width / new_width
        # scaled_dY =  dy * orig_height / new_height
        labels_name = ["LV", "LA", "MYO"]
        print(f"ef_seg: {ef_seg}")
        if self.transforms is not None:
            image = self.transforms(image)
        return {
            "images": image,
            "masks": target,
            "pred_view": sample["cardiac_view"],
            "edv":edv,
            "esv":esv,
            "ef_numeric":ef,
            "ef_visual": ef_seg
            } 


class EchoNet_Dynamic(echonet.EchoNetDataset):
    """
    Returns resized AP4 video frames (values normalized 0-1) 2 frames per sample ed, es
    with corresponding LV labels.

    Output shapes:
    - Video frames: (T, C=1, H, W)
    - Myocardium labels: (T, H, W)
    """
    
    def __init__(self, split="test"):
        super().__init__()
        self.split = split
        metadata = echonet.get_metadata()
        self.filtered_indices = metadata[metadata["split"] == self.split ].index.tolist()
    
    def __len__(self):
        return len(self.filtered_indices)
    
    
    def __getitem__(self, index):
        sample = super().__getitem__(self.filtered_indices[index])
        # (T, C, H, W)
        video = sample["video"]
        ED_index, ES_index = sample["end_diastolic_frame_index"], sample["end_systolic_frame_index"]
        ED_mask,  ES_mask = sample['end_diastolic_mask'], sample['end_systolic_mask']
        
        ED_mask_resized = cv2.resize(ED_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        ES_mask_resized = cv2.resize(ES_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        prediction_video_mask = np.stack([ED_mask_resized, ES_mask_resized], axis=0)
        prediction_video_mask = torch.from_numpy(prediction_video_mask).float()
        
        # prediction_video_mask = np.stack([ED_mask, ES_mask], axis=0)
        # prediction_video_mask = torch.from_numpy(prediction_video_mask).float()
        
        selected_video = []
        for frame_index in [ED_index, ES_index]:  
            frame = video[frame_index,:, :,:]
            print(frame.shape)
            if frame.ndim == 3 and frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0))  # (3, H, W) >(H, W, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
            frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR) 
            frame = frame / 255.0
            frame = np.expand_dims(frame, axis=0)
            selected_video.append(frame)
        
        selected_video = np.stack(selected_video, axis=0)
        selected_video = torch.from_numpy(selected_video).float()
        prediction_video_mask = F.one_hot(prediction_video_mask.long(), num_classes=2)  # (2, 112, 112, 2)
        prediction_video_mask = prediction_video_mask.permute(0, 3, 1, 2).float()  # (2, 2, 112, 112)

        print(f" in dataset: {selected_video.shape, prediction_video_mask.shape, type(selected_video[0])}")
        # return selected_video, prediction_video_mask
        labels_name = ["LV"]
        
        edv = torch.tensor(sample["end_diastolic_volume"], dtype=torch.float32)
        esv = torch.tensor(sample["end_systolic_volume"], dtype=torch.float32)
        ef =  torch.tensor(sample["ejection_fraction"], dtype=torch.float32)
        ef_seg, _, _ = calc_ef_from_seg(prediction_video_mask[0,1:2,:,:], prediction_video_mask[-1,1:2,:,:])
    
        return {
            "images": selected_video,
            "masks": prediction_video_mask,
            "pred_view": "AP4",
            "edv":edv,
            "esv":esv,
            "ef_numeric":ef,
            "ef_visual": ef_seg
            } 

    

class Video_Camus(camus.CAMUSDataset):
    
    def __init__(self, split = "train", transforms=None):
        super().__init__()
        
        self.transforms = transforms
        self.view_list=["AP4", "AP2"]
        # self.view = view
        self.num_classes = 4 
        self.split = split
        metadata = camus.get_metadata()
        if self.split  == "test":
            patients = np.loadtxt(
                "/home/bassant/data/camus/camus_test/subgroup_testing.txt",
                dtype=str
            )
        elif self.split  =="train":
            patients = np.loadtxt(
                "/home/bassant/data/camus/camus_test/subgroup_training.txt",
                dtype=str
            )
        elif self.split  =="val":
            patients = np.loadtxt(
                "/home/bassant/data/camus/camus_test/subgroup_validation.txt",
                dtype=str
            )
        # Filter for specific cardiac views (e.g., 4CH views)
        self.filtered_indices = metadata[
            (metadata["cardiac_view"].isin(self.view_list)) &
            (metadata["file_path"].apply(lambda x: os.path.basename(os.path.dirname(x))).isin(patients))
        ].index.tolist()
        # self.filtered_indices = self.filtered_indices[:2]

    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, index):
        sample = super().__getitem__(self.filtered_indices[index])
        image = sample["video"]
        # Convert to float32 and normalize to [0,1]
        image = image.astype('float32') / 255.0
        mask = torch.from_numpy(sample['mask']).long()
        
        mask_clipped = mask.clone()
        # print(np.unique(mask_clipped))

        # print("Mask min:", mask_clipped.min().item())
        # print("Mask max:", mask_clipped.max().item())
        
        # Resize each frame to 256x256
        B, H, W = image.shape
        resized_images = []
        resized_masks = []
        for b in range(B):
            resized_img = cv2.resize(image[b], (256, 256), interpolation=cv2.INTER_LINEAR)
            resized_images.append(torch.from_numpy(resized_img))
            
            resized_mask = cv2.resize(
                mask_clipped[b].numpy(),  # Convert tensor to numpy
                (256, 256),
                interpolation=cv2.INTER_NEAREST
            )
            resized_masks.append(torch.from_numpy(resized_mask))


        
        image = torch.stack(resized_images).unsqueeze(1) 
        mask_resized = torch.stack(resized_masks).long()

        # Convert to one-hot: shape (B, C, H, W)
        target = F.one_hot(mask_resized, num_classes=self.num_classes)  # shape (B, H, W, C)
        target = target.permute(0, 3, 1, 2).float()   
        
        # swap channel 2 and 3
        target[:, [2, 3]] = target[:, [3, 2]]
        
        if self.transforms is not None:
            image = self.transforms(image)
        dx = sample["spacing"][0] # spacing along x-axis
        dy = sample["spacing"][1]  # spacing along y-axis
        
        es_index = int(sample["end_systolic_frame_index"])
        ed_index = int(sample["end_diastolic_frame_index"])
        
        orig_height, orig_width = H, W
        new_height, new_width = 256, 256
        scaled_dX =  dx * orig_width / new_width
        scaled_dY =  dy * orig_height / new_height
        
        labels_name = ["LV", "LA", "MYO"]
        # if self.transforms is not None:
        #     image = self.transforms(image)
        return {
            "images": image,
            "masks": target,
            "spacing": torch.tensor([scaled_dX, scaled_dY]),
            "pred_view": sample["cardiac_view"],
            # "ef": torch.tensor(sample["ejection_fraction"], dtype=torch.float32),
            # "ed_index": torch.tensor(ed_index),
            # "es_index": torch.tensor(es_index),
            "labels_name": labels_name,
            "stream_id" : torch.tensor(-1)
            } 




class HMCQU_Dataset():
    """
    Returns resized AP4 video frames (values normalized 0-1, total frames: 2190)
    with corresponding myocardium labels.

    Output shapes:
    - Video frames: (T, C=1, H, W)
    - Myocardium labels: (T, C=1, H, W)
    """
    def __init__(self, data_dir="/data/project/users/bassant/HMC-QU/processed", W=224, H=224):
        self.data_dir = data_dir
        self.data_mat_files_names = os.listdir(self.data_dir)
        self.files_paths = [os.path.join(self.data_dir, f) for f in self.data_mat_files_names]
        self.W,self.H = W, H
    
    def __len__(self):
        return len(self.data_mat_files_names)
    
    def __getitem__(self, index):
        mat_data = loadmat(self.files_paths[index], simplify_cells=True)
        allData1 = mat_data['allData1']
        start_index = int(allData1['ReF'])
        end_index = int(allData1['EOC'])
        video_data = allData1['cleaned_vidframes'][:, :, :, start_index:end_index]
        resized_video_data = []
        
        for t in range(video_data.shape[3]):
            frame = video_data[:, :, :, t]
            # Convert to single channel if not already
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize
            frame = cv2.resize(frame, (self.W, self.H)) / 255.0
            # Add channel dimension
            frame = np.expand_dims(frame, axis=0)  # shape: (1, H, W)
            resized_video_data.append(frame)
        
        # Stack frames along T dimension
        resized_video_data = np.stack(resized_video_data, axis=0)  # shape: (T, 1, H, W)
        # Convert to torch tensor
        resized_video_data = torch.from_numpy(resized_video_data).float()

        # prediction mask (T, H, W) -> add channel dim if needed
        prediction_video_mask = allData1['predicted'][start_index:end_index, :, :]
        prediction_video_mask = np.expand_dims(prediction_video_mask, axis=1)  # (T, 1, H, W)
        prediction_video_mask = torch.from_numpy(prediction_video_mask).float()
        print(resized_video_data.shape, prediction_video_mask.shape)
        return resized_video_data, prediction_video_mask

  


if __name__ =="__main__":
    
    camus_test_dataset = camus_test()   
    sample = camus_test_dataset[0]
    print(sample)
# transforms (resize)