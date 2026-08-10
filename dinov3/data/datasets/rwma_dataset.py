import os
import random

import cv2
import torch
import numpy as np
from scipy.io import loadmat
from scipy import interpolate
import pandas as pd
# import torchvision.transforms.v2 as T
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
import torchvision.transforms.functional as T
# import imgaug.augmenters as iaa


if __name__ == "__main__":
    import sys
    sys.path.append('.')

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

# def apply_transform(image, mask):
#     if random.random() < 0.5:
#         image = T.hflip(image)
#         mask = T.hflip(mask)

#     angle = random.uniform(-15, 15)

#     image = T.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
#     mask = T.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

#     return image, mask

# seq = iaa.Sequential([
#     iaa.ElasticTransformation(alpha=50, sigma=5),
#     iaa.Affine(rotate=(-30, 30)),
#     iaa.Fliplr(0.4),
# ], random_order=True)

class PrivateRWMADataset(torch.utils.data.Dataset):
    def __init__(self,
                ann_csv_file: str= "/data/project/users/bassant/code/mulit_view_data_analysis/private_data_with_seg_label.csv", 
                split: str="test",
                ):
    
    
        self.ann_csv_file = ann_csv_file
        self.split = split
        
        self.metadata = self.get_metadata()
        
        self.ann_data = pd.read_csv(self.ann_csv_file)
        # self.ann_data = self.ann_data[:5]
        self.ann_data = self.ann_data[self.ann_data["split"] == self.split]
        # self.ann_data  = self.ann_data [self.ann_data["view"] == "AP4"]

        print(f"len of {self.split} rmwa data:{len(self.ann_data)}")
        self.labels = [2]
        self.corrupted_files = []
        # if self.split =="train":
        #     self.transform = T.Compose([
        #                 T.RandomHorizontalFlip(p=0.5),
        #                 T.RandomRotation(degrees=(-15, 15), interpolation=InterpolationMode.NEAREST),
        #             ])
        # else:
        #     self.transform = None
    def get_metadata(self):
        metadata = pd.read_csv(self.ann_csv_file)
        return metadata
    
    # def get_index(self, filename):
    #     return self.metadata[self.metadata.file_path.str.contains(filename)].index.item()
    
    def get_edge_points(self, inner_data, outer_data):
        inner_wall_coords = inner_data['coords_cropped']
        outer_wall_coords = outer_data['coords_cropped']

        # if outer_wall_coords.shape[0] != 7 or inner_wall_coords.shape[0] != 7:
        #     return None, None, None
        
        inner_reversed = False
        if inner_wall_coords[0][0] < inner_wall_coords[-1][0]:
            inner_wall_coords = np.flip(inner_wall_coords, axis=0)
            inner_reversed = True
        
        outer_reversed = False
        if outer_wall_coords[0][0] < outer_wall_coords[-1][0]:
            outer_wall_coords = np.flip(outer_wall_coords, axis=0)
            outer_reversed = True
            

        points = np.concatenate([outer_wall_coords, inner_wall_coords])
        
        return points, inner_reversed, outer_reversed
    
    def extract_all_info(self, seg_data, height, width): 
        if seg_data[0]['type'] == "LV inner wall":
            inner_data = seg_data[0]
            outer_data = seg_data[1]
        else:
            inner_data = seg_data[1]
            outer_data = seg_data[0]
            
        frame_num = inner_data['frame_num']
                
        points, inner_points_reversed, outer_points_reversed = self.get_edge_points(inner_data=inner_data, outer_data=outer_data)
        if points is None:
            return None, None, None, None, None
        
        in_frame_bool = ((points[:, 0] >= 0) & (points[:, 0] < width) & \
            (points[:, 1] >= 0) & (points[:, 1] < height)).all()
        
        # if not in_frame_bool:
        #     return None, None, None, None, None
        
        
        mask = generate_gt_mask_for_rwma(points=points, width=width, height=height, mask_labels=self.labels)
        return inner_points_reversed, outer_points_reversed, points, frame_num, mask
    

    
    def process_data(self, fp):
        no_seg_data = 0
        inner_data_reversed = 0
        outer_data_reversed = 0

        # for fp in tqdm(all_mat_filepaths[::-1]):
        mat_data = loadmat(fp, simplify_cells=True)
        dX = mat_data["dX"]
        dY = mat_data["dY"]
        
        try:
            all_segs = mat_data['labels']['segmentations']
            video = mat_data["cropped"]
            # print("Video shape:", video.shape)
            
            ED_data = [item for item in all_segs if item.get('keyframe') == 'ED']
            if len(ED_data) > 0:
                inner_reversed, outer_reversed, ED_points, ED_frame_num, ED_mask = self.extract_all_info(ED_data, height=video.shape[0], width=video.shape[1])
                if inner_reversed:
                    inner_data_reversed += 1
                    
                if outer_reversed:
                    outer_data_reversed += 1
            
            # ES_data = [item for item in all_segs if item.get('keyframe') == 'ES']
            # if len(ES_data) > 0:
            #     inner_reversed, outer_reversed, ES_points, ES_frame_num, ES_mask = self.extract_all_info(ES_data, height=video.shape[0], width=video.shape[1])
            #     if inner_reversed:
            #         inner_data_reversed += 1
                    
            #     if outer_reversed:
            #         outer_data_reversed += 1

            # valid_frames = [f for f in [ES_frame_num, ED_frame_num] if f is not None]
            # if len(valid_frames) == 0:
            #     self.corrupted_files.append(fp)    
            return video, ED_frame_num, ED_mask, dX, dY
            
        except Exception as e:
            print(fp)
            print(e)
            # self.corrupted_files.append(fp)
            # no_seg_data += 1
    def __len__(self):
        return len(self.ann_data)
    
    def get_corrupted_files(self):
        for index, row in self.ann_data.iterrows():
            data_path =  "/data/project/users/bassant/Private_myo_data/curved_all"
            filepath = row["file_path"]
            mat_filepath = os.path.join(data_path, filepath)
            self.process_data(mat_filepath)
        
            # valid_frames = [f for f in [ES_frame_num, ED_frame_num] if f is not None]
            # if len(valid_frames) == 0:
            #     self.corrupted_files.append(filepath)
        return self.corrupted_files
    
    def __getitem__(self, idx):
        row = self.ann_data.iloc[idx]
        filepath = row["file_path"]
        data_path =  "/data/project/users/bassant/Private_myo_data/curved_all"
        mat_filepath = os.path.join(data_path, filepath)

        
        # (H,W T)
        video, ED_frame_num, ED_mask, dX, dY = self.process_data(mat_filepath)


        frame_idx = ED_frame_num
        video = np.expand_dims(video, axis=0)


        orig_height, orig_width = video.shape[1], video.shape[2]
        new_height, new_width = 256, 256
    
        video_frames = []
        frame = video[0, :, :, frame_idx]
        frame_resized = cv2.resize(frame, (new_width, new_height) ,interpolation = cv2.INTER_NEAREST)[None, ...]
        video_frames.append(frame_resized)
        video = np.array(video_frames)
        
        T, C, H, W = video.shape
        
        masks = torch.zeros((T, 256, 256))                   
        if ED_mask is not None:
            ED_mask = cv2.resize(ED_mask, (new_height, new_width), interpolation=cv2.INTER_NEAREST)
            ED_mask = (ED_mask > 0.9).astype(np.float32)
            masks[0] = torch.from_numpy(ED_mask)
        else:
            self.corrupted_files.append(filepath)

        images = torch.from_numpy(video).to(torch.float32) / 255.0
        
        # print(f"in rwma data set: images {images.shape}, masks: {masks.shape}")
        # shape should be (c, h, w) (c, h,w)
        images = images.squeeze(0)
        
        # prob = random.random()
        # if prob > 0.5 and self.split == "train":
        #     images, masks = seq(image=images, segmentation_maps=masks)
        if self.split =="train"  is not None and random.random() < 0.3:
            images, masks = apply_transform(images, masks)


        # img = images[0].detach().cpu().numpy()
        # mask = masks[0].detach().cpu().numpy()

        # plt.figure()
        # plt.imshow(img, cmap="gray")
        # plt.imshow(mask, alpha=0.5)
        # plt.axis("off")

        # plt.savefig("debug_rmwa_transform_pb_0.3.png", bbox_inches="tight", dpi=200)
        # plt.close()
        
        scaled_dX = dX * orig_width / new_width
        scaled_dY = dY * orig_height / new_height
        images = images.unsqueeze(0)
        # print(f"in rwma data set: images {images.shape}, masks: {masks.shape}")
        targets = {
            "images": images, 
            "spacing": torch.tensor([scaled_dX, scaled_dY]),
            "masks": masks,
            "pred_view": row["view"], 
            "labels_name": "myo", 
            "stream_id": torch.tensor(-1),
        }
        
        return targets
    

def create_curved_polygon(coords , width=-1, height=-1, npoints=50):
        # create a binary image
        from PIL import Image
        data = Image.new(mode='L', size=(width, height), color=0)  # mode L = 8-bit pixels, black and white
        img = np.zeros((height,width, 3), np.uint8)
        coords_np = np.asarray(coords)
        new_coords = []
        x=coords_np[:,0]
        y=coords_np[:,1]
        value=np.nan
        
        tck,u = interpolate.splprep([x,y],k=3,s=0)
        u=np.linspace(0,1,num=npoints,endpoint=True)
        out = interpolate.splev(u,tck)
        red = [0,0,255]

        for i in range (len(out[0])):
            x1= int(out[0][i])
            y1 = int(out[1][i])
            new_coords.append((x1,y1))
            # img[ x1, y1] = red
            img[y1, x1] = red

        return new_coords
    
    
def generate_gt_mask_for_rwma(points, width, height, mask_labels=[1, 2]):
    """
    Generates a combined ground truth mask for LV and Myocardium.

    Args:
        points: np.ndarray (N, 2) — concatenation of outer + inner wall points.
        width, height: image dimensions.
        mask_labels: list[int] — which labels to include.
                     1 = LV cavity, 2 = Myocardium.

    Returns:
        gt_mask: np.ndarray of shape (H, W) with integer class labels.
    """
    npoints = points.shape[0]
    
    # Split into outer and inner walls
    outer_points = points[:npoints // 2]
    inner_points = points[npoints // 2:]
    
    # Generate smooth contours
    curved_outer = np.flip(np.array(create_curved_polygon(coords=outer_points, width=width, height=height)), axis=0)
    curved_inner = np.array(create_curved_polygon(coords=inner_points, width=width, height=height))
    
    # Initialize masks
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Temporary binary masks for convenience
    lv_mask  = cv2.fillPoly(np.zeros_like(gt_mask), [curved_inner], 1)
    myo_mask = cv2.fillPoly(np.zeros_like(gt_mask), [np.concatenate([curved_outer, curved_inner])], 1)
    
    # Assign labels based on mask_labels input
    if len(mask_labels) == 1:
        if 2 in mask_labels:
            gt_mask[myo_mask == 1] = 1
        elif 1 in mask_labels:
            gt_mask[lv_mask == 1] = 1
    else:
        if 2 in mask_labels:
            gt_mask[myo_mask == 1] = 2
        if 1 in mask_labels:
            gt_mask[lv_mask == 1] = 1
    
    return gt_mask

# import os
# import random

# import cv2
# import torch
# import numpy as np
# from scipy.io import loadmat
# from scipy import interpolate
# import pandas as pd

# if __name__ == "__main__":
#     import sys
#     sys.path.append('.')


# class PrivateRWMADataset(torch.utils.data.Dataset):
#     def __init__(self,
#                 ann_csv_file: str= "/data/project/users/bassant/code/code_from_b300/MV/info_f.csv", 
#                 split: str="test",
#                 ):
    
    
#         self.ann_csv_file = ann_csv_file
#         self.split = split
        
#         self.metadata = self.get_metadata()
        
#         self.ann_data = pd.read_csv(self.ann_csv_file)
#         # self.ann_data = self.ann_data[:5]
#         self.ann_data = self.ann_data[self.ann_data["split"] == self.split]

#         self.labels = [2]
#     def get_metadata(self):
#         metadata = pd.read_csv(self.ann_csv_file)
#         return metadata
    
#     # def get_index(self, filename):
#     #     return self.metadata[self.metadata.file_path.str.contains(filename)].index.item()
    
#     def get_edge_points(self, inner_data, outer_data):
#         inner_wall_coords = inner_data['coords_cropped']
#         outer_wall_coords = outer_data['coords_cropped']

#         if outer_wall_coords.shape[0] != 7 or inner_wall_coords.shape[0] != 7:
#             return None, None, None
        
#         inner_reversed = False
#         if inner_wall_coords[0][0] < inner_wall_coords[-1][0]:
#             inner_wall_coords = np.flip(inner_wall_coords, axis=0)
#             inner_reversed = True
        
#         outer_reversed = False
#         if outer_wall_coords[0][0] < outer_wall_coords[-1][0]:
#             outer_wall_coords = np.flip(outer_wall_coords, axis=0)
#             outer_reversed = True
            

#         points = np.concatenate([outer_wall_coords, inner_wall_coords])
        
#         return points, inner_reversed, outer_reversed
    
#     def extract_all_info(self, seg_data, height, width): 
#         if seg_data[0]['type'] == "LV inner wall":
#             inner_data = seg_data[0]
#             outer_data = seg_data[1]
#         else:
#             inner_data = seg_data[1]
#             outer_data = seg_data[0]
            
#         frame_num = inner_data['frame_num']
                
#         points, inner_points_reversed, outer_points_reversed = self.get_edge_points(inner_data=inner_data, outer_data=outer_data)
#         if points is None:
#             return None, None, None, None, None
        
#         in_frame_bool = ((points[:, 0] >= 0) & (points[:, 0] < width) & \
#             (points[:, 1] >= 0) & (points[:, 1] < height)).all()
        
#         if not in_frame_bool:
#             return None, None, None, None, None
        
        
#         mask = generate_gt_mask_for_rwma(points=points, width=width, height=height, mask_labels=self.labels)
#         return inner_points_reversed, outer_points_reversed, points, frame_num, mask
    
    
#     def process_data(self, fp):
#         no_seg_data = 0
#         inner_data_reversed = 0
#         outer_data_reversed = 0

#         # for fp in tqdm(all_mat_filepaths[::-1]):
#         mat_data = loadmat(fp, simplify_cells=True)
#         dX = mat_data["dX"]
#         dY = mat_data["dY"]
        

#         try:
#             video = mat_data["cropped"]
#             labels = mat_data["labels"]   # labels is a dict

#             all_segs = labels["segmentations"]

#             # ---------------- ED ----------------
#             ED_frame_num, ED_mask = None, None
#             ED_data = [item for item in all_segs if item.get("keyframe") == "ED"]

#             if len(ED_data) > 0:
#                 inner_reversed, outer_reversed, ED_points, ED_frame_num, ED_mask = \
#                     self.extract_all_info(
#                         ED_data,
#                         height=video.shape[0],
#                         width=video.shape[1],
#                     )

#                 if inner_reversed:
#                     inner_data_reversed += 1
#                 if outer_reversed:
#                     outer_data_reversed += 1

#             # ---------------- ES ----------------
#             ES_frame_num, ES_mask = None, None
#             ES_data = [item for item in all_segs if item.get("keyframe") == "ES"]

#             if len(ES_data) > 0:
#                 inner_reversed, outer_reversed, ES_points, ES_frame_num, ES_mask = \
#                     self.extract_all_info(
#                         ES_data,
#                         height=video.shape[0],
#                         width=video.shape[1],
#                     )

#                 if inner_reversed:
#                     inner_data_reversed += 1
#                 if outer_reversed:
#                     outer_data_reversed += 1

#             return video, ED_frame_num, ES_frame_num, ED_mask, ES_mask, dX, dY

#         except Exception as e:
#             print("Error in file:", fp)
#             print("Exception:", e)

#             if "labels" in mat_data:
#                 print("labels keys:", mat_data["labels"].keys())

#             return None     
#     def __len__(self):
#         return len(self.ann_data)
    
    
#     def __getitem__(self, idx):
#         row = self.ann_data.iloc[idx]
#         filepath = row["filepath"]
#         data_path =  "/data/project/users/bassant/Private_myo_data/curved_all"
#         mat_filepath = os.path.join(data_path, filepath)

#         # (H,W T)
#         video, ED_frame_num, ES_frame_num, ED_mask, ES_mask, dX, dY = self.process_data(mat_filepath)

        
#         # idx1, idx2 = min(ED_frame_num, ES_frame_num), max(ED_frame_num, ES_frame_num)
#         # data_index = self.get_index(filename=filepath)
        
#         video = np.expand_dims(video, axis=0)
#         # C, H, W, T
#         # print(f"video:{video.shape}")

#         orig_height, orig_width = video.shape[1], video.shape[2]
#         new_height, new_width = 256, 256
        
#         # video = np.array([
#         #     cv2.resize(
#         #         video[0, :, :, frame_idx],
#         #         (new_width, new_height)
#         #     )[None, ...]
#         #     for frame_idx in [ED_frame_num, ES_frame_num]
#         # ])
        
#         video_frames = []
#         for frame_idx in [ED_frame_num, ES_frame_num]:
#             if frame_idx is not None:
#             # Check if index is in range
#                 if frame_idx < 0 or frame_idx >= video.shape[3]:
#                     print(f"Warning: frame index {frame_idx} is out of range (0-{video.shape[3]-1})")
#                     # Optionally, skip or use a placeholder frame
#                     frame = video[0, :, :, 0]
#                 else:
#                     frame = video[0, :, :, frame_idx]
#                     if frame is None or not frame.any():
#                         print(f"Warning: frame at index {frame_idx} is empty")
#                         frame = video[0, :, :, 0]
#                         # frame = np.zeros((video.shape[1], video.shape[2]), dtype=video.dtype)
                
#                 # Resize and add channel dimension
#                 frame_resized = cv2.resize(frame, (new_width, new_height))[None, ...]
#                 video_frames.append(frame_resized)
#                 # frame: (np.float32(0.0), np.float32(1.0))
#                 # print(f"frame: {np.min(frame), np.max(frame)}")
#             # else:
#                 # print(f"frame_idx: {frame_idx}")
#         video = np.array(video_frames)
        
#         ED_mask = cv2.resize(ED_mask, (new_height, new_width), interpolation=cv2.INTER_NEAREST)
#         ES_mask = cv2.resize(ES_mask, (new_height, new_width), interpolation=cv2.INTER_NEAREST)


#         # dX and dY are in cm scale
#         scaled_dX =  dX * orig_width / new_width
#         scaled_dY =  dY * orig_height / new_height
#         # T = 2
#         T, C, H, W = video.shape
#         masks = torch.zeros((T, ED_mask.shape[0], ED_mask.shape[1]))
        
#         # if ED_frame_num < ES_frame_num:
#         #     masks[0] = torch.from_numpy(ED_mask)
#         #     masks[-1] = torch.from_numpy(ES_mask)
#         # else:
#         masks[-1] = torch.from_numpy(ES_mask)
#         masks[0] = torch.from_numpy(ED_mask)
#         # print(f"in rwma :{video.shape}, {masks.shape}")

#         targets = {
#             "images": torch.from_numpy(video).to(torch.float32) / 255.0, 
#             # "num_frames": T, 
#             "spacing": torch.tensor([scaled_dX, scaled_dY, 1.0]),
#             "masks": masks,
#             # "start_frame": idx1,
#             # "idx": data_index,
#             "pred_view": row["view"], 
#             "labels_name": "myo"
#         }
        
#         return targets
    

# def create_curved_polygon(coords , width=-1, height=-1, npoints=50):
#         # create a binary image
#         from PIL import Image
#         data = Image.new(mode='L', size=(width, height), color=0)  # mode L = 8-bit pixels, black and white
#         img = np.zeros((height,width, 3), np.uint8)
#         coords_np = np.asarray(coords)
#         new_coords = []
#         x=coords_np[:,0]
#         y=coords_np[:,1]
#         value=np.nan
        
#         tck,u = interpolate.splprep([x,y],k=3,s=0)
#         u=np.linspace(0,1,num=npoints,endpoint=True)
#         out = interpolate.splev(u,tck)
#         red = [0,0,255]

#         for i in range (len(out[0])):
#             x1= int(out[0][i])
#             y1 = int(out[1][i])
#             new_coords.append((x1,y1))
#             # img[ x1, y1] = red
#             img[y1, x1] = red

#         return new_coords
    
    
# def generate_gt_mask_for_rwma(points, width, height, mask_labels=[1, 2]):
#     """
#     Generates a combined ground truth mask for LV and Myocardium.

#     Args:
#         points: np.ndarray (N, 2) — concatenation of outer + inner wall points.
#         width, height: image dimensions.
#         mask_labels: list[int] — which labels to include.
#                      1 = LV cavity, 2 = Myocardium.

#     Returns:
#         gt_mask: np.ndarray of shape (H, W) with integer class labels.
#     """
#     npoints = points.shape[0]
    
#     # Split into outer and inner walls
#     outer_points = points[:npoints // 2]
#     inner_points = points[npoints // 2:]
    
#     # Generate smooth contours
#     curved_outer = np.flip(np.array(create_curved_polygon(coords=outer_points, width=width, height=height)), axis=0)
#     curved_inner = np.array(create_curved_polygon(coords=inner_points, width=width, height=height))
    
#     # Initialize masks
#     gt_mask = np.zeros((height, width), dtype=np.uint8)
    
#     # Temporary binary masks for convenience
#     lv_mask  = cv2.fillPoly(np.zeros_like(gt_mask), [curved_inner], 1)
#     myo_mask = cv2.fillPoly(np.zeros_like(gt_mask), [np.concatenate([curved_outer, curved_inner])], 1)
    
#     # Assign labels based on mask_labels input
#     if len(mask_labels) == 1:
#         if 2 in mask_labels:
#             gt_mask[myo_mask == 1] = 1
#         elif 1 in mask_labels:
#             gt_mask[lv_mask == 1] = 1
#     else:
#         if 2 in mask_labels:
#             gt_mask[myo_mask == 1] = 2
#         if 1 in mask_labels:
#             gt_mask[lv_mask == 1] = 1
    
#     return gt_mask