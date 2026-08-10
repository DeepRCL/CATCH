# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
from torch import nn
from monai.losses import  DiceLoss, HausdorffDTLoss
from scipy.ndimage import label, center_of_mass

class MultiSegmentationLoss(nn.Module):
    """
    Combine different losses used in segmentation.
    """

    def __init__(self):
        super(MultiSegmentationLoss, self).__init__()

        # class_weights = torch.tensor(
        #     [1., 1., 2., 1., 1.],
        #     dtype=torch.float32
        # )

        # self.register_buffer("class_weights", class_weights)
        
        self.dense_loss_fn = DiceLoss(
            include_background =True,
            to_onehot_y=False,
            softmax=True, 
            reduction='none',
            # weight = self.class_weights
        )
        

        # self.dense_ce_loss_fn = torch.nn.CrossEntropyLoss(
        #     ignore_index=0,
        #     reduction='none'   # IMPORTANT
        # )

        # here 
        # self.register_buffer("dense_classes", torch.tensor([0,1,2,3,4]))

        self.register_buffer("dense_classes", torch.tensor([0,1,2,3]))

    def _compute_dense_loss(self, pred, gt):
        # print(pred.shape, gt.shape)
        # here 
        print("pred shape:", pred.shape)
        print("gt shape:", gt.shape)
        print("pred channels:", pred.shape[1])
        
        dense_loss = self.dense_loss_fn(pred, gt)  # [B, C, H, W]
        dense_loss = dense_loss.mean(dim=(2, 3))    # reduce over H, W → [B, C]
        label_present = (gt.sum(dim=(2,3)) > 0).float() # [B, C] 
        label_present[:, 0] = False
        dense_loss = dense_loss[:, 1:]  # shape: [B, C-1]
        label_present_fg = label_present[:, 1:]  # shape: [B, C-1]
        
        
        # Apply mask first, then weights
        dense_loss = dense_loss * label_present_fg    
        
        
        # Per-sample aggregation (only valid classes)
        dense_loss_per_sample = (
            dense_loss.sum(dim=1) /
            label_present_fg.sum(dim=1).clamp_min(1)
        )
        
        
                
        hausdorf_loss = self.hausdorf_loss_fn(pred, gt)  # [B, C, H, W]
        hausdorf_loss = hausdorf_loss.mean(dim=(2, 3))    # reduce over H, W → [B, C]
        hausdorf_loss = hausdorf_loss[:, 1:]  # shape: [B, C-1]
        
        
        # Apply mask first, then weights
        hausdorf_loss = hausdorf_loss * label_present_fg    
        
        
        # Per-sample aggregation (only valid classes)
        hausdorf_loss_per_sample = (
            hausdorf_loss.sum(dim=1) /
            label_present_fg.sum(dim=1).clamp_min(1)
        )



        
        final_loss =  hausdorf_loss_per_sample.mean() + dense_loss_per_sample.mean()
        return final_loss,  dense_loss_per_sample.mean(), hausdorf_loss_per_sample.mean()
    
    def forward(self, pred, gt):
        """Forward function."""
        gt = gt.float()
        pred = pred.float()
        assert gt.min() >= 0
        dense_pred = pred[:,self.dense_classes, ...]
        dense_gt = gt[:,self.dense_classes, ...]
    
        
        # final_dense_loss, ce_dense, dice_dense = self._compute_dense_loss(dense_pred, dense_gt)
        
        final_dense_loss, dice_dense, hausdorf_dense = self._compute_dense_loss(dense_pred, dense_gt)

        
        # lambda_dense = 1.0
        # total_loss = lambda_dense * final_dense_loss 

        return {
            "dense_hausdorf_loss": hausdorf_dense,
            "dense_dice_loss": dice_dense,
            "total_loss": final_dense_loss,
        }



def get_batch_centers(mask):
    points = torch.zeros((mask.shape[0], mask.shape[1], 2, 2))
    for sample_id in range(mask.shape[0]):
        for channel in range(mask.shape[1]):
            channel_mask = mask[sample_id, channel, :, :]
            if torch.sum(channel_mask) == 0:
                continue
            channel_mask = channel_mask.detach().cpu().numpy().astype(int)
            labeled_mask, num_features = label(channel_mask)
            # Get centers of each circle
            # y, x format
            centers = center_of_mass(channel_mask, labeled_mask, range(1, num_features+1))
            centers = centers[:2]
            # print(f"Sample {sample_id}, Channel {channel}, Centers: {centers}, centers shape: {len(centers)}")
            if len(centers) == 0:
                continue
            points[sample_id, channel, :, 0] = torch.tensor([c[1] for c in centers])
            points[sample_id, channel, :, 1] = torch.tensor([c[0] for c in centers])
    return points
    



# class MultiSegmentationLoss_Measurements(nn.Module):
#     """
#     Combine different losses used in segmentation.
#     """

#     def __init__(self):
#         super(MultiSegmentationLoss_Measurements, self).__init__()

#         self.linear_loss_fn = DiceLoss(
#             include_background =False,
#             to_onehot_y=False,
#             sigmoid=True, 
#             reduction='none',        
#             )
#         self.linear_bce_loss_fn = torch.nn.BCEWithLogitsLoss(
#             reduction='none'
#         )

#         # self.linear_measurement_loss_fn = torch.nn.L1Loss()
#         self.l2_loss_fn = nn.MSELoss(reduction='none')

        
#         self.dense_loss_fn = DiceLoss(
#             include_background =True,
#             to_onehot_y=False,
#             softmax=True, 
#             reduction='none',
#             # weight = torch.tensor([1., 1., 1., 1., 4.])
#         )
#         # self.dense_ce_loss_fn = torch.nn.CrossEntropyLoss(
#         #     ignore_index=0,
#         #     reduction='none'   # IMPORTANT
#         # )

#         # self.register_buffer("dense_classes", torch.tensor([0,1,2,3]))
#         self.register_buffer("dense_classes", torch.tensor([0,1,2,3,4]))
#         self.register_buffer("linear_classes", torch.tensor([0,5,6,7,8,9,10,11,12]))

#     def get_centers_from_mask(self, mask, eps=1e-6):
#         """
#         mask: (B, C, H,W)
#         Return: centers: (B, C, 2)  (x, y)
#         """
#         B, C, H, W = mask.shape
#         device = mask.device
        
#         y_coords = torch.arange(H, device=device).view(1, 1, H, 1)
#         x_coords = torch.arange(W, device=device).view(1, 1, 1, W)
        
#         mass = mask.sum(dim=(2,3), keepdim=True).clamp_min(eps)
        
#         cx = (mask * x_coords).sum(dim=(2, 3), keepdim=True) / mass
#         cy = (mask * y_coords).sum(dim=(2, 3), keepdim=True) / mass
        
#         # labeled_mask, num_features = label(mask)
#         # centers = center_of_mass(mask, labeled_mask, range(1, num_features+1))
#         centers = torch.cat([cx, cy], dim=-1)  # [B, C, 1, 2]
#         return centers.squeeze(2)      # [B, C, 2]
   
        
#     def _compute_linear_seg_loss(self, pred, gt):
#         gt_bin = (gt > 0).float()
#         # print(pred.shape, gt_bin.shape)
#         linear_loss = self.linear_loss_fn(pred,gt_bin)
#         linear_loss = linear_loss.mean(dim=(2, 3))  # reduce over H, W → [B, C]
#         label_present = (gt.sum(dim=(2,3)) > 0).float() # [B, C]
#         label_present_fg = label_present[:, 1:]  # shape: [B, C-1]
  
        
#         # Apply mask first, then weights
#         linear_loss = linear_loss * label_present_fg        
        
#         # Per-sample aggregation (only valid classes)
#         loss_per_sample = (
#             linear_loss.sum(dim=1) /
#             label_present_fg.sum(dim=1).clamp_min(1)
#         )
        
#         linear_bce_loss = self.linear_bce_loss_fn(pred,gt_bin) # [B,C,H,W]
#         # print(f"linear_bce_loss shape: {linear_bce_loss.shape}")
#         # linear_bce_loss = linear_bce_loss[:, 1, :, :] 
#         # linear_bce_loss = linear_bce_loss.mean(dim=(2, 3))  # reduce over H, W → [B, C]
#         linear_bce_loss = linear_bce_loss[:, 1:2, :, :]  # keep C dim
#         linear_bce_loss = linear_bce_loss.mean(dim=(2, 3))  # → [B, 1]

#         linear_bce_loss = linear_bce_loss * label_present_fg    
        
#         # Per-sample aggregation (only valid classes)
#         bce_loss_per_sample = (
#             linear_bce_loss.sum(dim=1) /
#             label_present_fg.sum(dim=1).clamp_min(1)
#         )
        
#         final_linear_seg_loss =  bce_loss_per_sample.mean() +loss_per_sample.mean()
        

#         return final_linear_seg_loss , bce_loss_per_sample.mean() , loss_per_sample.mean()


#     # def linear_measurement_loss_fn(self, pred_centers, gt_centers):
#     #     return torch.norm(pred_centers - gt_centers, dim=-1)
#     #     # → [B, C]


#     # output shape: (B, C, N_POINTS, (x,y))


#     def _compute_linear_measurement_loss(self, pred, gt, gt_coords):
#         device = pred.device
#         gt = gt[:, 1:, ...]  # exclude background
#         pred = pred[:, 1:, ...]
#         gt_coords = gt_coords[:, 1:, ...]  # exclude background
        
#         gt_bin = (gt > 0).float()
#         filtered_pred = pred * gt_bin
#         # print(filtered_pred.shape, gt_bin.shape) # B , C, 
        
#         channel_pixel_sums = filtered_pred.sum(dim=[2, 3])  # shape: [batch, channels]

        
#         if torch.sum(filtered_pred) == 0:
#             print("There is no foreground pixel in the batch for linear measurement loss.")
#             final_linear_measurement_loss = torch.tensor(0.0, device=pred.device)
#             return final_linear_measurement_loss
        
#         pred_batch_centers = get_batch_centers(filtered_pred)
#         pred_batch_centers = pred_batch_centers.to(device)
#         gt_coords = gt_coords.to(device)
        
#         # gt_centers = self.get_centers_from_mask(gt_bin)
#         # pred_centers = self.get_centers_from_mask(filtered_pred)
    
#         # linear_loss = self.linear_measurement_loss_fn(pred_centers,gt_centers)
#         print(f"pred_batch_centers shape: {pred_batch_centers.shape}, gt_coords shape: {gt_coords.shape}")
#         print(f"pred_batch_centers : {pred_batch_centers}, gt_coords shape: {gt_coords}")

#         linear_loss = self.l2_loss_fn(pred_batch_centers, gt_coords)
#         linear_loss = linear_loss.mean(dim=(2, 3))  # reduce over two points → [B, C]
#         print(f"linear_loss shape: {linear_loss.shape}")
  
#         label_present_fg = (gt.sum(dim=(2,3)) > 0).float() # [B, C]
#         # linear_loss = linear_loss[:, 1:]  # shape: [B, C-1]
#         # label_present_fg = label_present[:, 1:]  # shape: [B, C-1]
#         # Apply mask first, then weights
#         linear_loss = linear_loss * label_present_fg        
        
#         # Per-sample aggregation (only valid classes)
#         loss_per_sample = (
#             linear_loss.sum(dim=1) /
#             label_present_fg.sum(dim=1).clamp_min(1)
#         )
        
#         print(f"linear_measurement_loss_fn shape: {linear_loss.shape}")

#         final_linear_measurement_loss = loss_per_sample.mean()
        

#         return final_linear_measurement_loss
        
#     def _compute_dense_loss(self, pred, gt):
#         # print(pred.shape, gt.shape)
#         dense_loss = self.dense_loss_fn(pred, gt)  # [B, C, H, W]
#         dense_loss = dense_loss.mean(dim=(2, 3))    # reduce over H, W → [B, C]
#         label_present = (gt.sum(dim=(2,3)) > 0).float() # [B, C] 
#         label_present[:, 0] = False
#         dense_loss = dense_loss[:, 1:]  # shape: [B, C-1]
#         label_present_fg = label_present[:, 1:]  # shape: [B, C-1]
        
        
#         # Apply mask first, then weights
#         dense_loss = dense_loss * label_present_fg    
        
        
#         # Per-sample aggregation (only valid classes)
#         dense_loss_per_sample = (
#             dense_loss.sum(dim=1) /
#             label_present_fg.sum(dim=1).clamp_min(1)
#         )

#         # calculate ce loss 
#         B, C, H, W = pred.shape
#         target = gt.argmax(dim=1)  # [B, H, W]

#         # ce_loss = self.dense_ce_loss_fn(pred, target) # [B, H, W]
#         # print(f"ce_loss shape: {ce_loss.shape}")
#         # valid_mask = label_present.gather(1, target.view(B, -1))  # [B, H*W]
#         # valid_mask = valid_mask.view(B, H, W)                     # reshape back
#         # valid_mask = valid_mask.bool()                             # ensure bool type

        
#         # final_ce_loss = (ce_loss[valid_mask]).mean()
#         # final_loss =  final_ce_loss + dense_loss_per_sample.mean()
#         final_loss =  dense_loss_per_sample.mean()
#         return final_loss, dense_loss_per_sample.mean()

#         # return final_loss, final_ce_loss, dense_loss_per_sample.mean()

    
#     def forward(self, pred, gt, gt_coords):
#         """Forward function."""
#         gt = gt.float()
#         pred = pred.float()
#         assert gt.min() >= 0
#         dense_pred = pred[:,self.dense_classes, ...]
#         dense_gt = gt[:,self.dense_classes, ...]
        
        
#         linear_pred = pred[:,self.linear_classes, ...]
#         linear_gt = gt[:,self.linear_classes, ...]
#         gt_coords = gt_coords[:,self.linear_classes, ...]
        
#         final_dense_loss, ce_dense, dice_dense = self._compute_dense_loss(dense_pred, dense_gt)
        
#         final_linear_seg_loss, bce_linear, dice_linear = self._compute_linear_seg_loss(linear_pred,linear_gt)
        
#         linear_measurement_loss = self._compute_linear_measurement_loss(linear_pred,linear_gt, gt_coords)
        
#         # linear_measurement_loss = 
#         lambda_dense = 2.0
#         lambda_linear_seg = 4.0   # linear loss weighted more
#         lambda_linear_measurement = 1.0

#         if linear_measurement_loss.item() == 0.0:
#             total_loss = lambda_dense * final_dense_loss + lambda_linear_seg * final_linear_seg_loss
#         else:  
#             total_loss = lambda_dense * final_dense_loss + lambda_linear_seg * final_linear_seg_loss + lambda_linear_measurement * linear_measurement_loss

#         return {
#             "dense_dice_ce_loss": final_dense_loss,
#             "dense_ce_loss": ce_dense,
#             "dense_dice_loss": dice_dense,
#             "linear_dice_ce_loss": final_linear_seg_loss,
#             "linear_bce_loss": bce_linear,
#             "linear_dice_loss": dice_linear,
#             "l2_linear_loss":linear_measurement_loss,
#             "total_loss": total_loss,
#         }
