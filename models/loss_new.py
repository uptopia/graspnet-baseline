""" Loss functions for training.
    Author: chenxi-wang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from loss_utils import GRASP_MAX_WIDTH, GRASP_MAX_TOLERANCE, THRESH_GOOD, THRESH_BAD,\
                       transform_point_cloud, generate_grasp_views,\
                       batch_viewpoint_params_to_matrix, huber_loss

def get_loss(end_points, epoch, batch_idx):
    print("################# epoch = {}, batch_idx = {}, batch_size = {} #################".format(epoch, batch_idx, len(end_points['point_clouds'])))
    objectness_loss, end_points = compute_objectness_loss(end_points)
    view_loss, end_points = compute_view_loss(end_points)
    grasp_loss, end_points = compute_grasp_loss(end_points)
    manipulability_loss, end_points = compute_manipulability_loss(end_points)
    loss = objectness_loss + view_loss + 0.2 * grasp_loss + 0.5 * manipulability_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points

def compute_objectness_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    objectness_score = end_points['objectness_score']
    objectness_label = end_points['objectness_label']
    fp2_inds = end_points['fp2_inds'].long()
    objectness_label = torch.gather(objectness_label, 1, fp2_inds)
    loss = criterion(objectness_score, objectness_label)

    end_points['loss/stage1_objectness_loss'] = loss
    objectness_pred = torch.argmax(objectness_score, 1)
    end_points['stage1_objectness_acc'] = (objectness_pred == objectness_label.long()).float().mean()

    end_points['stage1_objectness_prec'] = (objectness_pred == objectness_label.long())[objectness_pred == 1].float().mean()
    end_points['stage1_objectness_recall'] = (objectness_pred == objectness_label.long())[objectness_label == 1].float().mean()

    return loss, end_points

def compute_view_loss(end_points):
    criterion = nn.MSELoss(reduction='none')
    view_score = end_points['view_score']
    view_label = end_points['batch_grasp_view_label']
    objectness_label = end_points['objectness_label']
    fp2_inds = end_points['fp2_inds'].long()
    V = view_label.size(2)
    objectness_label = torch.gather(objectness_label, 1, fp2_inds)

    objectness_mask = (objectness_label > 0)
    objectness_mask = objectness_mask.unsqueeze(-1).repeat(1, 1, V)
    pos_view_pred_mask = ((view_score >= THRESH_GOOD) & objectness_mask)

    loss = criterion(view_score, view_label)
    loss = loss[objectness_mask].mean()

    end_points['loss/stage1_view_loss'] = loss
    end_points['stage1_pos_view_pred_count'] = pos_view_pred_mask.long().sum()

    return loss, end_points


def compute_grasp_loss(end_points, use_template_in_training=True):
    top_view_inds = end_points['grasp_top_view_inds'] # (B, Ns)
    vp_rot = end_points['grasp_top_view_rot'] # (B, Ns, view_factor, 3, 3)
    objectness_label = end_points['objectness_label']
    fp2_inds = end_points['fp2_inds'].long()
    objectness_mask = torch.gather(objectness_label, 1, fp2_inds).bool() # (B, Ns)

    # process labels
    batch_grasp_label = end_points['batch_grasp_label'] # (B, Ns, A, D)
    batch_grasp_offset = end_points['batch_grasp_offset'] # (B, Ns, A, D, 3)
    batch_grasp_tolerance = end_points['batch_grasp_tolerance'] # (B, Ns, A, D)
    B, Ns, A, D = batch_grasp_label.size()

    # pick the one with the highest angle score
    top_view_grasp_angles = batch_grasp_offset[:, :, :, :, 0] #(B, Ns, A, D)
    top_view_grasp_depths = batch_grasp_offset[:, :, :, :, 1] #(B, Ns, A, D)
    top_view_grasp_widths = batch_grasp_offset[:, :, :, :, 2] #(B, Ns, A, D)
    target_labels_inds = torch.argmax(batch_grasp_label, dim=2, keepdim=True) # (B, Ns, 1, D)
    target_labels = torch.gather(batch_grasp_label, 2, target_labels_inds).squeeze(2) # (B, Ns, D)
    target_angles = torch.gather(top_view_grasp_angles, 2, target_labels_inds).squeeze(2) # (B, Ns, D)
    target_depths = torch.gather(top_view_grasp_depths, 2, target_labels_inds).squeeze(2) # (B, Ns, D)
    target_widths = torch.gather(top_view_grasp_widths, 2, target_labels_inds).squeeze(2) # (B, Ns, D)
    target_tolerance = torch.gather(batch_grasp_tolerance, 2, target_labels_inds).squeeze(2) # (B, Ns, D)

    graspable_mask = (target_labels > THRESH_BAD)
    objectness_mask = objectness_mask.unsqueeze(-1).expand_as(graspable_mask)
    loss_mask = (objectness_mask & graspable_mask).float()

    # 1. grasp score loss
    target_labels_inds_ = target_labels_inds.transpose(1, 2) # (B, 1, Ns, D)
    grasp_score = torch.gather(end_points['grasp_score_pred'], 1, target_labels_inds_).squeeze(1)
    grasp_score_loss = huber_loss(grasp_score-target_labels, delta=1.0)
    grasp_score_loss = torch.sum(grasp_score_loss * loss_mask) / (loss_mask.sum() + 1e-6)
    end_points['loss/stage2_grasp_score_loss'] = grasp_score_loss

    # 2. inplane rotation cls loss
    target_angles_cls = target_labels_inds.squeeze(2) # (B, Ns, D)
    criterion_grasp_angle_class = nn.CrossEntropyLoss(reduction='none')
    grasp_angle_class_score = end_points['grasp_angle_cls_pred']
    grasp_angle_class_loss = criterion_grasp_angle_class(grasp_angle_class_score, target_angles_cls)
    grasp_angle_class_loss = torch.sum(grasp_angle_class_loss * loss_mask) / (loss_mask.sum() + 1e-6)
    end_points['loss/stage2_grasp_angle_class_loss'] = grasp_angle_class_loss
    grasp_angle_class_pred = torch.argmax(grasp_angle_class_score, 1)
    end_points['stage2_grasp_angle_class_acc/0_degree'] = (grasp_angle_class_pred==target_angles_cls)[loss_mask.bool()].float().mean()
    acc_mask_15 = ((torch.abs(grasp_angle_class_pred-target_angles_cls)<=1) | (torch.abs(grasp_angle_class_pred-target_angles_cls)>=A-1))
    end_points['stage2_grasp_angle_class_acc/15_degree'] = acc_mask_15[loss_mask.bool()].float().mean()
    acc_mask_30 = ((torch.abs(grasp_angle_class_pred-target_angles_cls)<=2) | (torch.abs(grasp_angle_class_pred-target_angles_cls)>=A-2))
    end_points['stage2_grasp_angle_class_acc/30_degree'] = acc_mask_30[loss_mask.bool()].float().mean()

    # 3. width reg loss
    grasp_width_pred = torch.gather(end_points['grasp_width_pred'], 1, target_labels_inds_).squeeze(1)
    grasp_width_loss = huber_loss((grasp_width_pred-target_widths)/GRASP_MAX_WIDTH, delta=1)
    grasp_width_loss = torch.sum(grasp_width_loss * loss_mask) / (loss_mask.sum() + 1e-6)
    end_points['loss/stage2_grasp_width_loss'] = grasp_width_loss

    # 4. tolerance reg loss
    grasp_tolerance_pred = torch.gather(end_points['grasp_tolerance_pred'], 1, target_labels_inds_).squeeze(1)
    grasp_tolerance_loss = huber_loss((grasp_tolerance_pred-target_tolerance)/GRASP_MAX_TOLERANCE, delta=1)
    grasp_tolerance_loss = torch.sum(grasp_tolerance_loss * loss_mask) / (loss_mask.sum() + 1e-6)
    end_points['loss/stage2_grasp_tolerance_loss'] = grasp_tolerance_loss

    grasp_loss = grasp_score_loss + grasp_angle_class_loss\
                + grasp_width_loss + grasp_tolerance_loss
    return grasp_loss, end_points

def compute_manipulability_loss(end_points):
    #see graspnet.py pre_decode()
    print("compute_manipulability_loss")

    batch_size = len(end_points['point_clouds'])

    grasp_preds = []
    # Generate flange pose (position+orientation)
    for i in range(batch_size):
        print("==>batch ID:", i)

        ## load predictions
        objectness_score = end_points['objectness_score'][i].float()
        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_center = end_points['fp2_xyz'][i].float()
        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_angle_class_score = end_points['grasp_angle_cls_pred'][i]

        # print("1_grasp_center: ", grasp_center)
        # print("1_approaching: ", approaching)

        ## slice preds by angle
        # grasp angle
        grasp_angle_class = torch.argmax(grasp_angle_class_score, 0)
        grasp_angle = grasp_angle_class.float() / 12 * np.pi
        # grasp score & width & tolerance
        grasp_angle_class_ = grasp_angle_class.unsqueeze(0)
        grasp_score = torch.gather(grasp_score, 0, grasp_angle_class_).squeeze(0)

        ## slice preds by score/depth
        # grasp depth
        grasp_depth_class = torch.argmax(grasp_score, 1, keepdims=True)
        grasp_depth = (grasp_depth_class.float()+1) * 0.01
        # grasp score & angle & width & tolerance
        grasp_angle = torch.gather(grasp_angle, 1, grasp_depth_class)

        ## slice preds by objectness
        objectness_pred = torch.argmax(objectness_score, 0)
        objectness_mask = (objectness_pred==1)
        approaching = approaching[objectness_mask]
        grasp_angle = grasp_angle[objectness_mask]
        grasp_center = grasp_center[objectness_mask]

        print("2_grasp_center: ", grasp_center.detach().cpu().numpy()[0])
        # print("2_approaching: ", approaching)
        # print("2_grasp_angle: ", grasp_angle)

        ## convert to rotation matrix
        Ns = grasp_angle.size(0)
        approaching_ = approaching.view(Ns, 3)
        grasp_angle_ = grasp_angle.view(Ns)
        rotation_matrix = batch_viewpoint_params_to_matrix(approaching_, grasp_angle_)
        rotation_matrix = rotation_matrix.view(Ns, 9)
        print("3_approaching: ", approaching.detach().cpu().numpy()[0])
        print("3_grasp_angle: ", grasp_angle.detach().cpu().numpy()[0])
        print("rotation_matrix: ", rotation_matrix.detach().cpu().numpy()[0])

    # Compute IK
    joint_angles = 0

    # Compute Joint-Limits Score (JL-score)
    JL_score = 0

    # Compute Singularity Score (S-score)
    S_score = 0

    # Compute Manipulability Score (M-score)
    M_score = 0
    manipulability_loss = M_score
    return manipulability_loss, end_points