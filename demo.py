""" Demo to show prediction results.
    Author: chenxi-wang
"""
# https://blog.csdn.net/zhuanzhuxuexi/article/details/132059773
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print("ROOT_DIR: " + ROOT_DIR)
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

# import pyrealsense2 as rs
import cv2
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print("ROOT_DIR: " + ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    
    # Check device (GPU/CPU)
    #https://www.cnblogs.com/xiaodai0/p/10413711.html
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    print("torch.cuda.is_available() = {}, use device: {}".format(torch.cuda.is_available(), device))
    
    # Load model checkpoint
    if torch.cuda.is_available() == False:
        print('use cpu')
        checkpoint = torch.load(cfgs.checkpoint_path, map_location=torch.device('cpu'))
    else:
        print('use gpu')
        checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    
    # Set model to Eval mode
    net.eval()
    
    return net

def get_and_process_data(data_dir):
    # load data
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    # print('meta:\n', meta)
    # print('intrinsic:\n', intrinsic)
    # print('factor_depth:\n', factor_depth)
    # print('workspace_mask:\n', workspace_mask) # True, False

    ##====example_data====##
    # intrinsic:
    # [[631.54864502   0.         638.43517329]
    # [  0.         631.20751953 366.49904066]
    # [  0.           0.           1.        ]]
    # factor_depth: [[1000.]]

    ##====motor_data====##
    #intrinsic: 
    # [[919.16595459   0.         641.86254883]
    #  [  0.         919.43945312 366.82495117]
    #  [  0.           0.           1.        ]]
    # factor_depth: [[1000]]


    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    print("cloud:", type(cloud), cloud.shape) #<class 'numpy.ndarray'>
    print("mask:", type(mask), mask.shape)
    print("cloud_masked:", type(cloud_masked), cloud_masked.shape)
    print("color_masked:", type(color_masked), color_masked.shape)

    # cloud: <class 'numpy.ndarray'> (720, 1280, 3)
    # mask: <class 'numpy.ndarray'> (720, 1280)
    # cloud_masked: <class 'numpy.ndarray'> (513688, 3)
    # color_masked: <class 'numpy.ndarray'> (513688, 3)
    print(cloud_masked)

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    # - reverse: bool of order, if False, from high to low, if True, from low to high.
    gg.sort_by_score(reverse=False)
    gg = gg[:50]  # return top 50 grasps  
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))

    # show top 50 grasps
    vis_grasps(gg, cloud)        

def cam_input():
    # create a pipline
    pipeline = rs.pipeline()

    # create a config obj to configure the pipeline
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # start the pipeline
    pipeline.start(config)
    align = rs.align(rs.stream.color) #align obj for depth-color

    try:
        while True:
            #wait for coherent pair of frams: color & depth
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            if not aligned_frames:
                continue

            color_frame = aligned_frames.get_color_frame()
            aligned_depth_frame =  aligned_frames.get_depth_frame()

            if not color_frame or not aligned_depth_frame:
                continue

            #convert frame to np arrays img
            color_frame_img = np.asanyarray(color_frame.get_data())
            aligned_depth_img = np.asanyarray(aligned_depth_frame.get_data())
            
            # display 
            aligned_depth_colormap = cv2.applyColorMap(cv2.converScaleAbs(aligned_depth_img, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow("Align Depth colormap", aligned_depth_colormap)
            cv2.imshow("aligned depth image", aligned_depth_img)
            cv2.imwrite('/home/iclab/work/graspnet-baseline/doc/stream_rs/depth.png', aligned_depth_img)

            cv2.imshow("aligned color image", color_frame_img)
            cv2.imwrite('/home/iclab/work/graspnet-baseline/doc/stream_rs/color.png', color_frame_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__=='__main__':
    
    # #======Method1=======#
    # # Use Streamed Images
    # #====================#
    # cam_input()
    # data_dir = 'doc/stream_rs'
    # demo(data_dir)

    #======Method2=======#
    # Use Saved Images
    #====================#
    data_dir = 'doc/example_data' #'doc/motor_data/test6' #1~6
    demo(data_dir)
