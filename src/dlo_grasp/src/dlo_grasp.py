#! /usr/bin/env python3

from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import rospy
# import ros_numpy.pointcloud2 as 
import os
import sys
import time
import struct
import ctypes
import torch
import numpy as np

ROOT_DIR = "/home/iclab/work/graspnet-baseline/" #os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
print(ROOT_DIR, os.path.join(ROOT_DIR, 'models'))
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector

# #=====Parameters Setting=====#
# parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
# parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
# parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
# parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
# parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
# cfgs = parser.parse_args()
# #=====Parameters Setting=====#

class DLO_Grasp():
    def __init__(self):
        print("DLOGRASP!PPPPPPPPPPPPPPPPPP")
        
        #=====Parameters Setting=====#
        self.checkpoint_path = "/home/iclab/work/graspnet-baseline/logs/log_rs/checkpoint-rs.tar"
        self.num_point = 20000
        self.num_view = 300
        self.collision_thresh = 0.1
        self.voxel_size = 0.01
        #=====Parameters Setting=====#
        net = self.get_net()

        print("nnnnnnnnnnnnnnnnnnnet")
        rospy.init_node('dlo_grasp_node', anonymous=False)
        rospy.Subscriber('/dlo_cloud', PointCloud2, self.dlo_grasp_cb)
        rospy.spin()

    def get_net(self):
        # Init the model
        net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        
        # Check device (GPU/CPU)
        #https://www.cnblogs.com/xiaodai0/p/10413711.html
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

        print("torch.cuda.is_available() = {}, use device: {}".format(torch.cuda.is_available(), device))
        
        # Load model checkpoint
        if torch.cuda.is_available() == False:
            print('use cpu')
            checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        else:
            print('use gpu')
            checkpoint = torch.load(self.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(self.checkpoint_path, start_epoch))
        
        # Set model to Eval mode
        net.eval()
        
        return net
    
    def process_cloud(self, gen):
        # https://answers.ros.org/question/344096/subscribe-pointcloud-and-convert-it-to-numpy-in-python/

        int_data = list(gen)

        for x in int_data:
            test = x[3] 
            # cast float32 to int so that bitwise operations are possible
            s = struct.pack('>f' ,test)
            i = struct.unpack('>l',s)[0]
            # you can get back the float value by the inverse operations
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000)>> 16
            g = (pack & 0x0000FF00)>> 8
            b = (pack & 0x000000FF)
            # prints r,g,b values in the 0-255 range
                        # x,y,z can be retrieved from the x[0],x[1],x[2]
            xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
            rgb = np.append(rgb,[[r,g,b]], axis = 0)
            print(type(xyz), xyz)

    #     # generate cloud
    #     camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    #     cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    #     # get valid points
    #     mask = (workspace_mask & (depth > 0))
    #     cloud_masked = cloud[mask]
    #     color_masked = color[mask]

    #     print("cloud:", type(cloud), cloud.shape) #<class 'numpy.ndarray'>
    #     print("mask:", type(mask), mask.shape)
    #     print("cloud_masked:", type(cloud_masked), cloud_masked.shape)
    #     print("color_masked:", type(color_masked), color_masked.shape)

    #     # sample points
    #     if len(cloud_masked) >= self.num_point:
    #         idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
    #     else:
    #         idxs1 = np.arange(len(cloud_masked))
    #         idxs2 = np.random.choice(len(cloud_masked), self.num_point-len(cloud_masked), replace=True)
    #         idxs = np.concatenate([idxs1, idxs2], axis=0)
    #     cloud_sampled = cloud_masked[idxs]
    #     color_sampled = color_masked[idxs]

    #     # convert data
    #     cloud = o3d.geometry.PointCloud()
    #     cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    #     cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    #     end_points = dict()
    #     cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     cloud_sampled = cloud_sampled.to(device)
    #     end_points['point_clouds'] = cloud_sampled
    #     end_points['cloud_colors'] = color_sampled

    #     return end_points, cloud

    # def get_grasps(self, net, end_points):
    #     # Forward pass
    #     with torch.no_grad():
    #         end_points = net(end_points)
    #         grasp_preds = pred_decode(end_points)
    #     gg_array = grasp_preds[0].detach().cpu().numpy()
    #     gg = GraspGroup(gg_array)
    #     return gg

    # def collision_detection(self, gg, cloud):
    #     mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
    #     collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
    #     gg = gg[~collision_mask]
    #     return gg

    # def vis_grasps(self, gg, cloud):
    #     gg.nms()
    #     gg.sort_by_score()
    #     gg = gg[:50]
    #     grippers = gg.to_open3d_geometry_list()
    #     o3d.visualization.draw_geometries([cloud, *grippers])

    def dlo_grasp_cb(self, cloud_msg):
        assert isinstance(cloud_msg, PointCloud2)
        print("cloud_msg:")
        gen = point_cloud2.read_points(cloud_msg, skip_nans=True)
        end_points, cloud_o3d = self.process_cloud(gen)
        time.sleep(1)

        # print(type(gen))
        # cnt=0
        # for p in gen:
        #     print("x:%.3f y:%.3f z:%.3f"%(p[0], p[1], p[2]))
        #     cnt+=1
        #     if cnt>10:
        #         break

        # #=====================
    
        # gg = self.get_grasps(net, end_points)
        # if self.collision_thresh > 0:
        #     gg = self.collision_detection(gg, np.array(cloud_o3d.points))
        # self.vis_grasps(gg, cloud_o3d)

if __name__ == '__main__':

    try:
        DLO_Grasp()
    except rospy.ROSInterruptException:
        pass


    # print("cloud_msg:")
    # gen = point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
    # time.sleep(1)

    # print(type(gen))
    # cnt=0
    # for p in gen:
    #     print("x:%.3f y:%.3f z:%.3f"%(p[0], p[1], p[2]))
    #     cnt+=1
    #     if cnt>10:
    #         break