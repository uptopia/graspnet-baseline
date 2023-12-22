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
import open3d as o3d
from graspnetAPI import GraspGroup
import message_filters

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
        self.collision_thresh = 0.05
        self.voxel_size = 0.005
        #=====Parameters Setting=====#
        self.net = self.get_net()

        # self.o3d_vis = o3d.visualization.Visualizer()
        # self.o3d_vis.create_window()
        # self.vis_cloud = o3d.geometry.PointCloud()
        # self.o3d_vis.add_geometry(self.vis_cloud)#[cloud, *grippers])
        # # self.vis_grasp = o3d.geometry.li
        # # o3d.visualization.draw_geometries([cloud, *grippers])

        print("nnnnnnnnnnnnnnnnnnnet")
        rospy.init_node('dlo_grasp_pose', anonymous=False)
        # #----1topic---#
        # rospy.Subscriber('/dlo_cloud', PointCloud2, self.dlo_grasp_cb)
 
        #----2topics---#
        dlo_sub = message_filters.Subscriber('/dlo_cloud', PointCloud2)
        scene_sub = message_filters.Subscriber('/ori_cloud_pub', PointCloud2)

        ts = message_filters.ApproximateTimeSynchronizer([dlo_sub, scene_sub], queue_size=10, slop=10, allow_headerless=False)
        ts.registerCallback(self.callback)
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
        print("process_cloud")
        # print("print(type(gen)):", type(gen))
        # # print(gen)
        cloud_masked = np.empty([1,3])
        color_masked = np.empty([1,3])
        for idx, g in enumerate(gen):

            #==point (x, y, z)==#
            pt = np.array([[g[0], g[1], g[2]]])
            cloud_masked = np.append(cloud_masked,[[g[0], g[1], g[2]]], axis = 0)

            #==rgb (r, g, b)==#
            # cast float32 to int so that bitwise operations are possible
            s = struct.pack('>f' ,g[3])
            i = struct.unpack('>l',s)[0]
            # you can get back the float value by the inverse operations
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000)>> 16
            g = (pack & 0x0000FF00)>> 8
            b = (pack & 0x000000FF)
            color_masked = np.append(color_masked,[[r,g,b]], axis = 0)

        # print("cloud_masked: ", type(cloud_masked), cloud_masked.shape)
        # print("color_masked: ", type(color_masked), color_masked.shape)
        # print(len(cloud_masked), self.num_point)
        # print(type(len(color_masked)), type(self.num_point))
        
        # sample points
        if len(cloud_masked) >= self.num_point:
            print('a')
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            print('b')
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # print("cloud_sampled: ", type(cloud_sampled), cloud_sampled.shape)
        # print("color_sampled: ", type(color_sampled), color_sampled.shape)

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

    def get_grasps(self, end_points):
        print("end_points: ", end_points)

        # Forward pass
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.15, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        return gg

    def vis_grasps(self, gg, cloud):
        gg.nms()
        gg.sort_by_score()
        gg = gg[:25]
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])

        # self.o3d_vis.update_geometry()
        # self.vis_cloud = cloud
        # self.o3d_vis.poll_events()
        # self.o3d_vis.update_renderer()

    def dlo_grasp_cb(self, cloud_msg):
        print("dlo_grasp_cb:")
        assert isinstance(cloud_msg, PointCloud2)

        gen = point_cloud2.read_points_list(cloud_msg, skip_nans=False)
        if gen:
            end_points, cloud_o3d = self.process_cloud(gen)
    
        gg = self.get_grasps(end_points)
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud_o3d.points))
        self.vis_grasps(gg, cloud_o3d)

    
    def callback(self, dlo_msg, scene_msg):
        print("dlo_grasp_cb:")
        # print(dlo_msg.header.stamp)
        # print(scene_msg.header.stamp)
        assert isinstance(dlo_msg, PointCloud2)
        start = time.time()
   
        dlo_gen = point_cloud2.read_points_list(dlo_msg, skip_nans=False)
        if dlo_gen:
            end_points, dlo_cloud_o3d = self.process_cloud(dlo_gen)

        scn_gen = point_cloud2.read_points_list(scene_msg, skip_nans=False)
        if scn_gen:
            _, scn_cloud_o3d = self.process_cloud(scn_gen)
    
        cloud_all = o3d.geometry.PointCloud()
        points_tmp = np.concatenate((dlo_cloud_o3d.points, scn_cloud_o3d.points))
        colors_tmp = np.concatenate((dlo_cloud_o3d.colors, scn_cloud_o3d.colors))
        cloud_all.points = o3d.utility.Vector3dVector(points_tmp)
        cloud_all.colors = o3d.utility.Vector3dVector(colors_tmp)

        gg = self.get_grasps(end_points)
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud_all.points))
        
        end = time.time()
        print("DLO grasp pose elapsed time: (sec)", end - start)
        self.vis_grasps(gg, cloud_all)

if __name__ == '__main__':
    # DLO_Grasp()
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