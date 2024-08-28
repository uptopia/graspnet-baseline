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
from geometry_msgs.msg import PoseStamped, TransformStamped
# import tf.transformations #as tr
import tf
import tf2_ros
from std_msgs.msg import ColorRGBA, Float32
from jsk_rviz_plugins.msg import OverlayText

from dlo_srv.srv import DloGraspSrv, DloGraspSrvRequest
from dlo_srv.msg import DloInsClouds

curr_dir =  os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = curr_dir[0:curr_dir.find("src")+3]  #"/home/iclab/work/" 

sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

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
        print("DLO_GraspPoseInCam")

        #=====Parameters Setting=====#
        self.checkpoint_path = os.path.join(ROOT_DIR, 'logs/log_rs/checkpoint-rs.tar') #"/home/iclab/work/graspnet-baseline/logs/log_rs/checkpoint-rs.tar"
        self.num_point = 20000
        self.num_view = 300
        self.collision_thresh = 0.01 #0.05
        self.voxel_size = 0.01  #0.005
        #=====Parameters Setting=====#
        self.net = self.get_net()

        # self.o3d_vis = o3d.visualization.Visualizer()
        # self.o3d_vis.create_window()
        # self.vis_cloud = o3d.geometry.PointCloud()
        # self.o3d_vis.add_geometry(self.vis_cloud)#[cloud, *grippers])
        # # self.vis_grasp = o3d.geometry.li
        # # o3d.visualization.draw_geometries([cloud, *grippers])

        rospy.init_node('DLO_GraspPoseInCam', anonymous=False)


        # self.tf_listener = tf.TransformListener()
        # self.tf_buffer = tf2_ros.Buffer()

        # #----1topic---#
        # rospy.Subscriber('/dlo_cloudInCam', PointCloud2, self.dlo_grasp_cb)
 
        # #----1dlo_2topics---#
        # dlo_cloudInCam_sub = message_filters.Subscriber('/dlo_cloudInCam', PointCloud2)
        # ori_cloudInCam_sub = message_filters.Subscriber('/ori_cloudInCam', PointCloud2)
        # ts = message_filters.ApproximateTimeSynchronizer([dlo_cloudInCam_sub, ori_cloudInCam_sub], queue_size=10, slop=10, allow_headerless=False)
        # ts.registerCallback(self.callback)

        #----all_dlo_2topics---#
        all_cloudInCam_sub = message_filters.Subscriber('/all_cloudInCam', DloInsClouds)
        ori_cloudInCam_sub = message_filters.Subscriber('/ori_cloudInCam', PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer([all_cloudInCam_sub, ori_cloudInCam_sub], queue_size=10, slop=10, allow_headerless=False)
        ts.registerCallback(self.callback_all)

        self.dlo_graspInCam = PoseStamped()
        self.dlo_graspInCam_pub = rospy.Publisher('/dlo_graspInCam', PoseStamped, queue_size=10)
        self.dlo_score_pub = rospy.Publisher('/dlo_grasp_score', OverlayText, queue_size=1)

        #=============================
        # tf StaticTransformBroadcaster  
        #=============================
        # self.dlo_graspInCam_br = tf.TransformBroadcaster()
        self.dlo_graspInCam_br = tf2_ros.StaticTransformBroadcaster()

        #=============================
        # ROS Server /dlo_grasp_srv  
        #=============================
        self.s = rospy.Service('dlo_grasp_srv', DloGraspSrv, self.dlo_graspInCam_server)

        rospy.spin()

    def dlo_graspInCam_server(self, req):
        print("req.arrived_to_take_pic:", req.arrived_to_take_pic)
        if req.arrived_to_take_pic == True:
            print("server===================send dlo_graspInCam\n", self.dlo_graspInCam)
            # res = DloGraspSrv()
            # res.grasp_in_cam = self.dlo_graspInCam
            return self.dlo_graspInCam
        else:
            print("not arrived to take pic")
            return


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
        # print("process_cloud")
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
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
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

        # print("end_points: ", end_points)

        # Forward pass
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def collision_detection(self, gg_ori, cloud):

        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg_ori, approach_dist=0.15, collision_thresh=self.collision_thresh)
        gg = gg_ori[~collision_mask]
        rr = gg_ori[collision_mask]

        # print("collision_mask", collision_mask)
        # print("np.size(gg_ori)", np.size(gg_ori))
        # print("np.size(gg)", np.size(gg))
        # print("np.size(rr)", np.size(rr))
        return gg, rr

    def vis_grasps(self, gg, cloud, show_top_num, windowname):

        # gg.nms()
        # gg.sort_by_score()
        if np.size(gg)<show_top_num:
            num = np.size(gg)
        else:
            num=show_top_num
        gg = gg[:num]
        
        grippers = gg.to_open3d_geometry_list()
        # o3d.visualization.draw_geometries([cloud, *grippers], windowname, \
        #     lookat=[ -0.022709501252327787, 0.17799084108354188, 0.40640778473950101 ], \
        #     up=[ 0.017900563028521502, -0.88416689576134511, -0.46682809500179911 ], \
        #     front=[ 0.10182361226540698, 0.46608747114240157, -0.87885972784586341 ], \
        #     zoom=0.33999999999999964)

        o3d.visualization.draw_geometries([cloud, *grippers], windowname, \
            lookat=[ -0.006281841072686649, 0.22370532914918007, 0.43965770683307587 ], \
            up=[ 0.011915128187709208, -0.71084909258354689, -0.7032436258462772 ], \
            front=[ -0.027342757202087748, 0.70279894195116566, -0.71086287061634468 ], \
            zoom=0.1799999999999996)
        
        # {
		# 	"boundingbox_max" : [ 0.45387211441993713, 0.2853831946849823, 0.55108187534386033 ],
		# 	"boundingbox_min" : [ -0.51234573125839233, -0.27358525991439819, 0.0 ],
		# 	"field_of_view" : 60.0,
		# 	"front" : [ -0.027342757202087748, 0.70279894195116566, -0.71086287061634468 ],
		# 	"lookat" : [ -0.006281841072686649, 0.22370532914918007, 0.43965770683307587 ],
		# 	"up" : [ 0.011915128187709208, -0.71084909258354689, -0.7032436258462772 ],
		# 	"zoom" : 0.1799999999999996
		# }

        # self.o3d_vis.update_geometry()
        # self.vis_cloud = cloud
        # self.o3d_vis.poll_events()
        # self.o3d_vis.update_renderer()

    def callback_all(self, all_dlo_msg, scene_msg):

        print("all_dlo_grasp_cb:")

        assert isinstance(all_dlo_msg, DloInsClouds)
        start = time.time()
   
        tot_dlo_clouds = len(all_dlo_msg.dlo_clouds)
        print("tot_dlo_clouds: ", tot_dlo_clouds)

        # scene point cloud
        scn_gen = point_cloud2.read_points_list(scene_msg, skip_nans=False)
        if scn_gen:
            _, scn_cloud_o3d = self.process_cloud(scn_gen)

        for n in range(tot_dlo_clouds):

            # dlo point cloud
            dlo_gen = point_cloud2.read_points_list(all_dlo_msg.dlo_clouds[n], skip_nans=False)
            if dlo_gen:
                end_points, dlo_cloud_o3d = self.process_cloud(dlo_gen)
        
            cloud_all = o3d.geometry.PointCloud()
            # points_tmp = dlo_cloud_o3d.points
            # colors_tmp = dlo_cloud_o3d.colors
            points_tmp = np.concatenate((dlo_cloud_o3d.points, scn_cloud_o3d.points))
            colors_tmp = np.concatenate((dlo_cloud_o3d.colors, scn_cloud_o3d.colors))
            cloud_all.points = o3d.utility.Vector3dVector(points_tmp)
            cloud_all.colors = o3d.utility.Vector3dVector(colors_tmp)

            gg_ori = self.get_grasps(end_points)
            gg_ori.nms()
            gg_ori.sort_by_score()
            top_more = 100
            top_less = 5
            # self.vis_grasps(gg_ori, cloud_all, top_more, "Before Collision #"+str(n)+" Top "+str(top_more))
            # self.vis_grasps(gg_ori, cloud_all, top_less, "Before Collision #"+str(n)+" Top "+str(top_less))

            if self.collision_thresh > 0:
                gg, rr = self.collision_detection(gg_ori, np.array(cloud_all.points))
            gg.nms()
            gg.sort_by_score()
            # self.vis_grasps(gg, cloud_all, top_more, "Remove Collision"+str(n)+" Top "+str(top_more))
            # self.vis_grasps(gg, cloud_all, top_less, "Remove Collision"+str(n)+" Top "+str(top_less))
            # # self.vis_grasps(rr, cloud_all, 100, "collision")

            if (np.size(gg))<5:
                id_max = np.size(gg)
            else:
                id_max=5

            for id in range(id_max):
                print('score: ', gg[id].score)
                print('width: ', gg[id].width)
                print('height: ', gg[id].height)
                print('depth: ', gg[id].depth)
                print('translation: ', gg[id].translation)
                print('rotation_matrix: ', gg[id].rotation_matrix)
                print('object_id: ', gg[id].object_id)

                # self.vis_grasps(gg, cloud_all, 1, "Remove Collision"+str(n)+" Top "+str(1))

                text = OverlayText()
                text.width = 400
                text.height = 80
                text.left = 10
                text.top = 10

                text.text_size = 12
                text.line_width = 2
                text.font = "DejaVu Sans Mono"

                text.text = """Grasp score: %f.
                M-score: %f.
                Manipulability score: %f.
                Manipulability score: %f.
                """ % (gg[id].score, gg[id].score, gg[id].score, gg[id].score)

                text.fg_color = ColorRGBA(25 / 255.0, 1.0, 240.0 / 255.0, 1.0)
                text.bg_color = ColorRGBA(0.0, 0.0, 0.0, 0.2)

                self.dlo_score_pub.publish(text)

                #=============================
                # Publish dlo_graspInCam
                # PoseStamped() cam_H_obj
                #=============================
                self.dlo_graspInCam.header.stamp = all_dlo_msg.header.stamp #rospy.Time.now()
                self.dlo_graspInCam.header.frame_id = 'camera_color_optical_frame'

                self.dlo_graspInCam.pose.position.x = gg[id].translation[0]
                self.dlo_graspInCam.pose.position.y = gg[id].translation[1]
                self.dlo_graspInCam.pose.position.z = gg[id].translation[2]

                matrix4x4=np.identity(4)
                matrix4x4[:3,:3]=gg[id].rotation_matrix
                q = tf.transformations.quaternion_from_matrix(matrix4x4)
                self.dlo_graspInCam.pose.orientation.x = q[0]
                self.dlo_graspInCam.pose.orientation.y = q[1]
                self.dlo_graspInCam.pose.orientation.z = q[2]
                self.dlo_graspInCam.pose.orientation.w = q[3]
                self.dlo_graspInCam_pub.publish(self.dlo_graspInCam)
                print('dlo_graspInCam:', self.dlo_graspInCam)


                # ###############
                # # grasp_inCam -->
                # # grasp_inBase
                # ###############

                # ###IMPORTANT####
                # # change graspInCam timestamp to rospy.Time()
                # # before transformPose!!!
                # ###IMPORTANT####
                # self.dlo_graspInCam.header.stamp =rospy.Time()
                # self.tf_listener.waitForTransform("/base_link", "/camera_color_optical_frame", rospy.Time(), rospy.Duration(4.0))
                # graspInBase = PoseStamped()
                # graspInBase = self.tf_listener.transformPose("/base_link", self.dlo_graspInCam)
                # print("graspInBase:\n", graspInBase)

                #=============================
                # /dlo_graspInCam_frame
                # StaticTransformBroadcaster
                #=============================
                broadcaster = tf2_ros.StaticTransformBroadcaster()
                static_transformStamped = TransformStamped()

                static_transformStamped.header.stamp = rospy.Time.now()
                static_transformStamped.header.frame_id = "camera_color_optical_frame"
                static_transformStamped.child_frame_id = "dlo_graspInCam_frame"

                static_transformStamped.transform.translation = self.dlo_graspInCam.pose.position
                static_transformStamped.transform.rotation = self.dlo_graspInCam.pose.orientation

                broadcaster.sendTransform(static_transformStamped)

                #=============================
                # ROS Server /dlo_grasp_srv  
                #=============================
                # s = rospy.Service('dlo_grasp_srv', DloGraspSrv, self.dlo_graspInCam_server)
                # print("server s: ", self.s)

            end = time.time()
            print("DLO grasp pose elapsed time: (sec)", end - start)
            # self.vis_grasps(gg, cloud_all, 100, "vis_grasps")

    # def dlo_grasp_cb(self, cloud_msg):
    #     print("dlo_grasp_cb:")
    #     assert isinstance(cloud_msg, PointCloud2)

    #     gen = point_cloud2.read_points_list(cloud_msg, skip_nans=False)
    #     if gen:
    #         end_points, cloud_o3d = self.process_cloud(gen)
    
    #     gg = self.get_grasps(end_points)
    #     if self.collision_thresh > 0:
    #         gg = self.collision_detection(gg, np.array(cloud_o3d.points))
    #     self.vis_grasps(gg, cloud_o3d, 100, "vis_grasps")

    def callback(self, dlo_msg, scene_msg):
        print("dlo_grasp_cb:")
        # print(dlo_msg.header.stamp)
        # print(scene_msg.header.stamp)
        assert isinstance(dlo_msg, PointCloud2)
        start = time.time()

        # dlo point cloud
        dlo_gen = point_cloud2.read_points_list(dlo_msg, skip_nans=False)
        if dlo_gen:
            end_points, dlo_cloud_o3d = self.process_cloud(dlo_gen)

        # scene point cloud
        scn_gen = point_cloud2.read_points_list(scene_msg, skip_nans=False)
        if scn_gen:
            _, scn_cloud_o3d = self.process_cloud(scn_gen)
    
        cloud_all = o3d.geometry.PointCloud()
        # points_tmp = dlo_cloud_o3d.points
        # colors_tmp = dlo_cloud_o3d.colors
        points_tmp = np.concatenate((dlo_cloud_o3d.points, scn_cloud_o3d.points))
        colors_tmp = np.concatenate((dlo_cloud_o3d.colors, scn_cloud_o3d.colors))
        cloud_all.points = o3d.utility.Vector3dVector(points_tmp)
        cloud_all.colors = o3d.utility.Vector3dVector(colors_tmp)

        gg = self.get_grasps(end_points)
        # if self.collision_thresh > 0:
        #     gg = self.collision_detection(gg, np.array(cloud_all.points))
        
        # # for t in range(np.size(gg)):
        # #     print(gg[t].score)

        gg.nms()
        gg.sort_by_score()
        # print("=========")
        # print("np.size(gg):", np.size(gg))
        # for t in range(np.size(gg)):
        #     print(gg[t].score)

        # print('type(gg):', type(gg), np.size(gg)) #<class 'graspnetAPI.grasp.GraspGroup'>
        # print(type(gg[id]))#<class 'graspnetAPI.grasp.Grasp'> :Grasp(score, width, height, depth, translation, rotation_matrix, object_id)
        
        if (np.size(gg))<5:
            id_max = np.size(gg)
        else:
            id_max=5

        for id in range(id_max):
            print('score: ', gg[id].score)
            print('width: ', gg[id].width)
            print('height: ', gg[id].height)
            print('depth: ', gg[id].depth)
            print('translation: ', gg[id].translation)
            print('rotation_matrix: ', gg[id].rotation_matrix)
            print('object_id: ', gg[id].object_id)
            # print('gg[id][0]:\n',gg[id][0])
            # print(gg[id][0].translation)
            # print(gg[id][0].rotation_matrix)

            #Grasp: score:0.0945352166891098, width:0.04796336218714714, height:0.019999999552965164, depth:0.009999999776482582, translation:[0. 0. 0.]
            # rotation:
            # [[ 8.5445970e-01  3.7061375e-02 -5.1819408e-01]
            # [-5.1770735e-01 -2.2455113e-02 -8.5526299e-01]
            # [-4.3333333e-02  9.9906063e-01 -4.3670326e-08]]
            # object id:-1

            # # see /graspnetAPI/docs/source/example_eval.rst
            # gg=GraspGroup(np.array([[score_1, width_1, height_1, depth_1, rotation_matrix_1(9), translation_1(3), object_id_1],
            #                         ...,
            #                         [score_N, width_N, height_N, depth_N, rotation_matrix_N(9), translation_N(3), object_id_N]]))
            # gg.save_npy(save_path)

            text = OverlayText()
            text.width = 400
            text.height = 80
            text.left = 10
            text.top = 10

            text.text_size = 12
            text.line_width = 2
            text.font = "DejaVu Sans Mono"

            text.text = """Grasp score: %f.
            M-score: %f.
            Manipulability score: %f.
            Manipulability score: %f.
            """ % (gg[id].score, gg[id].score, gg[id].score, gg[id].score)

            text.fg_color = ColorRGBA(25 / 255.0, 1.0, 240.0 / 255.0, 1.0)
            text.bg_color = ColorRGBA(0.0, 0.0, 0.0, 0.2)

            self.dlo_score_pub.publish(text)
            
            #=============================
            # Publish dlo_graspInCam
            # PoseStamped() cam_H_obj
            #=============================
            self.dlo_graspInCam.header.stamp = dlo_msg.header.stamp#rospy.Time.now()
            self.dlo_graspInCam.header.frame_id = 'camera_color_optical_frame'

            self.dlo_graspInCam.pose.position.x = gg[id].translation[0]
            self.dlo_graspInCam.pose.position.y = gg[id].translation[1]
            self.dlo_graspInCam.pose.position.z = gg[id].translation[2]

            matrix4x4=np.identity(4)
            matrix4x4[:3,:3]=gg[id].rotation_matrix
            # print('matrix4x4:', matrix4x4)
            q = tf.transformations.quaternion_from_matrix(matrix4x4)
            self.dlo_graspInCam.pose.orientation.x = q[0]
            self.dlo_graspInCam.pose.orientation.y = q[1]
            self.dlo_graspInCam.pose.orientation.z = q[2]
            self.dlo_graspInCam.pose.orientation.w = q[3]
            self.dlo_graspInCam_pub.publish(self.dlo_graspInCam)

            # =============================
            # /dlo_graspInCam_frame
            # TransformBroadcaster
            # =============================
            # # add a dlo_obj frame (cam_H_dlo)
            # self.dlo_graspInCam_br.sendTransform((self.dlo_graspInCam.pose.position.x, self.dlo_graspInCam.pose.position.y, self.dlo_graspInCam.pose.position.z),
            #                                 (self.dlo_graspInCam.pose.orientation.x, self.dlo_graspInCam.pose.orientation.y, self.dlo_graspInCam.pose.orientation.z, self.dlo_graspInCam.pose.orientation.w),
            #                                 self.dlo_graspInCam.header.stamp, #rospy.Time.now(),
            #                                 "dlo_graspInCam_frame", 
            #                                 "camera_color_optical_frame") #camera_link

            #=============================
            # /dlo_graspInCam_frame
            # StaticTransformBroadcaster
            #=============================
            broadcaster = tf2_ros.StaticTransformBroadcaster()
            static_transformStamped = TransformStamped()

            static_transformStamped.header.stamp = rospy.Time.now()
            static_transformStamped.header.frame_id = "camera_color_optical_frame"
            static_transformStamped.child_frame_id = "dlo_graspInCam_frame"

            static_transformStamped.transform.translation = self.dlo_graspInCam.pose.position
            static_transformStamped.transform.rotation = self.dlo_graspInCam.pose.orientation

            broadcaster.sendTransform(static_transformStamped)

            # s = rospy.Service('dlo_grasp_srv', DloGraspSrv, self.dlo_graspInCam_server)
            # print("server s: ", self.s)
            print('dlo_graspInCam:', self.dlo_graspInCam)

        end = time.time()
        print("DLO grasp pose elapsed time: (sec)", end - start)
        #self.vis_grasps(gg, cloud_all, 100, "vis_grasps")

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