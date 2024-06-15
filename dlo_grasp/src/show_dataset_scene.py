#! /usr/bin/env python3

import os
import sys
import struct
import numpy as np
import scipy.io as scio
from PIL import Image

import rospy
import std_msgs
import sensor_msgs
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField

import cv2
from cv_bridge import CvBridge

curr_dir =  os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = curr_dir[0:curr_dir.find("src")+3]  #"/home/iclab/work/" 
data_dir = os.path.join(ROOT_DIR, 'doc/example_data')#'doc/example_data' #'doc/motor_data/test6' #1~6

sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector

from graspnet_dataset import GraspNetDataset
from data_utils import CameraInfo, create_point_cloud_from_depth_image

class ImagePublisherNode:
    def __init__(self):
        rospy.init_node('image_publisher_node', anonymous=True)
        self.color_pub = rospy.Publisher('scene_color', sensor_msgs.msg.Image, queue_size=10)
        self.depth_pub = rospy.Publisher('scene_depth', sensor_msgs.msg.Image, queue_size=10)
        self.cloud_pub = rospy.Publisher('scene_cloud', PointCloud2, queue_size=10)

        self.bridge = CvBridge()
        self.run()

    def publish_color_depth_image(self, image_path, depth_path):
        # Read the image using OpenCV
        # color = cv2.imread(os.path.join(data_dir, 'color.png'), cv2.IMREAD_COLOR)
        # depth = cv2.imread(os.path.join(data_dir, 'depth.png'), cv2.IMREAD_COLOR)
        # image_path = os.path.join(data_dir, 'color.png')


        # PIL_color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0 #
        # PIL_depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
        # print(type(PIL_color), type(PIL_depth)) #<class 'numpy.ndarray'> 
        # print(PIL_color.shape, PIL_depth.shape) #(720, 1280, 3) (720, 1280)

        # im = PIL_color.convert('RGB')

        # im = PIL_depth.convert('RGB')
        # print(type(im))
        # im = PIL_color

        # msg = sensor_msgs.msg.Image()
        # msg.header.stamp = rospy.Time.now()
        # msg.height = im.shape[0] #height
        # msg.width = im.shape[1] #width
        # msg.encoding = "rgb8"
        # msg.is_bigendian = False
        # msg.step = 3 * msg.width#im.width
        # msg.data = PIL_color#np.array(im).tobytes()
        # pub.publish(msg)

        print("image_path:", image_path)
        color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_COLOR)
        # cv2.imshow("color_image", color_image)
        # cv2.waitKey(1)

        # Check if the image was successfully loaded
        if color_image is None:
            rospy.logerr(f"Failed to load image from {image_path}")
            return
        if depth_image is None:
            rospy.logerr(f"Failed to load image from {depth_path}")
            return
        
        # Convert the OpenCV image to a ROS Image message
        color_image_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
        depth_image_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="bgr8")
        # Publish the image message
        self.color_pub.publish(color_image_msg)
        self.depth_pub.publish(depth_image_msg)
        
        rospy.loginfo("Published color image to color_image_topic")


    def create_point_cloud_from_color_depth_image(self, color, depth, camera, organized=True):
        """ Generate point cloud using depth image only.

            Input:
                color: [numpy.ndarray, (H,W), numpy.float32]
                    color image
                depth: [numpy.ndarray, (H,W), numpy.float32]
                    depth image
                camera: [CameraInfo]
                    camera intrinsics
                organized: bool
                    whether to keep the cloud in image shape (H,W,3)

            Output:
                cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                    generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
        """
        assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
        assert(color.shape[0] == camera.height and color.shape[1] == camera.width) 

        xmap = np.arange(camera.width)
        ymap = np.arange(camera.height)
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth / camera.scale
        points_x = (xmap - camera.cx) * points_z / camera.fx
        points_y = (ymap - camera.cy) * points_z / camera.fy
        print(points_x.shape)

        # print("color", color[0][0][0], color[0][0][1], color[0][0][2])

        r = color[:,:,0] #(720, 1280)
        g = color[:,:,1]
        b = color[:,:,2]
        print(r.shape) 

        cloud_xyzrgb = np.stack([points_x, points_y, points_z, r, g, b], axis=-1)
        if not organized:
            cloud = cloud.reshape([-1, 6]) # x, y, z, r, g, b
        return cloud_xyzrgb
    
    # def point_cloud(self, points, parent_frame):

    #     ros_dtype = sensor_msgs.msg.PointField.FLOAT32
    #     dtype = np.float32
    #     itemsize = np.dtype(dtype).itemsize # A 32-bit float takes 4 bytes.

    #     data = points.astype(dtype).tobytes() 

    #     # The fields specify what the bytes represents. The first 4 bytes 
    #     # represents the x-coordinate, the next 4 the y-coordinate, etc.
    #     fields = [sensor_msgs.msg.PointField(
    #         name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
    #         for i, n in enumerate('xyz')]

    #     # The PointCloud2 message also has a header which specifies which 
    #     # coordinate frame it is represented in. 
    #     header = std_msgs.msg.Header(frame_id=parent_frame)

    #     return sensor_msgs.msg.PointCloud2(
    #         header=header,
    #         height=1, 
    #         width=points.shape[0],
    #         is_dense=False,
    #         is_bigendian=False,
    #         fields=fields,
    #         point_step=(itemsize * 3), # Every point consists of three float32s.
    #         row_step=(itemsize * 3 * points.shape[0]), 
    #         data=data
    #     )


    def numpy_to_pointcloud2(self, points, frame_id='map'):
        print(len(points), type(points), points.shape)

        msg = PointCloud2()
        msg.header = std_msgs.msg.Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Convert numpy array to list of tuples
        points_list = points.reshape(-1, 3).tolist()

        # Populate the PointCloud2 message with points
        msg.height = 1
        msg.width = len(points_list)
        msg.is_bigendian = False
        msg.point_step = 12  # 3 fields * 4 bytes (FLOAT32 is 4 bytes)
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True  # Assuming all points are valid

        # Convert points to binary data
        msg.data = []
        for p in points_list:
            msg.data += struct.pack('fff', *p)

        return msg
    
    def publish_point_cloud(self, points, colors, frame_id="map"):
        #https://blog.csdn.net/qq_44992157/article/details/130662947

        # print(len(points), len(colors))
        # print(points.shape, colors.shape)
        # assert(len(points) == len(colors))

        # The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
        FIELDS_XYZ = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        FIELDS_XYZRGB = FIELDS_XYZ + \
            [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]
        
        points_list=[]
        for i in range(colors.shape[0]):
            for j in range(colors.shape[1]):
                
                x, y, z = points[i, j]
                b, g, r = colors[i, j]
                # rgb = (int(r) << 16) | (int(g) << 8) | int(b)
                rgb = struct.unpack('I', struct.pack('BBBB', int(b), int(g), int(r), 255))[0]
               
                # print(x, y, z, r, g, b)
                # print(r,g, b)
                points_list.append([x, y, z, rgb])
                # print([x, y, z, rgb])
                # # r = colors[i,j,0] #(720, 1280)
                # # g = colors[i,j,1]
                # # b = colors[i,j,2]

                # # rgb = (int(r) << 16 | (int(g) << 8) | int(b))


        # # Convert numpy array to list of tuples
        # points_list = points.reshape(-1, 3).tolist()
        # tt = points_list[0]
        # tt.extend(points_list[1])
        # print(len(points_list), tt)

        cloud_msg = PointCloud2()
        cloud_msg.header = std_msgs.msg.Header()
        cloud_msg.header.stamp = rospy.Time.now()
        cloud_msg.header.frame_id = frame_id
        cloud_msg.fields = FIELDS_XYZRGB
        cloud_msg.height = 1                #points.shape[1] 
        cloud_msg.width  = len(points_list) #points.shape[0]
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 16# 12  # 3 fields * 4 bytes (FLOAT32 is 4 bytes)
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.is_dense = False  # Assuming all points are valid

        # cloud_msg.data = []
        # for p in points_list:
        #     cloud_msg.data += struct.pack('ffff', *p)
        # cloud_msg.data = np.asarray(points_list, dtype=np.float32).tobytes()

        buffer = []
        for point in points_list:
            buffer.append(struct.pack('fffI', *point))
        cloud_msg.data = b''.join(buffer)

        self.cloud_pub.publish(cloud_msg)

    #     # cloud_msg.is_bigendian = False
    #     # cloud_msg.point_step = 12  # 3 fields * 4 bytes (FLOAT32 is 4 bytes)
    #     # cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
    #     # cloud_msg.is_dense = True  # Assuming all points are valid
    #     # cloud_msg.is_dense = False  #True: unorganized, False:organized
    #     if colors is not None:
    #         cloud_msg.fields = FIELDS_XYZRGB
    #         print("xyzrgb")
    #     else:
    #         cloud_msg.fields = FIELDS_XYZ
    #         print("xyz")

    #     point_list = []

    #     for i in range(5):#colors.shape[0]):
    #         for j in range(5):#colors.shape[1]):
                
    #             x, y, z = points[i, j]
    #             r, g, b = colors[i, j]
    #             rgb = (int(r) << 16) | (int(g) << 8) | int(b)
    #             # print(x, y, z, r, g, b)

    #             point_list.append([x, y, z, rgb])
    #             print([x, y, z, rgb])
    #             # # r = colors[i,j,0] #(720, 1280)
    #             # # g = colors[i,j,1]
    #             # # b = colors[i,j,2]

    #             # # rgb = (int(r) << 16 | (int(g) << 8) | int(b))
    #             # # pt_xyzrgb = np.append(points[i,j], rgb)
    #             # # point_list.append(pt_xyzrgb)
    #             # # print(type(points[i,j].tolist()))
    #             # # print(type(points[i,j]))
    #             # pt_xyzrgb = points[i,j].tolist()
    #             # # pt_xyzrgb.append(rgb)
    #             # point_list.append(pt_xyzrgb)
    #     # print(point_list)
    #     # cloud_msg.data = np.asarray(point_list, dtype=np.float32).tobytes()
    # #     self.cloud_pub.publish(cloud_msg)

    # #    # Convert numpy array to list of tuples
    # #     points_list = points.reshape(-1, 3).tolist()

    # #     # Populate the PointCloud2 message with points
    # #     msg.height = 1
    # #     msg.width = len(points_list)
    # #     msg.is_bigendian = False
    # #     msg.point_step = 12  # 3 fields * 4 bytes (FLOAT32 is 4 bytes)
    # #     msg.row_step = msg.point_step * msg.width
    # #     msg.is_dense = True  # Assuming all points are valid

    #     # Convert points to binary data
    #     # cloud_msg.data = point_list
    #     # for p in point_list:
    #     #     cloud_msg.data += struct.pack('fff', *p)
    #     # print(cloud_msg)

        # Convert points to binary data



    def run(self):
        rate = rospy.Rate(1)  # 1 Hz
        while not rospy.is_shutdown():
            # Replace 'path_to_your_image.png' with your actual image path

            #================
            # COLOR, DEPTH
            #================
            color_path = os.path.join(data_dir, 'color.png')
            depth_path = os.path.join(data_dir, 'depth.png')
            self.publish_color_depth_image(color_path, depth_path)

            #================
            #   POINTCLOUD
            #================       
            meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']

            # generate cloud
            camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
            print("\033[33mcamera height, width:", camera.height, camera.width) #720, 1280
            PIL_depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
            PIL_color = np.array(Image.open(os.path.join(data_dir, 'color.png')))
            cloud = create_point_cloud_from_depth_image(PIL_depth, camera, organized=True) #<class 'numpy.ndarray'>
            # # cloud = self.create_point_cloud_from_color_depth_image(PIL_color, PIL_depth, camera)#, organized=True)
            # PIL_color =[]
            color_image = cv2.imread( os.path.join(data_dir, 'color.png'), cv2.IMREAD_COLOR)
            self.publish_point_cloud(cloud, color_image, 'map')

            # cc = self.numpy_to_pointcloud2(cloud)

            # # # print(type(cloud), type(cc))
            # # # print("cloud:", type(cloud), cloud.shape) #<class 'numpy.ndarray'>

            # self.cloud_pub.publish(cc)

            # # rate.sleep()

if __name__ == '__main__':
    try:
        node = ImagePublisherNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

# from sensor_msgs.msg import PointCloud2, PointField
# from sensor_msgs import point_cloud2
# import sensor_msgs
# import rospy
# import std_msgs.msg
# # import ros_numpy.pointcloud2 as 
# import os
# import sys
# import time
# import struct
# import ctypes
# import torch
# import numpy as np
# import open3d as o3d
# from graspnetAPI import GraspGroup
# import message_filters
# from geometry_msgs.msg import PoseStamped
# import tf.transformations #as tr
# import tf
# import cv2
# import scipy.io as scio
# from PIL import Image
# import argparse
# from cv_bridge import CvBridge

# curr_dir =  os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = curr_dir[0:curr_dir.find("line")+4]  #"/home/iclab/work/graspnet-baseline/" 

# sys.path.append(os.path.join(ROOT_DIR, 'models'))
# sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))

# from graspnet import GraspNet, pred_decode
# from collision_detector import ModelFreeCollisionDetector

# from graspnet_dataset import GraspNetDataset
# from data_utils import CameraInfo, create_point_cloud_from_depth_image

# # parser = argparse.ArgumentParser()
# # parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
# # parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
# # parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
# # parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
# # parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
# # cfgs = parser.parse_args()


# # #=====Parameters Setting=====#
# # parser = argparse.ArgumentParser()
# # parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
# # parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
# # parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
# # parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
# # parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
# # cfgs = parser.parse_args()
# # #=====Parameters Setting=====#

# # The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
# FIELDS_XYZ = [
#     PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
#     PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
#     PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
# ]
# FIELDS_XYZRGB = FIELDS_XYZ + \
#     [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

# # Bit operations
# BIT_MOVE_16 = 2**16
# BIT_MOVE_8 = 2**8
# convert_rgbUint32_to_tuple = lambda rgb_uint32: (
#     (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
# )
# # convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
# #     int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
# # )

# class DLO_Grasp():
#     def __init__(self):
#         print("DLO_GraspPoseInCam")
        
#         #=====Parameters Setting=====#
#         self.checkpoint_path = os.path.join(ROOT_DIR, 'logs/log_rs/checkpoint-rs.tar') #"/home/iclab/work/graspnet-baseline/logs/log_rs/checkpoint-rs.tar"
#         self.num_point = 20000
#         self.num_view = 300
#         self.collision_thresh = 0.05
#         self.voxel_size = 0.005
#         #=====Parameters Setting=====#
#         self.net = self.get_net()

#         # self.o3d_vis = o3d.visualization.Visualizer()
#         # self.o3d_vis.create_window()
#         # self.vis_cloud = o3d.geometry.PointCloud()
#         # self.o3d_vis.add_geometry(self.vis_cloud)#[cloud, *grippers])
#         # # self.vis_grasp = o3d.geometry.li
#         # # o3d.visualization.draw_geometries([cloud, *grippers])

#         rospy.init_node('DLO_GraspPoseInCam', anonymous=False)
#         # #----1topic---#
#         # rospy.Subscriber('/dlo_cloudInCam', PointCloud2, self.dlo_grasp_cb)
 
#         # #----2topics---#
#         # dlo_cloudInCam_sub = message_filters.Subscriber('/dlo_cloudInCam', PointCloud2)
#         # ori_cloudInCam_sub = message_filters.Subscriber('/ori_cloudInCam', PointCloud2)

#         # ts = message_filters.ApproximateTimeSynchronizer([dlo_cloudInCam_sub, ori_cloudInCam_sub], queue_size=10, slop=10, allow_headerless=False)
#         # ts.registerCallback(self.callback)

#         # self.dlo_graspInCam_pub = rospy.Publisher('/dlo_graspInCam', PoseStamped)
#         self.scene_cloud_pub = rospy.Publisher('open3d_point_cloud', PointCloud2, queue_size=10)
#         self.depth_pub = rospy.Publisher('ddepth', sensor_msgs.msg.Image,queue_size=10)
#         self.color_pub = rospy.Publisher('ccolor', sensor_msgs.msg.Image,queue_size=10)

#         # self.dlo_graspInCam_br = tf.TransformBroadcaster()

#         #======Method2=======#
#         # Use Saved Images
#         #====================#
#         data_dir = os.path.join(ROOT_DIR, 'doc/example_data')#'doc/example_data' #'doc/motor_data/test6' #1~6
#         self.get_and_process_data(data_dir)
    
#         # while not rospy.is_shutdown():
#         #     # Publish the image message
#         #     pub.publish(image_msg)
            
#         #     # Sleep to maintain the publishing rate
#         #     rate.sleep()
#         rospy.spin()


#     # def convert_cloud_from_open3d_to_ros(self, open3d_cloud, frame_id="map"):

#     #     points = np.asarray(open3d_cloud.points)
#     #     if len(points) == 0:
#     #         return None

#     #     if open3d_cloud.has_colors():
#     #         colors = np.asarray(open3d_cloud.colors)
#     #         colors = np.floor(colors * 255).astype(np.uint8)
#     #         colors = np.left_shift(colors[:, 0], 16) + np.left_shift(colors[:, 1], 8) + colors[:, 2]
#     #         points = np.c_[points, colors]
            
#     #         fields = [
#     #             PointField('x', 0, PointField.FLOAT32, 1),
#     #             PointField('y', 4, PointField.FLOAT32, 1),
#     #             PointField('z', 8, PointField.FLOAT32, 1),
#     #             PointField('rgb', 12, PointField.UINT32, 1)
#     #         ]
#     #     else:
#     #         fields = [
#     #             PointField('x', 0, PointField.FLOAT32, 1),
#     #             PointField('y', 4, PointField.FLOAT32, 1),
#     #             PointField('z', 8, PointField.FLOAT32, 1)
#     #         ]

#     #     header = std_msgs.msg.Header()
#     #     header.stamp = rospy.Time.now()
#     #     header.frame_id = frame_id

#     #     # # Extract points and optionally colors
#     #     # points = np.asarray(open3d_cloud.points)
#     #     # if len(points) == 0:
#     #     #     return None
#     #     # else:
#     #     #     print("total points:", len(points))

#     #     # fields = [
#     #     #     PointField('x', 0, PointField.FLOAT32, 1),
#     #     #     PointField('y', 4, PointField.FLOAT32, 1),
#     #     #     PointField('z', 8, PointField.FLOAT32, 1)
#     #     # ]

#     #     # header = std_msgs.msg.Header()
#     #     # header.stamp = rospy.Time.now()
#     #     # header.frame_id = frame_id

#     #     # if open3d_cloud.has_colors():
#     #     #     colors = np.floor(np.asarray(open3d_cloud.colors) * 255)
#     #     #     points = np.hstack([points, colors])
#     #     #     fields.append(PointField('rgb', 12, PointField.UINT32, 1))

#     #     cloud_data = point_cloud2.create_cloud(header, fields, points)
#     #     # cloud_data=[]
#     #     return cloud_data

#     # # The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
#     # FIELDS_XYZ = [
#     #     PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
#     #     PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
#     #     PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
#     # ]
#     # FIELDS_XYZRGB = FIELDS_XYZ + \
#     #     [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

#     # # Bit operations
#     # BIT_MOVE_16 = 2**16
#     # BIT_MOVE_8 = 2**8
#     # convert_rgbUint32_to_tuple = lambda rgb_uint32: (
#     #     (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
#     # )
#     # convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
#     #     int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
#     # )

#     # Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)
#     def convert_cloud_from_open3d_to_ros(self, open3d_cloud, frame_id="map"):
#         # Set "header"
#         header = std_msgs.msg.Header()
#         header.stamp = rospy.Time.now()
#         header.frame_id = frame_id

#         # Set "fields" and "cloud_data"
#         points=np.asarray(open3d_cloud.points)
#         print("points\n", points)
#         if not open3d_cloud.colors: # XYZ only
#             print("noCOLORRR")
#             fields=FIELDS_XYZ
#             cloud_data=points
#         else: # XYZ + RGB
#             print("COLORRR")
#             fields=FIELDS_XYZRGB
#             print("fields\n", fields)
#             # -- Change rgb color from "three float" to "one 24-byte int"
#             # 0x00FFFFFF is white, 0x00000000 is black.
#             colors = np.floor(np.asarray(open3d_cloud.colors)*255) # nx3 matrix
#             colors = colors[:,0] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,2]  
#             cloud_data=np.c_[points, colors]
#             # print(np.c_[points, colors])
        
#         # create ros_cloud
#         return point_cloud2.create_cloud(header, fields, cloud_data)

#     def get_net(self):
#         """_summary_

#         Returns:
#             _type_: _description_
#         """        
#         # Init the model
#         net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
#                 cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        
#         # Check device (GPU/CPU)
#         #https://www.cnblogs.com/xiaodai0/p/10413711.html
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         net.to(device)

#         print("torch.cuda.is_available() = {}, use device: {}".format(torch.cuda.is_available(), device))
        
#         # Load model checkpoint
#         if torch.cuda.is_available() == False:
#             print('use cpu')
#             checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
#         else:
#             print('use gpu')
#             checkpoint = torch.load(self.checkpoint_path)
#         net.load_state_dict(checkpoint['model_state_dict'])
#         start_epoch = checkpoint['epoch']
#         print("-> loaded checkpoint %s (epoch: %d)"%(self.checkpoint_path, start_epoch))
        
#         # Set model to Eval mode
#         net.eval()
        
#         return net
    
#     def process_cloud(self, gen):
#         """_summary_

#         Args:
#             gen (_type_): _description_

#         Returns:
#             _type_: _description_
#         """        
#         # https://answers.ros.org/question/344096/subscribe-pointcloud-and-convert-it-to-numpy-in-python/
#         print("process_cloud")
#         # print("print(type(gen)):", type(gen))
#         # # print(gen)
#         cloud_masked = np.empty([1,3])
#         color_masked = np.empty([1,3])
#         for idx, g in enumerate(gen):

#             #==point (x, y, z)==#
#             pt = np.array([[g[0], g[1], g[2]]])
#             cloud_masked = np.append(cloud_masked,[[g[0], g[1], g[2]]], axis = 0)

#             #==rgb (r, g, b)==#
#             # cast float32 to int so that bitwise operations are possible
#             s = struct.pack('>f' ,g[3])
#             i = struct.unpack('>l',s)[0]
#             # you can get back the float value by the inverse operations
#             pack = ctypes.c_uint32(i).value
#             r = (pack & 0x00FF0000)>> 16
#             g = (pack & 0x0000FF00)>> 8
#             b = (pack & 0x000000FF)
#             color_masked = np.append(color_masked,[[r,g,b]], axis = 0)

#         # print("cloud_masked: ", type(cloud_masked), cloud_masked.shape)
#         # print("color_masked: ", type(color_masked), color_masked.shape)
#         # print(len(cloud_masked), self.num_point)
#         # print(type(len(color_masked)), type(self.num_point))
        
#         # sample points
#         if len(cloud_masked) >= self.num_point:
#             print('a')
#             idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
#         else:
#             print('b')
#             idxs1 = np.arange(len(cloud_masked))
#             idxs2 = np.random.choice(len(cloud_masked), self.num_point-len(cloud_masked), replace=True)
#             idxs = np.concatenate([idxs1, idxs2], axis=0)
#         cloud_sampled = cloud_masked[idxs]
#         color_sampled = color_masked[idxs]

#         # print("cloud_sampled: ", type(cloud_sampled), cloud_sampled.shape)
#         # print("color_sampled: ", type(color_sampled), color_sampled.shape)

#         # convert data
#         cloud = o3d.geometry.PointCloud()
#         cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
#         cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
#         end_points = dict()
#         cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         cloud_sampled = cloud_sampled.to(device)
#         end_points['point_clouds'] = cloud_sampled
#         end_points['cloud_colors'] = color_sampled

#         return end_points, cloud
    
#     def convert_numpy_to_pointcloud2(self, points, frame_id="map"):
#         """
#         Convert a numpy array to a sensor_msgs/PointCloud2 message.
        
#         :param points: Nx3 or Nx6 numpy array. If Nx3, it should contain [x, y, z].
#                     If Nx6, it should contain [x, y, z, r, g, b] with r, g, b in range [0, 1].
#         :param frame_id: Frame ID to be used in the header of the PointCloud2 message.
#         :return: PointCloud2 message
#         """
#         assert points.shape[2] == 3 or points.shape[2] == 6, "Input should be Nx3 or Nx6 numpy array"

#         fields = [
#             PointField('x', 0, PointField.FLOAT32, 1),
#             PointField('y', 4, PointField.FLOAT32, 1),
#             PointField('z', 8, PointField.FLOAT32, 1)
#         ]
#         print(points.shape[2], points[0,:])
#         if points.shape[2] == 6:
#             points = np.hstack([points[:, :3], (points[:, 3:6] * 255).astype(np.uint8).view(np.uint32)])
#             fields.append(PointField('rgb', 12, PointField.UINT32, 1))

#         header = std_msgs.msg.Header()
#         header.stamp = rospy.Time.now()
#         header.frame_id = frame_id
#         rospy.sleep(1.0) 

#         return point_cloud2.create_cloud(header, fields, points)
    
#     def get_and_process_data(self, data_dir):
#         # load data
#         color = cv2.imread(os.path.join(data_dir, 'color.png'), cv2.IMREAD_COLOR)
#         depth = cv2.imread(os.path.join(data_dir, 'depth.png'), cv2.IMREAD_COLOR)

#         # color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
#         # depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
#         workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
#         # meta = scio.loadmat("/home/iclab/work/graspnet-baseline/doc/example_data/meta.mat") #(os.path.join(data_dir, 'meta.mat'))
#         meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))

#         print("color: ", os.path.join(data_dir, 'color.png'))
#         print("depth: ", os.path.join(data_dir, 'depth.png'))
#         print("mask: ", os.path.join(data_dir, 'workspace_mask.png'))
#         print("meta: ", os.path.join(data_dir, 'meta.mat'))
#         bridge = CvBridge()
#         encoding="rgb8"
#         color_image_msg = bridge.cv2_to_imgmsg(color, encoding)
#         depth_image_msg = bridge.cv2_to_imgmsg(depth, encoding)

#         rate = rospy.Rate(1)  # 1 Hz
#         while not rospy.is_shutdown():
#             # Replace 'path_to_your_image.png' with your actual image path
#             # self.publish_color_image('path_to_your_image.png')
#             self.depth_pub(depth_image_msg)
#             self.color_pub(color_image_msg)
#             rate.sleep()
#         # self.color_pub(color_image_msg)
#         # self.depth_pub(depth_image_msg)

#         intrinsic = meta['intrinsic_matrix']
#         factor_depth = meta['factor_depth']

#         # print('meta:\n', meta)
#         # print('intrinsic:\n', intrinsic)
#         # print('factor_depth:\n', factor_depth)
#         # print('workspace_mask:\n', workspace_mask) # True, False

#         ##====example_data====##
#         # intrinsic:
#         # [[631.54864502   0.         638.43517329]
#         # [  0.         631.20751953 366.49904066]
#         # [  0.           0.           1.        ]]
#         # factor_depth: [[1000.]]

#         ##====motor_data====##
#         #intrinsic: 
#         # [[919.16595459   0.         641.86254883]
#         #  [  0.         919.43945312 366.82495117]
#         #  [  0.           0.           1.        ]]
#         # factor_depth: [[1000]]


#         # generate cloud
#         camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
#         cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

#         # # get valid points
#         # mask = (workspace_mask & (depth > 0))
#         # cloud_masked = cloud[mask]
#         # color_masked = color[mask]

#         print("cloud:", type(cloud), cloud.shape) #<class 'numpy.ndarray'>
#         # print("mask:", type(mask), mask.shape)
#         # print("cloud_masked:", type(cloud_masked), cloud_masked.shape)
#         # print("color_masked:", type(color_masked), color_masked.shape)

#         # # cloud: <class 'numpy.ndarray'> (720, 1280, 3)
#         # # mask: <class 'numpy.ndarray'> (720, 1280)
#         # # cloud_masked: <class 'numpy.ndarray'> (513688, 3)
#         # # color_masked: <class 'numpy.ndarray'> (513688, 3)
#         # print(cloud_masked)

#         # # sample points
#         # if len(cloud_masked) >= self.num_point:
#         #     idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
#         # else:
#         #     idxs1 = np.arange(len(cloud_masked))
#         #     idxs2 = np.random.choice(len(cloud_masked), self.num_point-len(cloud_masked), replace=True)
#         #     idxs = np.concatenate([idxs1, idxs2], axis=0)
#         # cloud_sampled = cloud_masked[idxs]
#         # color_sampled = color_masked[idxs]

#         # # convert data
#         # cloud = o3d.geometry.PointCloud()
#         # cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
#         # cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
#         # end_points = dict()
#         # cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
#         # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         # cloud_sampled = cloud_sampled.to(device)
#         # end_points['point_clouds'] = cloud_sampled
#         # end_points['cloud_colors'] = color_sampled

#         cloud_data = self.convert_numpy_to_pointcloud2(cloud, frame_id="map")
#         # cloud_data = self.convert_cloud_from_open3d_to_ros(cloud, frame_id="map") #world
#         self.scene_cloud_pub(cloud_data)

#         return []#end_points, cloud

#     def get_grasps(self, end_points):
#         """_summary_

#         Args:
#             end_points (_type_): _description_

#         Returns:
#             _type_: _description_
#         """        
#         print("end_points: ", end_points)

#         # Forward pass
#         with torch.no_grad():
#             end_points = self.net(end_points)
#             grasp_preds = pred_decode(end_points)
#         gg_array = grasp_preds[0].detach().cpu().numpy()
#         gg = GraspGroup(gg_array)
#         return gg

#     def collision_detection(self, gg, cloud):
#         """_summary_

#         Args:
#             gg (_type_): _description_
#             cloud (_type_): _description_

#         Returns:
#             _type_: _description_
#         """        
#         mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
#         collision_mask = mfcdetector.detect(gg, approach_dist=0.15, collision_thresh=self.collision_thresh)
#         gg = gg[~collision_mask]
#         return gg

#     def vis_grasps(self, gg, cloud):
#         """_summary_

#         Args:
#             gg (_type_): _description_
#             cloud (_type_): _description_
#         """        
#         gg.nms()
#         gg.sort_by_score()
#         gg = gg[:25]
#         grippers = gg.to_open3d_geometry_list()
#         o3d.visualization.draw_geometries([cloud, *grippers])

#         # self.o3d_vis.update_geometry()
#         # self.vis_cloud = cloud
#         # self.o3d_vis.poll_events()
#         # self.o3d_vis.update_renderer()

#     def dlo_grasp_cb(self, cloud_msg):
#         """_summary_

#         Args:
#             cloud_msg (_type_): _description_
#         """        
#         print("dlo_grasp_cb:")
#         assert isinstance(cloud_msg, PointCloud2)

#         gen = point_cloud2.read_points_list(cloud_msg, skip_nans=False)
#         if gen:
#             end_points, cloud_o3d = self.process_cloud(gen)
    
#         gg = self.get_grasps(end_points)
#         if self.collision_thresh > 0:
#             gg = self.collision_detection(gg, np.array(cloud_o3d.points))
#         self.vis_grasps(gg, cloud_o3d)

    
#     def callback(self, dlo_msg, scene_msg):
#         """_summary_

#         Args:
#             dlo_msg (_type_): _description_
#             scene_msg (_type_): _description_
#         """        

#         print("dlo_grasp_cb:")
#         # print(dlo_msg.header.stamp)
#         # print(scene_msg.header.stamp)
#         assert isinstance(dlo_msg, PointCloud2)
#         start = time.time()
   
#         dlo_gen = point_cloud2.read_points_list(dlo_msg, skip_nans=False)
#         if dlo_gen:
#             end_points, dlo_cloud_o3d = self.process_cloud(dlo_gen)

#         scn_gen = point_cloud2.read_points_list(scene_msg, skip_nans=False)
#         if scn_gen:
#             _, scn_cloud_o3d = self.process_cloud(scn_gen)
    
#         cloud_all = o3d.geometry.PointCloud()
#         points_tmp = np.concatenate((dlo_cloud_o3d.points, scn_cloud_o3d.points))
#         colors_tmp = np.concatenate((dlo_cloud_o3d.colors, scn_cloud_o3d.colors))
#         cloud_all.points = o3d.utility.Vector3dVector(points_tmp)
#         cloud_all.colors = o3d.utility.Vector3dVector(colors_tmp)

#         gg = self.get_grasps(end_points)
#         if self.collision_thresh > 0:
#             gg = self.collision_detection(gg, np.array(cloud_all.points))
        
#         # for t in range(np.size(gg)):
#         #     print(gg[t].score)

#         gg.nms()
#         gg.sort_by_score()
#         # print("=========")
#         # for t in range(np.size(gg)):
#         #     print(gg[t].score)

#         # print('type(gg):', type(gg), np.size(gg)) #<class 'graspnetAPI.grasp.GraspGroup'>
#         # print(type(gg[0]))#<class 'graspnetAPI.grasp.Grasp'> :Grasp(score, width, height, depth, translation, rotation_matrix, object_id)
#         print('score: ', gg[0].score)
#         print('width: ', gg[0].width)
#         print('height: ', gg[0].height)
#         print('depth: ', gg[0].depth)
#         print('translation: ', gg[0].translation)
#         print('rotation_matrix: ', gg[0].rotation_matrix)
#         print('object_id: ', gg[0].object_id)

#         #Grasp: score:0.0945352166891098, width:0.04796336218714714, height:0.019999999552965164, depth:0.009999999776482582, translation:[0. 0. 0.]
#         # rotation:
#         # [[ 8.5445970e-01  3.7061375e-02 -5.1819408e-01]
#         # [-5.1770735e-01 -2.2455113e-02 -8.5526299e-01]
#         # [-4.3333333e-02  9.9906063e-01 -4.3670326e-08]]
#         # object id:-1

#         # # see /graspnetAPI/docs/source/example_eval.rst
#         # gg=GraspGroup(np.array([[score_1, width_1, height_1, depth_1, rotation_matrix_1(9), translation_1(3), object_id_1],
#         #                         ...,
#         #                         [score_N, width_N, height_N, depth_N, rotation_matrix_N(9), translation_N(3), object_id_N]]))
#         # gg.save_npy(save_path)

#         #--publish Pose (objInCam) cam_H_obj--#
#         dlo_graspInCam = PoseStamped()
#         dlo_graspInCam.header.stamp = dlo_msg.header.stamp#rospy.Time.now()
#         dlo_graspInCam.header.frame_id = 'camera_color_optical_frame'

#         dlo_graspInCam.pose.position.x = gg[0].translation[0]
#         dlo_graspInCam.pose.position.y = gg[0].translation[1]
#         dlo_graspInCam.pose.position.z = gg[0].translation[2]

#         matrix4x4=np.identity(4)
#         matrix4x4[:3,:3]=gg[0].rotation_matrix
#         print('matrix4x4:', matrix4x4)
#         q = tf.transformations.quaternion_from_matrix(matrix4x4)
#         dlo_graspInCam.pose.orientation.x = q[0]
#         dlo_graspInCam.pose.orientation.y = q[1]
#         dlo_graspInCam.pose.orientation.z = q[2]
#         dlo_graspInCam.pose.orientation.w = q[3]
#         self.dlo_graspInCam_pub.publish(dlo_graspInCam)

#         print('dlo_graspInCam:', dlo_graspInCam)

#         # add a dlo_obj frame (cam_H_dlo)
#         self.dlo_graspInCam_br.sendTransform((dlo_graspInCam.pose.position.x, dlo_graspInCam.pose.position.y, dlo_graspInCam.pose.position.z),
#                                         (dlo_graspInCam.pose.orientation.x, dlo_graspInCam.pose.orientation.y, dlo_graspInCam.pose.orientation.z, dlo_graspInCam.pose.orientation.w),
#                                         dlo_graspInCam.header.stamp, #rospy.Time.now(),
#                                         "dlo_graspInCam_frame", 
#                                         "camera_color_optical_frame") #camera_link

#         end = time.time()
#         print("DLO grasp pose elapsed time: (sec)", end - start)
#         # self.vis_grasps(gg, cloud_all)

# if __name__ == '__main__':
#     # DLO_Grasp()
#     try:
#         DLO_Grasp()
#     except rospy.ROSInterruptException:
#         pass


#     # print("cloud_msg:")
#     # gen = point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
#     # time.sleep(1)

#     # print(type(gen))
#     # cnt=0
#     # for p in gen:
#     #     print("x:%.3f y:%.3f z:%.3f"%(p[0], p[1], p[2]))
#     #     cnt+=1
#     #     if cnt>10:
#     #         break