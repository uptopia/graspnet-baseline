#! /usr/bin/env python3

from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import rospy
import time

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
        
        # net = self.get_net()
        rospy.init_node('dlo_grasp_node', anonymous=True)
        rospy.Subscriber('/dlo_cloud_pub', PointCloud2, self.dlo_grasp_cb)
        rospy.spin()

    # def get_net(self):
    #     # Init the model
    #     net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
    #             cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        
    #     # Check device (GPU/CPU)
    #     #https://www.cnblogs.com/xiaodai0/p/10413711.html
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     net.to(device)

    #     print("torch.cuda.is_available() = {}, use device: {}".format(torch.cuda.is_available(), device))
        
    #     # Load model checkpoint
    #     if torch.cuda.is_available() == False:
    #         print('use cpu')
    #         checkpoint = torch.load(cfgs.checkpoint_path, map_location=torch.device('cpu'))
    #     else:
    #         print('use gpu')
    #         checkpoint = torch.load(cfgs.checkpoint_path)
    #     net.load_state_dict(checkpoint['model_state_dict'])
    #     start_epoch = checkpoint['epoch']
    #     print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
        
    #     # Set model to Eval mode
    #     net.eval()
        
    #     return net
    
    # def process_cloud(self, data_dir):
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
    #     if len(cloud_masked) >= cfgs.num_point:
    #         idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    #     else:
    #         idxs1 = np.arange(len(cloud_masked))
    #         idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
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
    #     mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    #     collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
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
        gen = point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        time.sleep(1)

        print(type(gen))
        cnt=0
        for p in gen:
            print("x:%.3f y:%.3f z:%.3f"%(p[0], p[1], p[2]))
            cnt+=1
            if cnt>10:
                break

        # #=====================
        # end_points, cloud = self.process_cloud(data_dir)
        # gg = self.get_grasps(net, end_points)
        # if cfgs.collision_thresh > 0:
        #     gg = self.collision_detection(gg, np.array(cloud.points))
        # self.vis_grasps(gg, cloud)

if __name__ == '__main__':

    try:
        DLO_Grasp()
    except rospy.ROSInterruptException:
        pass