https://roboticsbackend.com/ros-service-command-line-tools-rosservice-and-rossrv/

rosservice list | grep dlo
/dlo_grasp_srv
/dlo_seg/get_loggers
/dlo_seg/set_logger_level
/dlo_seg_pose/get_loggers
/dlo_seg_pose/set_logger_level

user@33a40f5441a1:~/work$ rosservice info /dlo_grasp_srv
Node: /DLO_GraspPoseInCam
URI: rosrpc://172.18.0.3:38201
Type: dlo_srv/DloGraspSrv
Args: arrived_to_take_pic

user@33a40f5441a1:~/work$ rossrv show dlo_srv/DloGraspSrv 
bool arrived_to_take_pic
---
geometry_msgs/PoseStamped grasp_in_cam
  std_msgs/Header header
    uint32 seq
    time stamp
    string frame_id
  geometry_msgs/Pose pose
    geometry_msgs/Point position
      float64 x
      float64 y
      float64 z
    geometry_msgs/Quaternion orientation
      float64 x
      float64 y
      float64 z
      float64 w

rosservice call /dlo_grasp_srv "arrived_to_take_pic: True"
grasp_in_cam: 
  header: 
    seq: 0
    stamp: 
      secs: 1725123768
      nsecs: 735963696
    frame_id: "camera_color_optical_frame"
  pose: 
    position: 
      x: 0.022466322407126427
      y: 0.1633252501487732
      z: 0.33000001311302185
    orientation: 
      x: -0.04892290000403794
      y: -0.04718463791052606
      z: 0.1379871029541436
      w: 0.6987916611721167
