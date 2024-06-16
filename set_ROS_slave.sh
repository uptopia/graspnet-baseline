#https://answers.ros.org/question/272065/specification-of-ros_master_uri-and-ros_hostname/
# #echo "export ROS_HOSTNAME=163.13.164.152" >> ~/.bashrc
# echo "export ROS_MASTER_URI=http://163.13.164.152:11311" >> ~/.bashrc
# echo "export ROS_IP=163.13.164.1" >> ~/.bashrc

echo "export ROS_MASTER_URI=http://172.25.0.2:11311" >> ~/.bashrc
echo "export ROS_IP=172.25.0.3" >> ~/.bashrc
source ~/.bashrc
