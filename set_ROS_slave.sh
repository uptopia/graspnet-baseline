#https://answers.ros.org/question/272065/specification-of-ros_master_uri-and-ros_hostname/
# #echo "export ROS_HOSTNAME=163.13.164.152" >> ~/.bashrc
# echo "export ROS_MASTER_URI=http://163.13.164.152:11311" >> ~/.bashrc
# echo "export ROS_IP=163.13.164.1" >> ~/.bashrc

master_ip=$1
ros_ip=$2

echo "master_ip: ${master_ip}"
echo "ros_ip:    ${ros_ip}"

echo "export ROS_MASTER_URI=http://${master_ip}:11311" >> ~/.bashrc
echo "export ROS_IP=${ros_ip}" >> ~/.bashrc
source ~/.bashrc