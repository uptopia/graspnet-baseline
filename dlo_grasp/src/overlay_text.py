#! /usr/bin/env python3

# sudo apt update
# sudo apt install ros-noetic-jsk-rviz-plugins
# Rviz Add jsk_rviz_plugins > OverlayText > choose rostopic

import rospy
import math
from std_msgs.msg import ColorRGBA, Float32
from jsk_rviz_plugins.msg import OverlayText

class OverlayTextNode():

    def __init__(self):
        rospy.init_node("overlay_sample")

        self.text_pub = rospy.Publisher("text_sample", OverlayText, queue_size=1)
        self.value_pub = rospy.Publisher("value_sample", Float32, queue_size=1)
        self.counter = 0
        self.rate = 100
        self.r = rospy.Rate(self.rate)

    def run(self):
        while not rospy.is_shutdown():
            self.counter = self.counter + 1

            text = OverlayText()

            text.width = 400
            text.height = 400
            text.left = 10
            text.top = 10

            text.text_size = 12
            text.line_width = 2
            text.font = "DejaVu Sans Mono"

            text.text = """This is OverlayText plugin.
            The update rate is %d Hz.
            You can write several text to show to the operators.
            New line is supported and automatical wrapping text is also supported.
            And you can choose font, this text is now rendered by '%s'

            You can specify background color and foreground color separatelly.

            Of course, the text is not needed to be fixed, see the counter: %d.

            You can change text color like <span style="color: red;">this</span>
            by using <span style="font-style: italic;">css</style>.
            """ % (self.rate, text.font, self.counter)

            text.fg_color = ColorRGBA(25 / 255.0, 1.0, 240.0 / 255.0, 1.0)
            text.bg_color = ColorRGBA(0.0, 0.0, 0.0, 0.2)

            self.text_pub.publish(text)
            self.value_pub.publish(math.sin(self.counter * math.pi * 2 / 100))

            if int(self.counter % 500) == 0:
                rospy.logdebug('This is ROS_DEBUG.')
            elif int(self.counter % 500) == 100:
                rospy.loginfo('This is ROS_INFO.')
            elif int(self.counter % 500) == 200:
                rospy.logwarn('This is ROS_WARN.')
            elif int(self.counter % 500) == 300:
                rospy.logerr('This is ROS_ERROR.')
            elif int(self.counter % 500) == 400:
                rospy.logfatal('This is ROS_FATAL.')
            self.r.sleep()

            if rospy.is_shutdown():
                break

if __name__ == '__main__':
    try:
        m = OverlayTextNode()
        m.run()

    except rospy.ROSInterruptException:
        pass
