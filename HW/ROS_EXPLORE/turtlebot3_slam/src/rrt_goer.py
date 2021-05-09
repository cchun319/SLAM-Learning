#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import message_filters
from sensor_msgs.msg import Imu, LaserScan

import rospy
from rospy_tutorials.msg import Floats

from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float32MultiArray
import numpy as np
from slam import *

slam = slam_t(resolution=0.05);
pub = rospy.Publisher('NextExplore', numpy_msg(Floats), queue_size=10)
ax = plt.subplot(111)

def callback(map):
    # rospy.loginfo(rospy.get_caller_id() + 'Imu  %s', str(imu))
    # rospy.loginfo(rospy.get_caller_id() + 'Laser  %s', str(laser))


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    rospy.init_node('INFOGAIN', anonymous=True)

    image_sub = message_filters.Subscriber('imu', Imu)
    info_sub = message_filters.Subscriber('scan', LaserScan)

    ts = message_filters.TimeSynchronizer([image_sub, info_sub], 10)
    ts.registerCallback(callback)


    # rospy.spin();
    plt.show(block=True)

    # spin() simply keeps python from exiting until this node is stopped


if __name__ == '__main__':
    print("start listening");
    # initial particle filter
    # update map 
    # publish next desitantion with information gain

    slam.init_particles(n = 100);
    
    listener();

