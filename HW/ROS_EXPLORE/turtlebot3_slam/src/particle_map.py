#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import message_filters
from sensor_msgs.msg import Imu, LaserScan
from nav_msgs.msg import Odometry

from std_msgs.msg import Float32MultiArray
from std_msgs.msg import UInt8MultiArray
from std_msgs.msg import MultiArrayDimension
import numpy as np
from slam import *

particles_num = 1;

slam = slam_t(resolution=0.05);
mapPub = rospy.Publisher('mapNode', UInt8MultiArray, queue_size=10)
posPub = rospy.Publisher('posNode', Float32MultiArray, queue_size=10)
ax = plt.subplot(111)

def callback0(odom):
    slam.dynamics_step(odom);


def callback1(laser):
    # print(laser)
    slam.observation_step(laser);

    # do correction 

    # publish the map

    x, y, _ = slam.cur

    gx, gy = slam.map.grid_cell_from_xy(np.array([x]), np.array([y]))

    plt.imshow(slam.map.cells)
    plt.scatter(gx,gy, s = 0.5, color='r', marker='.')
    # plt.scatter(600,750, color='r')

    # print("converted pos: " + str(gx) + " || " + str(gy))

    plt.pause(0.01)
    return;

def callback(odom, laser):
    # rospy.loginfo(rospy.get_caller_id() + 'Imu  %s', str(odom))
    # rospy.loginfo(rospy.get_caller_id() + 'Laser  %s', str(laser))
    # slam.dynamics_step(odom);
    # slam.observation_step(laser);
    # mapmsg = UInt8MultiArray();

    # mapmsg.data = list(slam.map.cells.reshape([slam.map.numOfGrid]));

    # mapmsg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]

    # # dim[0] is the vertical dimension of your matrix
    # mapmsg.layout.dim[0].label = "rows"
    # mapmsg.layout.dim[0].size = slam.map.szx
    # mapmsg.layout.dim[0].stride = slam.map.numOfGrid
    # # dim[1] is the horizontal dimension of your matrix
    # mapmsg.layout.dim[1].label = "columns"
    # mapmsg.layout.dim[1].size = slam.map.szy
    # mapmsg.layout.dim[1].stride = slam.map.szy

    # posmsg = Float32MultiArray()
    # posmsg.data = slam.cur.reshape([3])
    # posmsg.layout.data_offset = 0 

    # # create two dimensions in the dim array
    # posmsg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]

    # # dim[0] is the vertical dimension of your matrix
    # posmsg.layout.dim[0].label = "rows"
    # posmsg.layout.dim[0].size = 2
    # posmsg.layout.dim[0].stride = 6
    # # dim[1] is the horizontal dimension of your matrix
    # posmsg.layout.dim[1].label = "columns"
    # posmsg.layout.dim[1].size = 3
    # posmsg.layout.dim[1].stride = 3 

    # # output current pos in the grid to node
    # mapPub.publish(mapmsg)
    # posPub.publish(posmsg)

    # output the currentMap to the subscriber
    x, y, _ = slam.cur
    # plt.imshow(slam.map.cells)
    # print("pos: " + str(x) + " || " + str(y))
    gx, gy = slam.map.grid_cell_from_xy(np.array([x]), np.array([y]))
    # plt.scatter(gx,gy,color='r')


    # print(slam.cur)
    # plt.pause(0.01)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    rospy.init_node('particle_map', anonymous=True)

    # image_sub = message_filters.Subscriber('odom', Odometry)
    # info_sub = message_filters.Subscriber('scan', LaserScan)

    # ts = message_filters.TimeSynchronizer([image_sub, info_sub], 10)
    # ts.registerCallback(callback)
    rospy.loginfo(rospy.get_caller_id() + 'registered callback')

    rospy.Subscriber('odom', Odometry, callback0)
    rospy.Subscriber('scan', LaserScan, callback1)


    # rospy.spin();
    plt.show(block=True)


if __name__ == '__main__':
    print("start listening");
    # initial particle filter
    # update map 
    # publish next desitantion with information gain

    slam.init_particles(n = particles_num);
    
    listener();

