#!/usr/bin/env python
import rospy
import matplotlib.pyplot as plt

from std_msgs.msg import String
import message_filters
from sensor_msgs.msg import Imu, LaserScan

import rospy

from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
from Dijktra import *
from geometry_msgs.msg import Twist, Point, Quaternion

from std_msgs.msg import Float32MultiArray, UInt8MultiArray, MultiArrayDimension
from utils import*

gx = 0;
gy = 0;
x = 0;
y = 0;

moving = False;

ax = plt.subplot(111)

posPub = rospy.Publisher('plannar', Float32MultiArray, queue_size=10)

rot = get_so2(2*np.pi/2)
uncertainty = [384**2]
def callback0(odom):
    # rospy.loginfo(rospy.get_caller_id() + 'odom  %s', str(odom.pose.pose.position.x))
    global gx, gy, x, y
    x = odom.pose.pose.position.x
    y = odom.pose.pose.position.y

    # cor = np.array([[x], [y]])
    # cor = np.matmul(rot, cor)
    
    gx = (-x) // 0.05 + 184
    gy = (-y) // 0.05 + 184
    # print(gx)
    # print(gy)
    twist = odom.twist.twist

    if twist.linear.x**2 + twist.linear.y**2 + twist.linear.z**2 + twist.angular.x**2 + twist.angular.y**2 + twist.angular.z**2 < 5e-5:
        global moving
        moving = False;
    else:
        global moving
        moving = True;  

    # print("moving: " + str(moving))

def callback(map):
    # rospy.loginfo(rospy.get_caller_id() + 'map  %s', str(map.header))

    w_, h_ = map.info.width, map.info.height

    nmap_list = np.asarray(map.data)

    nmap = np.flip(np.rot90(np.reshape(nmap_list, (h_, w_))), axis=1);
    # nmap = np.reshape(nmap_list, (h_, w_));

    # print(map.info.origin.position.x, )

    if not moving:
        planner = Dijkstra(nmap, np.array([x, y]));

        routx, routy = planner.explore();
        ids = np.concatenate((routx, routy), axis=None)
        posmsg = Float32MultiArray()
        posmsg.data = ids.reshape([len(routx) * 2])
        posmsg.layout.data_offset = 0 

        # create two dimensions in the dim array
        posmsg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]

        # dim[0] is the vertical dimension of your matrix
        posmsg.layout.dim[0].label = "rows"
        posmsg.layout.dim[0].size = 2
        posmsg.layout.dim[0].stride = 2 * len(routx)
        # dim[1] is the horizontal dimension of your matrix
        posmsg.layout.dim[1].label = "columns"
        posmsg.layout.dim[1].size = len(routx)
        posmsg.layout.dim[1].stride = len(routx)
        print("route published")
        posPub.publish(posmsg)
        plt.imshow(nmap)
        # plt.imshow(planner.map)
        uncertainty.append(planner.uncertainty)
        plt.scatter(gy, gx, s = 20, color='g', marker='o')

        # # plt.plot(-np.asarray(routy)//0.05 + 184, -np.asarray(routx)//0.05 + 184, 'r.', markersize = 2);
        plt.plot(planner.unsy, planner.unsx, 'r.', markersize = 2);
        print(uncertainty)
        plt.pause(0.01)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    rospy.init_node('mapper', anonymous=True)
    rospy.loginfo("registered")

    rospy.Subscriber("/odom", Odometry, callback0)
    rospy.Subscriber("/map", OccupancyGrid, callback)


    # rospy.spin();
    plt.show(block=True)

    # spin() simply keeps python from exiting until this node is stopped

if __name__ == '__main__':
    print("mapper start listening");
    # initial particle filter
    # update map 
    # publish next desitantion with information gain    
    listener();

