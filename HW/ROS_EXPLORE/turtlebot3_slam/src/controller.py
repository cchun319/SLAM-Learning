#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
from geometry_msgs.msg import Twist, Point, Quaternion
import tf
from math import radians, copysign, sqrt, pow, pi, atan2
from tf.transformations import euler_from_quaternion
import numpy as np
import subprocess
import os
import time
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import UInt8MultiArray
from std_msgs.msg import MultiArrayDimension

msg = """
control your Turtlebot3!
-----------------------
Insert xyz - coordinate.
x : position x (m)
y : position y (m)
z : orientation z (degree: -180 ~ 180)
If you want to close, insert 's'
-----------------------
"""

class GotoPoint():
    def __init__(self):
        rospy.init_node('controller', anonymous=False)
        rospy.on_shutdown(self.shutdown)
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        # self.moving = rospy.Publisher('moving', Twist, queue_size=5)
        self.tf_listener = tf.TransformListener()
        self.odom_frame = 'odom'


        self.prevPos = np.zeros((3,1), dtype=np.float64)
        self.curPos = np.zeros((3,1), dtype=np.float64)

        try:
            self.tf_listener.waitForTransform(self.odom_frame, 'base_footprint', rospy.Time(), rospy.Duration(1.0))
            self.base_frame = 'base_footprint'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform(self.odom_frame, 'base_link', rospy.Time(), rospy.Duration(1.0))
                self.base_frame = 'base_link'
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Cannot find transform between odom and base_link or base_footprint")
                rospy.signal_shutdown("tf Exception")

        
    def moveTo(self, points):
        position = Point()
        move_cmd = Twist()
        r = rospy.Rate(10)

        self.curPos = self.get_odom()
        last_rotation = 0
        linear_speed = 1
        angular_speed = 1
        (goal_x, goal_y, goal_z) = points[0], points[1], points[2]
        if goal_z > 180 or goal_z < -180:
            print("you input wrong z range.")
            self.shutdown()
        goal_z = np.deg2rad(goal_z)
        goal_distance = sqrt(pow(goal_x - self.curPos[0], 2) + pow(goal_y - self.curPos[1], 2))
        distance = goal_distance

        while distance > 0.1:
            # print(self.curPos)
            self.curPos = self.get_odom()
            x_start = self.curPos[0]
            y_start = self.curPos[1]
            rotation = self.curPos[2]

            path_angle = atan2(goal_y - y_start, goal_x - x_start)

            if path_angle < -pi/4 or path_angle > pi/4:
                if goal_y < 0 and y_start < goal_y:
                    path_angle = -2*pi + path_angle
                elif goal_y >= 0 and y_start > goal_y:
                    path_angle = 2*pi + path_angle
            if last_rotation > pi-0.1 and rotation <= 0:
                rotation = 2*pi + rotation
            elif last_rotation < -pi + 0.1 and rotation > 0:
                rotation = -2*pi + rotation
            move_cmd.angular.z = angular_speed * path_angle - rotation

            distance = sqrt(pow((goal_x - x_start), 2) + pow((goal_y - y_start), 2))
            move_cmd.linear.x = min(linear_speed * distance, 0.1)

            if abs(move_cmd.angular.z) > 1e-2:
                move_cmd.angular.z = min(move_cmd.angular.z, 1.5)
            else:
                move_cmd.angular.z = max(move_cmd.angular.z, -1.5)

            last_rotation = rotation
            self.cmd_vel.publish(move_cmd)
            r.sleep()
        self.curPos = self.get_odom()

        while abs(self.curPos[2] - goal_z) > 0.1:
            self.curPos = self.get_odom()
            rotation = self.curPos[2]
            if goal_z >= 0:
                if rotation <= goal_z and rotation >= goal_z - pi:
                    move_cmd.linear.x = 0.00
                    move_cmd.angular.z = 0.25
                else:
                    move_cmd.linear.x = 0.00
                    move_cmd.angular.z = -0.25
            else:
                if rotation <= goal_z + pi and rotation > goal_z:
                    move_cmd.linear.x = 0.00
                    move_cmd.angular.z = -0.25
                else:
                    move_cmd.linear.x = 0.00
                    move_cmd.angular.z = 0.25
            self.cmd_vel.publish(move_cmd)
            r.sleep()
        # print(self.curPos)
        self.prevPos = self.curPos
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel.publish(Twist())

    def get_odom(self):

        # newdata = rospy.wait_for_message("posNode", Float32MultiArray);
        # while self.prevPos == newdata.data:
        #     # print("data is the same as old data!")
        #     newdata = rospy.wait_for_message("posNode", Float32MultiArray);
        #     rospy.sleep(0.2)
        # return newdata.data

        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            rotation = euler_from_quaternion(rot)

        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return


        return np.array([(Point(*trans)).x, (Point(*trans)).y, rotation[2]])


    def shutdown(self):
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

mover = GotoPoint();

def callback(dest):


    pos = np.array(dest.data)

    pos = np.reshape(pos, (dest.layout.dim[0].size, dest.layout.dim[1].size))
    prev = 0
    pos = np.vstack((pos, np.zeros((1, pos.shape[1]))));
    for j in range(pos.shape[1]):
        # self.cmd_vel = rospy.Publisher('cmd_vel', 'moving', queue_size=5)
        if j + 1 < pos.shape[1]:
            # print(np.arctan2(pos[1,j + 1] - pos[1,j], pos[0,j + 1], pos[0,j]))
            pos[2,j] = (atan2(pos[1,j + 1] - pos[1,j], pos[0,j + 1] - pos[0,j])) * 180.0 / np.pi;
            prev = pos[2,j]
        else:
            pos[2,j] = prev;

        print("going to: " + str(pos[:,j]))


        mover.moveTo(pos[:,j])

    # self.cmd_vel = rospy.Publisher('cmd_vel', 'arrived', queue_size=5)
    

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    rospy.Subscriber("/plannar", Float32MultiArray, callback)

    rospy.spin();
    # plt.show(block=True)

    # spin() simply keeps python from exiting until this node is stopped




if __name__ == '__main__':

    print("controller start listening");
    # initial particle filter
    # update map 
    # publish next desitantion with information gain    
    listener();
