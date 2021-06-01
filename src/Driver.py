#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class driver:
    def __init__(self):
        print("twist")
        self.move = Twist()  # Creates a Twist message type object
        print("init node")
        rospy.init_node('avoid_obs_node')  # Initializes a node

        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        # Publisher object which will publish "Twist" type messages
        # on the "/cmd_vel" Topic, "queue_size" is the size of the
        # outgoing message queue used for asynchronous publishing

        print("cmd vel published")
        rospy.spin()  # Loops infinitely until someone stops the program execution
        print("end spin")

    def follow(self, x:float, speed:float):
        self.move.linear.x=speed
        self.move.angular.z=x
        self.pub.publish(self.move)

