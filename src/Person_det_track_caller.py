#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""@author: ambakick
"""
import sys
import time

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import matplotlib.pyplot as plt
import glob
#import NotCvBridge as ncv2
#from moviepy.editor import VideoFileClip
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment

import helpers
import detector
import tracker
import cv2

import person_tracker_core as tcore


W = 640
H = 480

# Global variables to be used by funcitons of VideoFileClop
frame_count = 0 # frame counter

debug = False

currentFollow = int

trackerCore=tcore.PersonTrackerCore()
ncv2=CvBridge()

lastMin = float("inf")
minCount = 10

    
def get_biggest_distance_of_box(image, depth_image, left, right, top, bottom):
    global minCount
    global lastMin
    #return 800
    #if(minCount<10):
    #    minCount=minCount+1
    #    return lastMin

    minCount=0
    if image is None :
        print('image not true. break')
        return
    if depth_image is None:
        print('depth image not True, break')
        return
    cuttedD=depth_image[top:bottom, left:right]
    #print(cuttedD)
    cuttedO=image[top:bottom, left:right]

    try:
        min = float("inf")
        x=np.array(cuttedD).flatten()
        mx=np.ma.masked_array(x, mask=x==0)
        min=mx.min()
    except:
        min=lastMin

    if image is None:
        print('image2 not true. break')
        return
    if depth_image is None:
        print('depth image2 not True, break')
        return
    #print('cuttedD : ',len(cuttedD), ' cuttedO : ', len(cuttedO))
    print('minimum distance : ',min,'mm')


    lastMin=min
    return min



def pipeline(img, depth_img):

    global debug
    global currentFollow

    detects = trackerCore.get_good_trackers(img)

    if len(detects) <=0:
        currentFollow = -1
        cv2.imshow('frame', img)
        return

    pos = float(0)
    distance = float(0)

    for trk in detects:
        x_cv2 = trk.box


        if (pos == 0):
            left, top, right, bottom = x_cv2[1], x_cv2[0], x_cv2[3], x_cv2[2]
            if ((right - left) * (bottom - top) <= 0):
                print('size of square smaller than 0. skip')
                continue
            # if(left-right <=0):
            #    continue

            center = left + ((right - left) / 2)
            posProto = center - (W / 2)
            pos = posProto / (W / 2)
            distance = get_biggest_distance_of_box(img, depth_img, left, right, top, bottom)

            print('center: ', center, "half width: ", W / 2, ", posProto: ", posProto, ", pos: ", pos)
            # pos = center/(W/2)

            if driveMode == True:
                if (currentFollow == -1):
                    currentFollow = trk.id
                    distance = get_biggest_distance_of_box(img, depth_img, left, right, top, bottom)
                    print('set target: id : ' + str(trk.id))
                else:
                    if currentFollow != trk.id:
                        print('not target. pass id : ' + str(trk.id))
                        pos = 0
            else:
                print('not drive Mode. set target none')
                currentFollow = -1
                pos = 0
                distance = get_biggest_distance_of_box(img, depth_img, left, right, top, bottom)
        else:
            left, top, right, bottom = x_cv2[1], x_cv2[0], x_cv2[3], x_cv2[2]
            distance = get_biggest_distance_of_box(img, depth_img, left, right, top, bottom)
            if ((right - left) * (bottom - top) <= 0):
                print('size of square smaller than 0. skip')
                continue

        if trk.id == currentFollow:
            print("currentfollow ",currentFollow, "trkid ",trk.id)
            img = helpers.draw_box_label(trk.id, img, x_cv2, box_color=(0,0,255), distance=distance)
        else:
            img = helpers.draw_box_label(trk.id, img, x_cv2)

    cv2.imshow("frame", img)
    print('target position - ' + str(pos), ' target distance : ',distance, 'mm')
    dst = float(distance)
    drive(pos, dst)
    return

def drive(pos, distance):
    global move
    if pos!=0:
        move.linear.x=0.5
    else:
        move.linear.x=0
    move.angular.z=pos * -0.5

    if(distance < 1000):
        print("so close. stop")
        move.linear.x=0
        #move.angular.z = move.angular.z*0.05

    pub.publish(move)

def image_callback(self, msg):
	lastMsg=msg
if __name__ == "__main__":    
    currentFollow=-1
    driveMode=False
    det = detector.PersonDetector()

    if debug: # test on a sequence of images
        images = [plt.imread(file) for file in glob.glob('./test_images/*.jpg')]
        
        for i in range(len(images))[0:7]:
             image = images[i]
             image_box = pipeline(image)   
             plt.imshow(image_box)
             plt.show()
           
    else:
        print('main started')
        move = Twist()
        print('twist')
        rospy.init_node('follower')
        print('node inited')
        pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        print('published')
        #image_sub = rospy.Subscriber('/camera/color/image_raw',Image, image_callback)
        print('subscribed')
        #rospy.spin()
        #print('ros spin start')
        #bridge=CvBridge()
        #cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print('will start loop now')

        frame_rate = 10
        prev = 0
        while(True):
            time_elapsed = time.time() - prev

            if time_elapsed > 1. / frame_rate:
                prev = time.time()

            else:
                continue
            #msg = lastMsg
            msg=rospy.wait_for_message('/camera/color/image_raw',Image)
            data=rospy.wait_for_message('/camera/depth/image_rect_raw', Image)
            #print('get img encoding : '+str(msg.encoding))
            #np_arr=np.fromstring(msg.data,np.uint8)
            #cap = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') 
            #ret, img = cap.read()
            img=ncv2.imgmsg_to_cv2(msg, "bgr8")
            cv_image = ncv2.imgmsg_to_cv2(data, data.encoding)

            #pix=(msg2.width/2, msg2.height/2)
            #print('Depth at center- ' + str(img2[pix[1], pix[0]]))
            #ret, img=cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            #print(img)

            pix = (data.width / 2, data.height / 2)
            W=int(data.width)
            H=int(data.height)
            sys.stdout.write(
                'Depth at center(%d, %d): %f(mm)\r' % (pix[0], pix[1], cv_image[int(pix[1]), int(pix[0])]))
            sys.stdout.flush()
            
            np.asarray(img)
            new_img = pipeline(img, cv_image)
            #cv2.imshow("frame2", img2)

            pressed = cv2.waitKey(1) & 0xFF
            if pressed== ord('q'):
                print('exit with q')
                break
            if pressed == ord('s'):
                driveMode= not driveMode
                print('drive mode change to '+str(driveMode))


        cap.release()
        cv2.destroyAllWindows()
        print(round(end-start, 2), 'Seconds to finish')
