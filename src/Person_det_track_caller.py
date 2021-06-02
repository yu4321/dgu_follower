#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""@author: ambakick
"""
import sys
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

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

import target

from person_tracker_core import Direction, Mode

W = 640
H = 480

# Global variables to be used by funcitons of VideoFileClop
frame_count = 0 # frame counter

debug = False

currentFollow = int

currentTarget:target.Target = None

trackerCore = tcore.PersonTrackerCore()
ncv2 = CvBridge()

lastMin = float("inf")
minCount = 10

currentTurn : Direction = Direction.Center
lastTurn : Direction = Direction.Center

currentMode: Mode = Mode.Chasing

waitStartedTime: time = None

isWorking=False

def pipeline(img, depth_img):
    global currentFollow
    global currentTarget
    global currentMode
    global currentTurn
    global lastTurn
    global waitStartedTime
    global isWorking

    if(isWorking == True):
        print('atomic blocked')
        return img
    try:
        isWorking = True

        detects = trackerCore.get_good_trackers(img)

        if currentMode == Mode.Chasing:
            isIdLost = True

            trk: tracker.Tracker
            for trk in detects:
                x_cv2 = trk.box
                # 사용 불가 박스는 넘김
                if (trk.isBoxValid() == False):
                    continue

                # 현재 타겟이 없을 경우 : 타겟 획득 행동
                if (currentTarget == None):
                    if (trk.score > 0.8):
                        RegisterTarget(trk, img)
                        currentFollow = trk.id

                # 현재 트래커의 id가 현재 추적 id와 같을 경우
                if (trk.id == currentFollow):
                    isIdLost = False
                    RefreshTargetData(trk, img, depth_img)
                    currentTurn = currentTarget.lastDirection
                    img = helpers.draw_box_label_Trac(trk, img, (0, 0, 255), True, currentTarget.latestDistance)

                else:
                    img = helpers.draw_box_label(trk.id, img, x_cv2)

            # id 소실시 follow는 제거됨. 추적 모드로 들어가야 하니까
            if (isIdLost):
                currentFollow = -1
                # currentMode = Mode.NearSearching
                if (currentTarget != None):
                    ChangeModeToFarSearching()
                    waitStartedTime = time.time()
            else:
                driveToTarget()
            #return img

        elif currentMode == Mode.FarSearching:
            if(lastTurn != Direction.Center):
                print('start stand turn to ',currentTarget.lastDirection)
                standTurn(currentTarget.lastDirection)
                lastTurn=Direction.Center

            trk: tracker.Tracker
            for trk in detects:
                # 사용 불가 박스는 넘김
                if (trk.isBoxValid() == False):
                    continue
                if (trk.score > 0.8):
                    RegisterTarget(trk, img)
                    currentFollow = trk.id
                    ChangeModeToChasing()
                    break
            if currentMode == Mode.FarSearching:
                if time.time() - waitStartedTime > 30:
                    DisposeTarget()
                    ChangeModeToChasing()

        # cv2.imshow("frame", img)
    finally:
        isWorking = False
        return img


def ChangeModeToChasing():
    global currentMode
    print('current mode changed to chasing')
    currentMode = Mode.Chasing

def ChangeModeToFarSearching():
    global currentMode
    print('current mode changed to farsearching')
    currentMode = Mode.FarSearching

def RefreshTargetData(trk:tracker.Tracker, img, depth_img):
    global currentTarget
    global lastMin
    x_cv2 = trk.box
    left, top, right, bottom = x_cv2[1], x_cv2[0], x_cv2[3], x_cv2[2]
    currentTarget.lastImg= img[top:bottom, left:right]
    currentTarget.latestTracker = trk

    tDistance = tcore.get_biggest_distance_of_box(depth_img,trk)
    if(tDistance ==0):
        currentTarget.latestDistance = lastMin
    else:
        currentTarget.latestDistance = tDistance
        lastMin=tDistance

    currentTarget.lastDirection = trackerCore.GetDirectionOfTracker(trk)
    return


def IdentifyTarget(trk : tracker.Tracker, img):
    if(trk.score > 0.8):
        return True

def RegisterTarget(trk: tracker.Tracker, img):
    global currentTarget

    currentTarget = target.Target()
    currentTarget.firstTracker = trk
    currentTarget.latestTracker = trk

    x_cv2 = trk.box
    left, top, right, bottom = x_cv2[1], x_cv2[0], x_cv2[3], x_cv2[2]
    currentTarget.firstImg = img[top:bottom, left:right]

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

def standTurn(direction : Direction):

    if (direction == Direction.Right):
        move.angular.z = -0.2
    elif direction == Direction.Left:
        move.angular.z = 0.2
    pub.publish(move)
    time.sleep(0.1)

def driveToTarget():

    global move
    global currentTarget
    global driveMode
    global lastTurn
    global currentTurn

    if(currentTarget != None):
        if(currentTarget.latestDistance < 500):
            print("so close. stop")
            move.linear.x = 0
        else:
            move.linear.x = 0.4

        if(currentTurn == Direction.Right):
            move.angular.z = -0.2
        elif currentTurn == Direction.Left:
            move.angular.z = 0.2

        # else:
        #     if(lastTurn == Direction.Right):
        #         move.angular.z = 0.05
        #     elif lastTurn == Direction.Left:
        #         move.angular.z = -0.05

        lastTurn = currentTurn

        if(driveMode == False):
            print('not drive mode')
            move.linear.x = 0

        pub.publish(move)

def DisposeTarget():
    global  currentTarget
    print('remove current target')
    currentTarget=None

def PrintCenterDistance(x, y, depth_img):
    pix = (x/ 2, y / 2)
    sys.stdout.write(
        'Depth at center(%d, %d): %f(mm)\r' % (pix[0], pix[1], depth_img[int(pix[1]), int(pix[0])]))
    sys.stdout.flush()

if __name__ == "__main__":
    print('current path : ',os.getcwd())
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

        print('will start loop now')

        frame_rate = 10
        prev = 0
        while(True):
            time_elapsed = time.time() - prev

            if time_elapsed > 1. / frame_rate:
                prev = time.time()

            else:
                continue

            msg=rospy.wait_for_message('/camera/color/image_raw',Image)
            data=rospy.wait_for_message('/camera/depth/image_rect_raw', Image)

            start=time.time()
            img=ncv2.imgmsg_to_cv2(msg, "bgr8")
            cv_image = ncv2.imgmsg_to_cv2(data, data.encoding)
            endcv2 = time.time() - start

            W=int(data.width)
            H=int(data.height)
            tcore.W = W
            tcore.H = H

            #PrintCenterDistance(W,H,cv_image)

            start = time.time()
            new_img = pipeline(img, cv_image)
            endpipeline=time.time()-start

            start = time.time()
            try:
                cv2.imshow('frame', new_img)
            except:
                continue

            endimshow=time.time()-start

            sys.stdout.write(
                'Time: cv2 %f, pipeline %f, imshow %f\r' % (endcv2,endpipeline,endimshow))
            sys.stdout.flush()

            pressed = cv2.waitKey(1) & 0xFF

            if pressed== ord('q'):
                print('exit with q')
                break
            if pressed == ord('s'):
                driveMode= not driveMode
                print('drive mode change to '+str(driveMode))
                if(driveMode == False):
                    DisposeTarget();
