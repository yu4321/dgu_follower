#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import sys
import time

import os

import rospy
from sensor_msgs.msg import Image, CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from darknet_ros_msgs.msg import BoundingBoxes
import matplotlib.pyplot as plt
import glob

import helpers
import detector
import tracker
import cv2

import person_tracker_core as tcore

import target

from target import LidarData

from person_tracker_core import Direction, Mode

# 현재 화면의 가로 세로
W = 640
H = 480

detectBaseScore = 0.3

# 얘네 둘은 무시
# Global variables to be used by funcitons of VideoFileClop
frame_count = 0 # frame counter

debug = False

driveMode = True

#현재 추적하는 물체의 Object ID
currentFollow = int

#현재 진짜로 추적하는 물체
currentTarget:target.Target = None

#추적 메인 모듈
trackerCore = tcore.PersonTrackerCore()

#로스파이 이미지에서 오픈씨비 이미지로 변환
ncv2 = CvBridge()

#마지막 최소 거리값. 측정 안될때 재사용하려고
lastMin = float("inf")

#현재의 회전 방향. rawDrive면 안 씀
currentTurn : Direction = Direction.Center
#마지막 회전 방향. rawDrive면 안 씀
lastTurn : Direction = Direction.Center

lastObstacleDirection : Direction = Direction.Center
lastObstacleScore = 0

#현재 작동 모드
currentMode: Mode = Mode.Chasing

#소실 후 타겟 파기까지 30초 기다릴 때 쓰는 변수
waitStartedTime: time = None

#pipeline 중복 실행 방지용
isWorking=False

#마지막으로 감지된 라이다 데이터. 파이프라인과 무관하게 계속 갱신됨
lastFrontLidarData : LidarData=None

lastImageData : Image = None
lastYoloData : BoundingBoxes = None
lastYoloAddedTime : time = None
lastDepthData : Image = None

def preTargetLoop(img, depth_img, darknets:BoundingBoxes):
    global currentFollow
    global currentMode

    if(darknets == None):
        print('preTarget - darknet void')
        return img

    detects = trackerCore.get_darknet_trackers(img,darknets)
    trk: tracker.Tracker
    for trk in detects:
        x_cv2 = trk.box
        # print('pt cur box : ',trk.box, ', id : ',trk.id, ' score:', trk.score)
        # 사용 불가 박스는 넘김
        if (trk.isBoxValid() == False):
            print('pt box not valid ', trk.box)
            continue
        # 현재 타겟이 없을 경우 : 타겟 획득 행동
        if (currentTarget == None):
            if (trk.score > detectBaseScore):
                RegisterTarget(trk, img)
                ChangeModeToChasing()
                currentFollow = trk.id
                RefreshTargetData(trk, img, depth_img)
                img = helpers.draw_box_label_Trac(trk, img, (0, 0, 255), True, currentTarget.latestDistance)
                continue
        img = helpers.draw_box_label(trk.id, img, x_cv2)
    return img


def chasingLoop(img, depth_img, darknets: BoundingBoxes):
    global currentFollow
    global currentTarget
    global currentMode
    global lastFrontLidarData
    global lastObstacleDirection
    global lastObstacleScore

    if (darknets == None):
        print('preTarget - darknet void')
        ChangeModeToNearSearching()
        return img

    isIdLost = True
    detects = trackerCore.get_darknet_trackers(img, darknets)
    trk: tracker.Tracker
    for trk in detects:
        x_cv2 = trk.box
        #print('cl cur box : ', trk.box, ', id : ', trk.id, ' score:', trk.score)
        # 사용 불가 박스는 넘김
        if (trk.isBoxValid() == False):
            print('cl box not valid ', trk.box)
            continue
        if (trk.id == currentFollow):
            isIdLost = False
            RefreshTargetData(trk, img, depth_img)
            img = helpers.draw_box_label_Trac(trk, img, (0, 0, 255), True, currentTarget.latestDistance)
            continue
        img = helpers.draw_box_label(trk.id, img, x_cv2)

    if (isIdLost):
        print('chasing -target not found')
        ChangeModeToNearSearching()
        return img

    nowObsInfo = lastFrontLidarData.GetObstacleScore()
    if(nowObsInfo.score != 0) :
        if (nowObsInfo.Direction == Direction.Center ):
            print('start large Stand Turn ',time.time())
            while(True):
                standTurn(Direction.Left)
                curObsInfo = lastFrontLidarData.GetObstacleScore()
                if(curObsInfo.Direction==Direction.Center and curObsInfo.score != 0):
                    continue
                else:
                    break
            print('end large Stand Turn ',time.time())
        else:
            lastObstacleDirection = nowObsInfo.Direction
            lastObstacleScore = nowObsInfo.score
    else:
        if(lastObstacleDirection!=Direction.Center):
            lastObstacleDirection =Direction.Center

    if(driveMode):
        newRawDrive()
    else:
        print('no drive mode')

    return img

def farSearchingLoop(img, depth_img, darknets:BoundingBoxes):
    global currentTarget
    global currentFollow
    global waitStartedTime
    if (darknets != None):
        detects = trackerCore.get_darknet_trackers(img, darknets)
        trk: tracker.Tracker
        for trk in detects:
            x_cv2 = trk.box
            print('fs cur box : ', trk.box, ', id : ', trk.id, ' score:', trk.score)
            # 사용 불가 박스는 넘김
            if (trk.isBoxValid() == False):
                print('fs box not valid ', trk.box)
                continue
            # 현재 타겟이 없을 경우 : 타겟 획득 행동
            if (isTargetFound == False):
                if (IdentifyTarget(trk, img)):
                    isTargetFound = True
                    RefreshTargetData(trk, img, depth_img)
                    ChangeModeToChasing()
                    currentFollow = trk.id
                    img = helpers.draw_box_label_Trac(trk, img, (0, 0, 255), True, currentTarget.latestDistance)
                    continue
            img = helpers.draw_box_label(trk.id, img, x_cv2)

    if(time,time() - waitStartedTime > 30):
        DisposeTarget()
    return img

def nearSearchingLoop(img, depth_img, darknets:BoundingBoxes):
    global currentTarget
    global currentFollow
    global waitStartedTime
    global lastTurn
    isTargetFound=False
    if(darknets!=None):
        detects = trackerCore.get_darknet_trackers(img, darknets)
        trk: tracker.Tracker
        for trk in detects:
            x_cv2 = trk.box
            print('ns cur box : ', trk.box, ', id : ', trk.id, ' score:', trk.score)
            # 사용 불가 박스는 넘김
            if (trk.isBoxValid() == False):
                print('ns box not valid ', trk.box)
                continue
            # 현재 타겟이 없을 경우 : 타겟 획득 행동
            if(isTargetFound==False):
                if (IdentifyTarget(trk, img)):
                    isTargetFound=True
                    RefreshTargetData(trk, img, depth_img)
                    ChangeModeToChasing()
                    currentFollow = trk.id
                    img = helpers.draw_box_label_Trac(trk, img, (0, 0, 255), True, currentTarget.latestDistance)
                    continue
            img = helpers.draw_box_label(trk.id, img, x_cv2)

    if(isTargetFound == False):
        standTurn(lastTurn)
        if(waitStartedTime - time.time() > 3):
            ChangeModeToFarSearching()
    return img

#메인 루프
#받는 패러미터 : 이미지, 뎁스이미지, 다크넷 바운딩박스
#반환값 : 표시할 이미지(사각형 그려놓은거)
def pipeline(img, depth_img, darknets:BoundingBoxes):
    global currentFollow
    global currentTarget
    global currentMode
    global currentTurn
    global lastTurn
    global waitStartedTime
    global isWorking
    
    #중복실행 방지
    if(isWorking == True):
        print('atomic blocked')
        UseLidarDataToSpin()
        return img

    isWorking=True
    if(currentTarget==None):
        img= preTargetLoop(img,depth_img,darknets)
    else:
        if(currentMode == Mode.Chasing):
            img= chasingLoop(img, depth_img,darknets)
        elif currentMode == Mode.FarSearching:
            img= farSearchingLoop(img, depth_img,darknets)
        elif currentMode == Mode.NearSearching:
            img = nearSearchingLoop(img, depth_img, darknets)

    isWorking=False
    return img


def ChangeModeToChasing():
    global currentMode
    print('current mode changed to chasing')
    currentMode = Mode.Chasing

def ChangeModeToFarSearching():
    global currentMode
    global waitStartedTime
    currentMode = Mode.FarSearching
    waitStartedTime=time.time()
    print('current mode changed to farsearching at ',waitStartedTime)

def ChangeModeToNearSearching():
    global currentMode
    global waitStartedTime
    global lastTurn
    global currentTarget
    currentMode = Mode.NearSearching
    waitStartedTime=time.time()
    lastTurn = currentTarget.lastDirection
    print('current mode changed to nearsearching at ',waitStartedTime)


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
    if(trk.score > detectBaseScore):
        return True

def RegisterTarget(trk: tracker.Tracker, img):
    global currentTarget

    try:
        currentTarget = target.Target()
        currentTarget.firstTracker = trk
        currentTarget.latestTracker = trk

        x_cv2 = trk.box
        left, top, right, bottom = x_cv2[1], x_cv2[0], x_cv2[3], x_cv2[2]
        currentTarget.firstImg = img[top:bottom, left:right]

        print('target registered : ', trk.id)
    except:
        print('register target error')
        raise

def newRawDrive():
    global move
    global currentTurn
    global lastObstacleDirection
    global lastObstacleScore
    #print('newRawDrive enter. width : ', W)

    trk = currentTarget.latestTracker
    distance =currentTarget.latestDistance

    half = W / 2
    center = trackerCore.GetCenterOfTracker(trk)
    posProto = center - half
    pos = posProto / half

    #print('current pos ', pos)
    if(abs(pos) <= 0.1):
        currentTurn = Direction.Center
    else:
        if(pos>0):
            currentTurn = Direction.Right
        else:
            currentTurn = Direction.Left
    print('rawdrive - currentTurn is ',currentTurn, 'current pos is ',pos)

    if (distance < 500):
        print("so close. stop")
        move.linear.x = 0
    else:
        move.linear.x=0.5
    TryBoost(distance)

    move.angular.z = pos * -0.5

    if(currentTurn == Direction.Center):
        move.angular.z = 0

    if(lastObstacleDirection == currentTurn):
        if(currentTurn == Direction.Right):
            move.angular.z += -1 * lastObstacleScore
        else:
            move.angular.z += lastObstacleScore
    if (distance < 500):
        print("so close. no turn")
        move.angular.z = 0
    pub.publish(move)
    return

def rawDrive(trk:tracker.Tracker, distance):
    global move
    print('rawDrive enter. width : ',W)

    half = W/2
    center = trackerCore.GetCenterOfTracker(trk)
    posProto =center - half
    pos = posProto / half

    print('pos ',pos)
    if pos != 0:
        move.linear.x = 0.5
    else:
        move.linear.x = 0
    move.angular.z = pos * -0.5

    if (distance < 500):
        print("so close. stop")
        move.linear.x = 0
        # move.angular.z = move.angular.z*0.05
    UseLidarDataToSpin()
    TryBoost(distance)
    pub.publish(move)

def TryBoost(distance):
    global move
    if(move.linear.x <=0):
        return
    difff = (distance - 500)/2000
    if(difff > 0.5):
        difff =0.5
    move.linear.x += difff


# 최종 라이다 데이터를 받아와서 현재의 이동 메시지에 추가로 회전값 부여 or 직접 회전
def UseLidarDataToSpin():
    global lastFrontLidarData
    global move
    if(lastFrontLidarData == None or time.time() - lastFrontLidarData.inserted > 1):
        return
    else:
        alert = lastFrontLidarData.GetObstacleScore()
        print('get obstacle data : ', alert.Direction,", ",alert.score)
        if(alert.Direction != Direction.Center):
            if(alert.Direction == Direction.Right):
                print('stand turn obs left start ',time.time())
                standTurn(Direction.Left)
                print('stand turn obs left finish', time.time())
                # move.angular.z += alert.score
                # if(move.linear.x >0):
                #     move.linear.x -=alert.score
                return
            else:
                print('stand turn obs right start ', time.time())
                standTurn(Direction.Right)
                print('stand turn obs right finish', time.time())
                # move.angular.z += -1 * alert.score
                # if (move.linear.x > 0):
                #     move.linear.x -= alert.score
                return
        else:
            if alert.score == 1:
                print('start avoid front obstacle')
                standTurn(Direction.Left)
                print('end')
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

def standTurn(direction : Direction):
    global move
    if (direction == Direction.Right):
        move.angular.z = -0.2
    elif direction == Direction.Left:
        move.angular.z = 0.2
    move.linear.x=0
    if(driveMode):
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
            move.angular.z = -0.1
        elif currentTurn == Direction.Left:
            move.angular.z = 0.1
        else:
            move.angular.z = 0

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

def f_lidar_callback(data):
    global lastFrontLidarData
    #print('lidar data plus')
    lastFrontLidarData= LidarData(data.ranges)

def darknet_callback(data):
    global lastYoloData
    global lastYoloAddedTime
    lastYoloData=data
    lastYoloAddedTime=time.time()
    #print('yolo data plus')

def detImage_callback(data):
    global lastImageData
    lastImageData=data
    #print('image data plus')

def depImage_callback(data):
    global lastDepthData
    lastDepthData=data
    #print('depth data plus')

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
        subLidarF = rospy.Subscriber("laser_f/scan", LaserScan, f_lidar_callback)
        darknetY=rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, darknet_callback)
        detImage=rospy.Subscriber('/camera/color/image_raw', Image, detImage_callback)
        depImage=rospy.Subscriber('/camera/depth/image_rect_raw', Image,depImage_callback)
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

            # msg=rospy.wait_for_message('/camera/color/image_raw',Image)
            # data=rospy.wait_for_message('/camera/depth/image_rect_raw', Image)
            # try:
            #     darknets= rospy.wait_for_message('/darknet_ros/bounding_boxes', BoundingBoxes, 1)
            # except:
            #     darknets=None
            #print(darknets)


            msg=lastImageData
            data =lastDepthData
            darknets=lastYoloData


            if msg == None or data == None:
                continue

            if lastYoloAddedTime != None:
                if(time.time() - lastYoloAddedTime < 0.5):
                    darknets = lastYoloData
                else:
                    darknets=None

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
            if(darknets == None):
                new_img = pipeline(img,cv_image,None)
            else:
                new_img = pipeline(img, cv_image, darknets.bounding_boxes)
            endpipeline=time.time()-start

            start = time.time()
            try:
                cv2.imshow('frame', new_img)
            except:
                print('exception show')
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
