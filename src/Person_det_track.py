#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""@author: ambakick
"""
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import glob
# import NotCvBridge as ncv2
# from moviepy.editor import VideoFileClip
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
frame_count = 0  # frame counter

debug = False

currentFollow = int

currentTarget: target.Target = None

trackerCore = tcore.PersonTrackerCore()

lastMin = float("inf")
minCount = 10

currentTurn: Direction = Direction.Center
lastTurn:Direction = Direction.Center

currentMode: Mode = Mode.Chasing

waitStartedTime: time = None


def pipeline(img):
    global currentFollow
    global currentTarget
    global currentMode
    global currentTurn
    global waitStartedTime

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
            if (trk.id == currentFollow and currentTarget != None):
                isIdLost = False
                RefreshTargetData(trk, img, img)
                lastTurn = currentTarget.lastDirection
                img = helpers.draw_box_label_Trac(trk, img, (0, 0, 255), True, currentTarget.latestDistance)

            else:
                img = helpers.draw_box_label(trk.id, img, x_cv2)

        # id 소실시 follow는 제거됨. 추적 모드로 들어가야 하니까
        if (isIdLost):
            currentFollow = -1
            # currentMode = Mode.NearSearching
            if(currentTarget!=None):
                ChangeModeToFarSearching()
                waitStartedTime = time.time()
        else:
            driveToTarget()

    elif currentMode == Mode.FarSearching:
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

    cv2.imshow("frame", img)


def ChangeModeToChasing():
    global currentMode
    print('current mode changed to chasing')
    currentMode = Mode.Chasing


def ChangeModeToFarSearching():
    global currentMode
    print('current mode changed to farsearching')
    currentMode = Mode.FarSearching


def RefreshTargetData(trk: tracker.Tracker, img, depth_img):
    global currentTarget
    global lastMin
    x_cv2 = trk.box
    left, top, right, bottom = x_cv2[1], x_cv2[0], x_cv2[3], x_cv2[2]
    currentTarget.lastImg = img[top:bottom, left:right]
    currentTarget.latestTracker = trk

    tDistance = tcore.get_biggest_distance_of_box(depth_img, trk)
    if (tDistance == 0):
        currentTarget.latestDistance = lastMin
    else:
        currentTarget.latestDistance = tDistance
        lastMin = tDistance

    currentTarget.lastDirection = trackerCore.GetDirectionOfTracker(trk)
    return


def IdentifyTarget(trk: tracker.Tracker, img):
    if (trk.score > 0.8):
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
    print('no running')


def driveToTarget():
    print('not running to target')


def DisposeTarget():
    global currentTarget
    print('remove current target')
    currentTarget = None


def PrintCenterDistance(x, y, depth_img):
    try:
        pix = (x / 2, y / 2)
        sys.stdout.write(
            'Depth at center(%d, %d): %f(mm)\r' % (pix[0], pix[1], depth_img[int(pix[1]), int(pix[0])]))
        sys.stdout.flush()
    except:
        print("failed PrintCenterDistance")


if __name__ == "__main__":
    currentFollow = -1
    driveMode = False
    det = detector.PersonDetector()

    if debug:  # test on a sequence of images
        images = [plt.imread(file) for file in glob.glob('./test_images/*.jpg')]

        for i in range(len(images))[0:7]:
            image = images[i]
            image_box = pipeline(image)
            plt.imshow(image_box)
            plt.show()

    else:
        print('main started')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        print('will start loop now')

        frame_rate = 10
        prev = 0
        while (True):
            time_elapsed = time.time() - prev

            if time_elapsed > 1. / frame_rate:
                prev = time.time()

            else:
                continue

            ret, img = cap.read()

            W = int(640)
            H = int(480)
            tcore.W = W
            tcore.H = H

            #PrintCenterDistance(W, H, img)

            new_img = pipeline(img)

            pressed = cv2.waitKey(1) & 0xFF

            if pressed == ord('q'):
                print('exit with q')
                break
            if pressed == ord('s'):
                driveMode = not driveMode
                print('drive mode change to ' + str(driveMode))
                if (driveMode == False):
                    DisposeTarget();
