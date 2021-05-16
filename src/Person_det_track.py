#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""@author: ambakick
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
#from moviepy.editor import VideoFileClip
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment

import helpers
import detector
import tracker
import cv2

import person_tracker_core as tcore

trackerCore=tcore.PersonTrackerCore()

W = 640
H = 480

# Global variables to be used by funcitons of VideoFileClop
frame_count = 0 # frame counter

debug = False

def pipeline(img):
    global debug
    global frame_count
    frame_count+=1

    if debug:
       print('Frame:', frame_count)

    detects = trackerCore.get_good_trackers(img)
    for trk in detects:
        x_cv2 = trk.box
        img = helpers.draw_box_label(trk.id, img, x_cv2)  # Draw the bounding boxes on the
    cv2.imshow("frame",img)
    return img
    
if __name__ == "__main__":    
    
    det = detector.PersonDetector()
    
    if debug: # test on a sequence of images
        images = [plt.imread(file) for file in glob.glob('./test_images/*.jpg')]
        
        for i in range(len(images))[0:7]:
             image = images[i]
             image_box = pipeline(image)   
             plt.imshow(image_box)
             plt.show()
           
    else: # test on a video file.
        
        # start=time.time()
        # output = 'test_v7.mp4'
        # clip1 = VideoFileClip("project_video.mp4")#.subclip(4,49) # The first 8 seconds doesn't have any cars...
        # clip = clip1.fl_image(pipeline)
        # clip.write_videofile(output, audio=False)
        # end  = time.time()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter('output.avi',fourcc, 8.0, (640,480))

        while(True):
            
            ret, img = cap.read()
            #print(img)
            
            np.asarray(img)
            new_img = pipeline(img)
            #out.write(new_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()
        print(round(end-start, 2), 'Seconds to finish')
