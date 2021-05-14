#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""@author: ambakick
"""
import sys
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


W = 640
H = 480

# Global variables to be used by funcitons of VideoFileClop
frame_count = 0 # frame counter

max_age = 15  # no.of consecutive unmatched detection before 
             # a track is deleted

min_hits =1  # no. of consecutive matches needed to establish a track

tracker_list =[] # list for trackers
# list for track ID
track_id_list= deque(['1', '2', '3', '4', '5', '6', '7', '7', '8', '9', '10'])

debug = False

currentFollow = int

ncv2=CvBridge()

def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''    
    
    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk) 
        for d,det in enumerate(detections):
         #   det = convert_to_cv2bbox(det)
            IOU_mat[t,d] = helpers.box_iou2(trk,det) 
    
    # Produces matches       
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)
    
    matched_idx = linear_assignment(-IOU_mat)        

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []
   
    # For creating trackers we consider any detection with an 
    # overlap less than iou_thrd to signifiy the existence of 
    # an untracked object
    
    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if (len(matches) == 0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)       
    
def get_biggest_distance_of_box(image, depth_image, left, right, top, bottom):
    cuttedD=depth_image[top:bottom, left:right]
    cuttedO=image[top:bottom, left:right]
    cv2.imshow("or",cuttedO)
    cv2.imshow("dp",cuttedD)



def pipeline(img, depth_img, distance):
    '''
    Pipeline function for detection and tracking
    '''
    global frame_count
    global tracker_list
    global max_age
    global min_hits
    global track_id_list
    global debug
    global currentFollow

    #print('distnace : '+str(distance))
    frame_count+=1

    #print('entered pipeline')
    
    img_dim = (img.shape[1], img.shape[0])
    z_box = det.get_localization(img) # measurement
    if len(z_box) <=0 :
        #print('z_box empty')
        currentFollow=-1
        cv2.imshow('frame',img)
        return img
    if debug:
       print('Frame:', frame_count)
       
    x_box =[]
    if debug: 
        for i in range(len(z_box)):
           img1= helpers.draw_box_label(img, z_box[i], box_color=(255, 0, 0))
           plt.imshow(img1)
        plt.show()
    
    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)
    
    
    matched, unmatched_dets, unmatched_trks \
    = assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3)  
    if debug:
         print('Detection: ', z_box)
         print('x_box: ', x_box)
         print('matched:', matched)
         print('unmatched_det:', unmatched_dets)
         print('unmatched_trks:', unmatched_trks)
    
         
    # Deal with matched detections     
    if matched.size >0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk= tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box =xx
            tmp_trk.hits += 1
    
    # Deal with unmatched detections      
    if len(unmatched_dets)>0:
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker.Tracker() # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
            print(tmp_trk.id)
            tracker_list.append(tmp_trk)
            x_box.append(xx)
    
    # Deal with unmatched tracks       
    if len(unmatched_trks)>0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box =xx
            x_box[trk_idx] = xx
                   
       
    # The list of tracks to be annotated  
    good_tracker_list =[]

    pos=float(0)
    for trk in tracker_list:
        if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
             good_tracker_list.append(trk)
             x_cv2 = trk.box
             if debug:
                 print('updated box: ', x_cv2)
                 print()
             img= helpers.draw_box_label(trk.id,img, x_cv2)

             if(pos==0):
                left, top, right, bottom = x_cv2[1], x_cv2[0], x_cv2[3], x_cv2[2]
                if(left-right <=0):
                    continue
                get_biggest_distance_of_box(img, depth_img, left, right, top, bottom)
                center = left+ ((right-left)/2)
                posProto=center-(W/2)
                pos = posProto/(W/2)

                print('center: ',center, "half width: ",W/2, ", posProto: ", posProto, ", pos: ", pos)
                #pos = center/(W/2)

                if driveMode == True:
                    if(currentFollow==-1):
                        currentFollow=trk.id
                        print('set target: id : '+str(trk.id))
                    else:
                        if currentFollow!=trk.id:
                            print('not target. pass id : '+str(trk.id))
                            pos=0
                else:
                    print('not drive Mode. set target none')
                    currentFollow=-1
                    pos=0
    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)
    
    for trk in deleted_tracks:
            track_id_list.append(trk.id)
    
    tracker_list = [x for x in tracker_list if x.no_losses<=max_age]
    
    if debug:
       print('Ending tracker_list: ',len(tracker_list))
       print('Ending good tracker_list: ',len(good_tracker_list))
    
    cv2.imshow("frame", img)
    print('target position - '+str(pos))
    dst=float(distance)
    drive(pos, dst)
    return img

def drive(pos, distance):
    global move
    if pos!=0:
        move.linear.x=0.5
    else:
        move.linear.x=0
    move.angular.z=pos * -0.5

    if(distance < 500):
        print("so close. stop")
        move.linear.x=0
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
        while(True):
            #msg = lastMsg
            msg=rospy.wait_for_message('/camera/color/image_raw',Image)
            data=rospy.wait_for_message('/camera/depth/image_rect_raw', Image)
            #print('get img encoding : '+str(msg.encoding))
            #np_arr=np.fromstring(msg.data,np.uint8)
            #cap = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') 
            #ret, img = cap.read()
            img=ncv2.imgmsg_to_cv2(msg)
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
            new_img = pipeline(img, cv_image, cv_image[int(pix[1]), int(pix[0])])
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
