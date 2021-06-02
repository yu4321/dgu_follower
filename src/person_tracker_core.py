from typing import List

import numpy as np
import matplotlib.pyplot as plt
import glob

from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment
from darknet_ros_msgs.msg import BoundingBoxes

import helpers
import detector
import tracker
import cv2

from enum import Enum

class Direction(Enum):
    Left = 1
    Center = 0
    Right = 2

class Mode(Enum):
    Chasing = 1
    NearSearching = 2
    FarSearching = 3

class PersonTrackerCore:


    def __init__(self):
        self.W : int = 640
        self.H : int = 480

        # Global variables to be used by funcitons of VideoFileClop
        self.frame_count = 0  # frame counter

        self.max_age = 15  # no.of consecutive unmatched detection before
        # a track is deleted

        self.min_hits = 1  # no. of consecutive matches needed to establish a track

        self.tracker_list = []  # list for trackers
        # list for track ID
        #self.track_id_list = deque(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17'])

        self.track_id = 0
        self.debug = False

        self.det = detector.PersonDetector()

    def getTrackId(self):
        ret=self.track_id
        print('published id ',ret)
        self.track_id+=1
        return ret

    def GetDirectionOfTracker(self, trk:tracker.Tracker):
        x_cv2 = trk.box
        left, top, right, bottom = x_cv2[1], x_cv2[0], x_cv2[3], x_cv2[2]
        center = left + ((right - left) / 2)

        #w3=self.W/3
        if center < 150 :
            return Direction.Left
        elif center < 330 :
            return Direction.Center
        else:
            return Direction.Right

    def get_darknet_trackers(self, img, darknets : BoundingBoxes):
        good_tracker_list = []
        #print('get boxes ',len(darknets))
        for box in darknets:
            if(box.Class != "person"):
                continue
            n=tracker.Tracker()
            n.score=box.probability
            n.box = (box.ymin, box.xmin, box.ymax, box.xmax)
            #left, top, right, bottom = x_cv2[1], x_cv2[0], x_cv2[3], x_cv2[2]
            good_tracker_list.append(n)

        return good_tracker_list

    def get_good_trackers(self, img):
        good_tracker_list = []
        img_dim = (img.shape[1], img.shape[0])
        scoreBoxes = self.det.get_localization(img)  # measurement
        z_box = list(map(lambda x: x.box, scoreBoxes))
        if len(z_box) <= 0:
            good_tracker_list.clear()
            return good_tracker_list

        x_box = []

        if len(self.tracker_list) > 0:
            for trk in self.tracker_list:
                x_box.append(trk.box)
        matched, unmatched_dets, unmatched_trks \
            = assign_detections_to_trackers(x_box, z_box, iou_thrd=0.3)
        if matched.size > 0:
            for trk_idx, det_idx in matched:
                z = z_box[det_idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk = self.tracker_list[trk_idx]
                tmp_trk.kalman_filter(z)
                xx = tmp_trk.x_state.T[0].tolist()
                xx = [xx[0], xx[2], xx[4], xx[6]]
                x_box[trk_idx] = xx
                tmp_trk.box = xx
                tmp_trk.score = scoreBoxes[det_idx].score
                tmp_trk.hits += 1

        # Deal with unmatched detections
        if len(unmatched_dets) > 0:
            for idx in unmatched_dets:
                z = z_box[idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk = tracker.Tracker()  # Create a new tracker
                x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
                tmp_trk.x_state = x
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx = [xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box = xx
                tmp_trk.id = self.getTrackId() # assign an ID for the tracker
                tmp_trk.score=scoreBoxes[idx].score
                print(tmp_trk.id)
                self.tracker_list.append(tmp_trk)
                x_box.append(xx)

        # Deal with unmatched tracks
        if len(unmatched_trks) > 0:
            for trk_idx in unmatched_trks:
                tmp_trk = self.tracker_list[trk_idx]
                tmp_trk.no_losses += 1
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx = [xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box = xx
                x_box[trk_idx] = xx

        # The list of tracks to be annotated
        for trk in self.tracker_list:
            if ((trk.hits >= self.min_hits) and (trk.no_losses <= self.max_age)):
                good_tracker_list.append(trk)
        deleted_tracks = filter(lambda x: x.no_losses > self.max_age, self.tracker_list)

        #for trk in deleted_tracks:
        #    self.track_id_list.append(trk.id)

        self.tracker_list = [x for x in self.tracker_list if x.no_losses <= self.max_age]

        return good_tracker_list



def get_biggest_distance_of_box(image, depth_image, left, right, top, bottom) -> float:
    if image is None :
        print('image not true. break')
        return
    if depth_image is None:
        print('depth image not True, break')
        return
    cuttedD=depth_image[top:bottom, left:right]

    min = float("inf")
    try:
        x=np.array(cuttedD).flatten()
        mx=np.ma.masked_array(x, mask=x==0)
        min=mx.min()
    except:
        return 0

    if image is None:
        print('image2 not true. break')
        return 0
    if depth_image is None:
        print('depth image2 not True, break')
        return 0
    print('minimum distance : ',min,'mm')

    return min

def get_biggest_distance_of_box(depth_image, tracker:tracker.Tracker) -> float:
    x_cv2 = tracker.box
    left, top, right, bottom = x_cv2[1], x_cv2[0], x_cv2[3], x_cv2[2]
    if depth_image is None:
        print('depth image not True, break')
        return
    cuttedD = depth_image[top:bottom, left:right]

    min = float("inf")
    try:
        x = np.array(cuttedD).flatten()
        mx = np.ma.masked_array(x, mask=x == 0)
        min = mx.min()
    except:
        return 0

    if depth_image is None:
        print('depth image2 not True, break')
        return 0
    print('minimum distance : ', min, 'mm')

    return min


def assign_detections_to_trackers(trackers, detections, iou_thrd=0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''

    IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    for t, trk in enumerate(trackers):
        # trk = convert_to_cv2bbox(trk)
        for d, det in enumerate(detections):
            #   det = convert_to_cv2bbox(det)
            IOU_mat[t, d] = helpers.box_iou2(trk, det)

            # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)

    matched_idx = linear_assignment(-IOU_mat)

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if (t not in matched_idx[:, 0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if (d not in matched_idx[:, 1]):
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signifiy the existence of
    # an untracked object

    for m in matched_idx:
        if (IOU_mat[m[0], m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

