import tracker
import uuid
import time
from person_tracker_core import Direction
from collections import deque
import numpy as np


class Target():

    def __init__(self):
        self.firstImg = []
        self.lastImg = []
        self.lastImages = deque(maxlen=60)
        self.firstTracker: tracker.Tracker = None
        self.latestTracker: tracker.Tracker = None
        self.latestDistance: float = 0
        self.guid: uuid.uuid4()
        self.lastDirection = Direction.Center
        self.firstColors = []


class ObstacleAlert():

    def __init__(self, direction, score):
        self.Direction = direction
        self.score = score


class LidarData():

    def __init__(self, ranges: []):
        self.L15 = self.GetMinimumNonZero(ranges[337:351])
        self.L30 = self.GetMinimumNonZero(ranges[322:336])
        self.L45 = self.GetMinimumNonZero(ranges[307:321])
        self.L60 = self.GetMinimumNonZero(ranges[292:306])
        self.L75 = self.GetMinimumNonZero(ranges[277:291])
        self.L90 = self.GetMinimumNonZero(ranges[270:276])
        self.C0 = min(self.GetMinimumNonZero(ranges[0:8]), self.GetMinimumNonZero(ranges[352:359]))
        self.R15 = self.GetMinimumNonZero(ranges[9:23])
        self.R30 = self.GetMinimumNonZero(ranges[24:38])
        self.R45 = self.GetMinimumNonZero(ranges[39:53])
        self.R60 = self.GetMinimumNonZero(ranges[54:68])
        self.R75 = self.GetMinimumNonZero(ranges[69:83])
        self.R90 = self.GetMinimumNonZero(ranges[84:90])
        self.inserted = time.time()

    def GetMinimumNonZero(self, cuttedArray):
        x = np.array(cuttedArray).flatten()
        mx = np.ma.masked_array(x, mask=x == 0)
        min = mx.min()
        return min

    def GetObstacleScore(self):
        thr = 0.5
        front = min(self.C0, self.L15, self.L30, self.L45, self.R15, self.R30, self.R45, self.L60 * 1.5, self.R60 * 1.5)
        if (front < thr):
            leftmin = min(self.L45, self.L60, self.L75, self.L90)
            rightmin = min(self.R45, self.R60, self.R75, self.R90)

            if (leftmin < thr):
                return ObstacleAlert(Direction.Left, 1)
            else:
                return ObstacleAlert(Direction.Right, 1)
        else:
            return ObstacleAlert(Direction.Center, 0)
