import tracker
import uuid
import time
from person_tracker_core import Direction
import numpy as np


class Target():

    def __init__(self):
        self.firstImg = []
        self.lastImg=[]
        self.firstTracker: tracker.Tracker = None
        self.latestTracker: tracker.Tracker = None
        self.latestDistance: float = 0
        self.guid:uuid.uuid4()
        self.lastDirection = Direction.Center
        self.firstColors=[]

class ObstacleAlert():

    def __init__(self, direction, score):
        self.Direction = direction
        self.score = score

class LidarData():

    def __init__(self, ranges:[]):
        self.L15= self.GetMinimumNonZero(ranges[337:351]) 
        self.L30= self.GetMinimumNonZero(ranges[322:336]) 
        self.L45= self.GetMinimumNonZero(ranges[307:321]) 
        self.L60= self.GetMinimumNonZero(ranges[292:306])
        self.L75= self.GetMinimumNonZero(ranges[277:291])
        self.L90= self.GetMinimumNonZero(ranges[270:276])
        self.C0= min(self.GetMinimumNonZero(ranges[0:8]),self.GetMinimumNonZero(ranges[352:359])) 
        self.R15 =  self.GetMinimumNonZero(ranges[9:23]) 
        self.R30 =  self.GetMinimumNonZero(ranges[24:38]) 
        self.R45 =  self.GetMinimumNonZero(ranges[39:53]) 
        self.R60 =  self.GetMinimumNonZero(ranges[54:68])
        self.R75 =  self.GetMinimumNonZero(ranges[69:83])
        self.R90= self.GetMinimumNonZero(ranges[84:90])
        self.inserted = time.time()

    def GetMinimumNonZero(self,cuttedArray):
        x = np.array(cuttedArray).flatten()
        mx = np.ma.masked_array(x, mask=x == 0)
        min = mx.min()
        return min

    def GetObstacleScore(self):
        thr = 0.5
        front = min(self.C0, self. L15, self.L30, self.L45, self.R15, self.R30, self.R45,self.L60,self.R60)
        if(front < thr):
            leftmin = min(self.L45, self.L60, self.L75, self.L90)#self.L45 * 1 + self.L60 * 1.5 + self.L75 * 2 + self.L90 * 3
            rightmin= min(self.R45, self.R60, self.R75, self.R90) #self.R45 * 1 + self.R60 * 1.5 + self.R75 * 2 + self.R90 * 3

            print('GetOBS leftmin : ',leftmin, 'rightmin : ',rightmin)
            if(leftmin<thr):
                return ObstacleAlert(Direction.Left,1)
            else:
                return ObstacleAlert(Direction.Right,1)
            # print('leftmin : ',leftmin, ', rightmin : ',rightmin)
            # if(abs(leftmin-rightmin) < 0.1):
            # leftsum = sum([self.L45, self.L60, self.L75,self.L90])  # self.L45 * 1 + self.L60 * 1.5 + self.L75 * 2 + self.L90 * 3
            # rightsum = sum([self.R45, self.R60, self.R75,self.R90])  # self.R45 * 1 + self.R60 * 1.5 + self.R75 * 2 + self.R90 * 3
            #
            # print('leftsum : ', leftsum, ', rightsum : ', rightsum)
            # if (abs(leftsum - rightsum) < 0.1):
                return ObstacleAlert(Direction.Center,1)
            # else:
            #     if(leftsum<rightsum):
            #         return ObstacleAlert(Direction.Left,1)
            #     else:
            #         return ObstacleAlert(Direction.Right, 1)
        else:
            return ObstacleAlert(Direction.Center, 0)
            # return ObstacleAlert(Direction.Center,1)
        # if(self.C0 > thr):
        #     return ObstacleAlert(Direction.Center, 1)

        # leftscore = self.L15*0.5 + self.L30 * 0.4 + self.L45 * 0.3 + self.L60 * 0.2 + self.L75 * 0.1
        # rightscore = self.R15*0.5 + self.R30 * 0.4 + self.R45 * 0.3 + self.R60 * 0.2 + self.R75 * 0.1
        #
        # leftscore =  self.L45 * 0.3 + self.L60 * 0.2 + self.L75 * 0.1
        # rightscore = self.R45 * 0.3 + self.R60 * 0.2 + self.R75 * 0.1
        #
        # print('leftscore : ',leftscore, ', rightscore : ',rightscore)
        # dir = Direction.Center
        # score = 0
        # if(leftscore == rightscore == 0):
        #     return ObstacleAlert(Direction.Center, 0)
        # if(leftscore != rightscore and leftscore<rightscore):
        #     dir = Direction.Left
        #     score =rightscore
        # else:
        #     dir = Direction.Right
        #     score =leftscore
        #
        # return ObstacleAlert(dir,score)
