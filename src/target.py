import tracker
import uuid
import time
from person_tracker_core import Direction


class Target():

    def __init__(self):
        self.firstImg = []
        self.lastImg=[]
        self.firstTracker: tracker.Tracker = None
        self.latestTracker: tracker.Tracker = None
        self.latestDistance: float = 0
        self.guid:uuid.uuid4()
        self.lastDirection = Direction.Center

class ObstacleAlert():

    def __init__(self, direction, score):
        self.Direction = direction
        self.score = score

class LidarData():

    def __init__(self, ranges:[]):
        thr = 0.60


        #print(ranges[345], ranges[330], ranges[315], ranges[300], ranges[285] ,ranges[270], ranges[0] ,ranges[15] ,ranges[30], ranges[45], ranges[60], ranges[75], ranges[90])
        #print('c0 = ',ranges[0])
        self.L15=(1 if ranges[345] < thr else 0)
        self.L30=1 if ranges[330] < thr else 0
        self.L45=1 if ranges[315] < thr else 0
        self.L60=1 if ranges[300] < thr else 0
        self.L75=1 if ranges[285] < thr else 0
        self.L90=1 if ranges[270] < thr else 0
        self.C0=1 if ranges[0] < thr else 0
        self.R15 = 1 if ranges[15] < thr else 0
        self.R30 = 1 if ranges[30] < thr else 0
        self.R45 = 1 if ranges[45] < thr else 0
        self.R60 = 1 if ranges[60] < thr else 0
        self.R75 = 1 if ranges[75] < thr else 0
        self.R90=1 if ranges[90] < thr else 0
        self.inserted = time.time()

    def GetObstacleScore(self):
        thr = 0.5
        if(self.C0 > thr):
            return ObstacleAlert(Direction.Center, 1)

        leftscore = self.L15*0.5 + self.L30 * 0.4 + self.L45 * 0.3 + self.L60 * 0.2 + self.L75 * 0.1
        rightscore = self.R15*0.5 + self.R30 * 0.4 + self.R45 * 0.3 + self.R60 * 0.2 + self.R75 * 0.1

        dir = Direction.Center
        score = 0
        if(leftscore == rightscore == 0):
            return ObstacleAlert(Direction.Center, 0)
        if(leftscore != rightscore and leftscore<rightscore):
            dir = Direction.Left
            score =1
        else:
            dir = Direction.Right
            score =1

        return ObstacleAlert(dir,score)
