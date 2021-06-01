import tracker
import uuid
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
