import TopAdv as TA
import TopAdv_PBC as TAp
import numpy as np
import math


class TopologicalAdvection:

    def __init__(self, TrajectorySlices, Times, Domain, PeriodicBC = False):
        self.Tslices = TrajectorySlices
        self.Times = Times
        self.Domain = Domain
        #
