import TopAdv as TA
import TopAdv_PBC as TAp
import numpy as np
import math


class TopologicalAdvection:

    def __init__(self, TrajectorySlices, Times, Domain, PeriodicBC = False):
        self.Tslices = TrajectorySlices
        self.Times = Times
        self.Domain = Domain
        #use the appropriate module based on periodic BC or regular
        if PeriodicBC:
            self.TA = TAp
        else:
            self.TA = TA
        #now initialize a triangulation object
        self.tri = self.TA.triangulation2D(self.Tslices[0],)
        #and make a copy of it to do initial loop evaluations
        self.tri_init = self.tri.copy()
