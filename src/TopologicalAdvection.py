import TopAdv as TA
import TopAdv_PBC as TAp
import HelperFns as HF
import numpy as np
import math
from scipy.optimize import curve_fit

class TopologicalAdvection:

    def __init__(self, TrajectorySlices, Times, Domain, PeriodicBC = False):
        self.Tslices = TrajectorySlices
        self.Times = Times
        self.num_times = len(self.Times)
        self.Domain = Domain
        #use the appropriate module based on periodic BC or regular
        if PeriodicBC:
            self.TA = TAp
        else:
            self.TA = TA
        self.PrintParameters = self.TA.PrintParameters(Bounds = self.Domain)
        #now initialize a triangulation object
        self.tri = self.TA.triangulation2D(self.Tslices[0], self.Domain)
        #and make a copy of it to do initial loop evaluations
        self.tri_init = self.tri.TriCopy()
        self.evolved = False
        self.TopologicalEntropy = None
        self.TotalWeightOverTime = None
        

    def EvolveTri(self, Delaunay = False):
        for i in range(1, self.num_times):
            TA.HF.progressBar(i, self.num_times)
            self.tri.Evolve(self.Tslices[i], Maintain_Delaunay = Delaunay)
        self.evolved = True

    #make a function to calculate the topological entropy from the best fit

    def GetTopologicalEntropy(self):
        if not self.evolved:
            self.EvolveTri()
        loopM = TA.Loop(self.tri_init, mesh = True)
        WeightsM = self.tri.OperatorAction(loopM, num_times = self.num_times)
        LogWeightsM = [np.log(w) for w in WeightsM]
        iend = len(LogWeightsM)
        #istart = int(iend/5)
        istart = 0
        TE, TE_err = self.GetSlopeFit(LogWeightsM,istart,iend)
        self.TopologicalEntropy = [TE, TE_err]
        self.TotalWeightOverTime = WeightsM
        return TE, TE_err, WeightsM

    def GetSlopeFit(self, LWeightsIn, istart, iend):
        def linear_func(x, a, b):
            return a*x+b
        #fitting to a linear function ax+b
        popt, pcov = curve_fit(linear_func, self.Times[istart:iend], LWeightsIn[istart:iend])
        perr = np.sqrt(np.diag(pcov))
        return [popt[0],perr[0]]


    
    #can use HF.CounterToStr(countin) for plotting to movie files



            