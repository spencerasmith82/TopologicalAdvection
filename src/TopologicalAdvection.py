import TopAdv as TA
import TopAdv_PBC as TAp
import HelperFns as HF
import numpy as np
import math
from scipy.optimize import curve_fit

class TopologicalAdvection:

    def __init__(self, TrajectorySlices, Times, Domain = None, PeriodicBC = False):
        self.Tslices = TrajectorySlices
        self.Times = Times
        self.NumTimes = len(self.Times)
        self.Domain = Domain
        self.PeriodicBC = PeriodicBC
        #use the appropriate module based on periodic BC or regular
        if PeriodicBC:
            self.TA = TAp
        else:
            self.TA = TA
        if self.Domain is None:
            if PeriodicBC:
                print("Trajectories live on a doubly periodic domain, but no fundamental domain boundary was specifed. \n")
                print("Generating a fundamental domain based on max x and y values of the particle trajectories.")
            self.Domain = self.TA.HF.GetBoundingDomainTraj(self.Tslices, PeriodicBC = self.PeriodicBC)
        self.PrintParameters = self.TA.PrintParameters(Bounds = self.Domain)
        #now initialize a triangulation object
        self.Tri = self.TA.triangulation2D(self.Tslices[0], self.Domain)
        #and make a copy of it to do initial loop evaluations
        self.TriInit = self.Tri.TriCopy()
        self.TriEvolved = False
        self.TopologicalEntropy = None
        self.TotalWeightOverTime = None
        self.Loops = None
        self.LoopsEvolved = False
        
        

    def EvolveTri(self, Delaunay = False):
        for i in range(1, self.NumTimes):
            TA.HF.progressBar(i, self.NumTimes)
            self.Tri.Evolve(self.Tslices[i], Maintain_Delaunay = Delaunay)
        self.TriEvolved = True

    def ResetTri(self):
        self.Tri = self.TriInit.TriCopy()
        self.TriEvolved = False
    #make a function to calculate the topological entropy from the best fit

    def GetTopologicalEntropy(self):
        if not self.TriEvolved:
            self.EvolveTri()
        loopM = TA.Loop(self.TriInit, mesh = True)
        WeightsM = self.Tri.OperatorAction(loopM, num_times = self.NumTimes)
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


    def ClearCurves(self):
        self.Loops = []
        self.LoopsEvolved = False

    def AddCurve():

    
    
    

    #to change plotting parameters, before using this function, set the prameters in 
    #the PrintParameters attribute (a data object)
    def Plot(self, LoopIn = None, ):
        self.
        
    #can use HF.CounterToStr(countin) for plotting to movie files



            