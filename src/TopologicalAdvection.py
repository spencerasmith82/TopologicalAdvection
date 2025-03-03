import TopAdv as TA
import TopAdv_PBC as TAp
import TopAdvBase as TAbase
import HelperFns as HF
import numpy as np
import math
from scipy.optimize import curve_fit
from dataclasses import dataclass

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
            self.Domain = HF.GetBoundingDomainTraj(self.Tslices, PeriodicBC = self.PeriodicBC)
        self.PrintParameters = self.TA.PrintParameters(Bounds = self.Domain)
        #now initialize a triangulation object
        self.Tri = self.TA.triangulation2D(self.Tslices[0], self.Domain)
        #and make a copy of it to do initial loop evaluations
        self.TriInit = self.Tri.TriCopy()
        self.TriEvolved = False
        self.IsDelaunay = True
        self.TopologicalEntropy = None
        self.TotalWeightOverTime = None
        self.Loop = None
        self.CurveGenerator = CurveGenerator(self.Domain, self.PeriodicBC)
        
        

    def EvolveTri(self, Delaunay = False):
        for i in range(1, self.NumTimes):
            HF.progressBar(i, self.NumTimes)
            self.Tri.Evolve(self.Tslices[i], Maintain_Delaunay = Delaunay)
        self.TriEvolved = True
        self.IsDelaunay = Delaunay

    def ResetTri(self):
        self.Tri = self.TriInit.TriCopy()
        self.TriEvolved = False
    #make a function to calculate the topological entropy from the best fit

    def GetTopologicalEntropy(self, frac_start = 0.0):
        if not self.TriEvolved:
            self.EvolveTri()
        loopM = self.TA.Loop(self.TriInit, mesh = True)
        WeightsM = self.Tri.OperatorAction(loopM, num_times = self.NumTimes)
        LogWeightsM = [np.log(w) for w in WeightsM]
        iend = len(LogWeightsM)
        istart = int(iend*frac_start)
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


    def ClearLoop(self):
        self.Loop =  None

    def ClearCurves(self):
        self.CurveGenerator.ClearCurves()

    def LoadCurves(self):
        if len(self.CurveGenerator.Curves) > 0:
            loop = self.TA.Loop(self.TriInit, curves = self.CurveGenerator.Curves)
            self.Loop = LoopData(LoopInitial = loop, LoopFinal = loop.LoopCopy(), LoopEvolved = False)

    def EvolveLoop(self):
        if not self.TriEvolved:
            self.EvolveTri()
        if not self.Loop.LoopEvolved:
            self.Tri.OperatorAction(self.Loop.LoopFinal, option = 1)
            self.Loop.LoopEvolved = True

                
    def SetPrintParameters(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self.PrintParameters, key, value)
    

    #to change plotting parameters, before using this function, set the prameters in 
    #the PrintParameters attribute (a data object)
    def Plot(self, Loop = True, Initial = False):
        setattr(self.PrintParameters, "Delaunay", self.IsDelaunay)
        if not Loop:
            if Initial:
                self.TriInit.Plot(LoopIn = None, PP = self.PrintParameters)
            else:
                self.Tri.Plot(LoopIn = None, PP =self.PrintParameters)
        else:
            if self.Loop is not None:
                if Initial:
                    self.TriInit.Plot(LoopIn = self.Loop.LoopInitial , PP = self.PrintParameters)
                else:
                    self.EvolveLoop()  #does nothing if already evolved
                    self.Tri.Plot(LoopIn = self.Loop.LoopFinal, PP = self.PrintParameters)
            else:
                print("Need to create a loop")
            


    def MoviePlots(self, ImageFolder = "MovieImages/", ImageName = "EvolvingLoop", filetype = ".png"):
        setattr(self.PrintParameters, "Delaunay", True)
        if self.Loop is not None:
            self.ResetTri()
            loop = self.Loop.LoopInitial.LoopCopy()
            counter = 0
            #make sure that folder exists, and if not, make it
            fname = ImageFolder + ImageName + HF.CounterToStr(counter) + filetype
            setattr(self.PrintParameters, "filename", fname)
            self.Tri.Plot(LoopIn = loop, PP = self.PrintParameters)
            startind, stopind = 0, 0
            for i in range(1, self.NumTimes):
                startind = stopind+1
                HF.progressBar(i, self.NumTimes)
                self.Tri.Evolve(self.Tslices[i], Maintain_Delaunay = True)
                stopind = len(self.Tri.WeightOperatorList)-1
                self.Tri.OperatorAction(loop, index = [startind,stopind], option = 1)
                fname = ImageFolder + ImageName + HF.CounterToStr(i) + filetype
                setattr(self.PrintParameters, "filename", fname)
                self.Tri.Plot(LoopIn = loop, PP = self.PrintParameters)
            self.TriEvolved = True
            self.IsDelaunay = True
            setattr(self.PrintParameters, "filename", None)
        else:
            print("Need to create an initial loop first")
            
        
        
        
    #can use HF.CounterToStr(countin) for plotting to movie files



@dataclass
class LoopData:
    LoopInitial: TAbase.Loop = None
    LoopFinal: TAbase.Loop = None
    LoopEvolved: bool = False

class CurveGenerator:
    
    def __init__(self, Domain, PeriodicBC):
        self.Domain = Domain
        self.PeriodicBC = PeriodicBC
        self.NumPoints = 100
        self.Curves = []

    def ClearCurves(self):
        self.Curves = []

    def AddCircle(self, center, radius):
        self.AddEllipse(center, radius, radius)

    def AddEllipse(self, center, a, b, phi = 0):
        theta = np.linspace(0, 2*np.pi, num=self.NumPoints, endpoint=False)
        points = np.array([center[0] + a*np.cos(theta)*np.cos(phi) - b*np.sin(theta)*np.sin(phi) ,
                           center[1] + a*np.cos(theta)*np.sin(phi) + b*np.sin(theta)*np.cos(phi)]).T
        self.AddClosedCurve(points)

    def AddRectangle(self, center, w, h, phi = 0):
        points = np.array([[-w/2,-h/2], [w/2,-h/2], [w/2,h/2], [-w/2,h/2]])
        points = np.array([center[0] + points[:,0]*np.cos(phi) - points[:,1]*np.sin(phi) ,
                          center[1] + points[:,0]*np.sin(phi) + points[:,1]*np.cos(phi)])
        self.AddClosedCurve(points)

    def AddSquare(self, center, L, phi = 0):
        self.AddRectangle(center, L, L, phi)

    def AddVerticalLine(self, x_val):
        if x_val < self.Domain[0][0] or x_val > self.Domain[1][0]:
            print("Curve is not contained in the domain ", self.Domain)
            return []
        else:
            points = [[x_val, self.Domain[0][1]],[x_val, self.Domain[1][1]]]
            if self.PeriodicBC:
                self.Curves.append([points, False, [False, False], 1.0])
            else:
                self.Curves.append([points, False, [True, True], 0.5])

    def AddHorizontalLine(self, y_val):
        if y_val < self.Domain[0][1] or y_val > self.Domain[1][1]:
            print("Curve is not contained in the domain ", self.Domain)
            return []
        else:
            points = [[self.Domain[0][0], y_val ], [self.Domain[1][0], y_val]]
            if self.PeriodicBC:
                self.Curves.append([points, False, [False, False], 1.0])
            else:
                self.Curves.append([points, False, [True, True], 0.5])
        
    def AddLineSegment(self, pt1, pt2):
        points = [pt1,pt2]
        self.AddOpenCurve(points)

    def AddOpenCurve(self, points):
        if not self.ContainedInDomain(np.array(points)):
            print("Curve is not contained in the domain ", self.Domain)
        else:
            self.Curves.append([points, False, [True, True], 0.5])

    def AddClosedCurve(self, points):
        if not self.ContainedInDomain(np.array(points)):
            print("Curve is not contained in the domain ", self.Domain)
        else:
            self.Curves.append([points, True, [False, False], 1.0])  #point_set, is_closed, end_pts_pin, wadd
        
    
    def ContainedInDomain(self, points):
        x_max = np.max(points[:,0])
        x_min = np.min(points[:,0])
        y_max = np.max(points[:,1])
        y_min = np.min(points[:,1])
        if x_max > self.Domain[1][0] or x_min < self.Domain[0][0]:
            return False
        elif y_max > self.Domain[1][1] or y_min < self.Domain[0][1]:
            return False
        else:
            return True
        
        
            