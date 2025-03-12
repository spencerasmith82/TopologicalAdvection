"""Main module for using the topological advection algorithm.

This top level module provides classes and methods to do the most common
actions with the topological advection algorithm (Evolve the triangulation,
calculate the topological entropy, initialize a curve, evolve the curve, plot
the triangulation and curve, create images for a movie).  This module accesses
the appropriate child module of top_advec_base.py (top_advec_bnd.py or
top_advec_pbc.py) based on the specified use case (trajectories are bounded
on a plane or are on a doubly periodic domain/torus).

The topological advection algorithm takes trajectories of point particles in
2D and determines how this motion affects the state of material curves in the
surrounding medium. Curves are encoded topologically as 'loops' with a
triangulation of the points acting as a basis for the loops. As the points
move, the triangulation is updated, and operators which act on loops are
accumulated.

Classes
-------
TopologicalAdvection
    Main class. Trajectories & times as attributes and main actions as methods

CurveGenerator
    Built-in options for generating geometric curves (which are used to
    initialize topological loops in TopologicalAdvection objects)
"""

import top_advec_bnd
import top_advec_pbc
import helper_fns as HF
import numpy as np
import os
from scipy.optimize import curve_fit
from dataclasses import asdict


class TopologicalAdvection:
    """Class realizing the topological advection algorithm.

    The topological advection algorithm takes trajectories of point particles
    in 2D and determines how this motion affects the state of material curves
    in the surrounding medium. Curves are encoded topologically as 'loops'
    with a triangulation of the points acting as a basis for the loops. As the
    points move, the triangulation is updated, and operators which act on
    loops are accumulated.

    This top level class has attributes and methods to do the most common
    actions with the topological advection algorithm (Evolve the triangulation,
    calculate the topological entropy, initialize a curve, evolve the curve,
    plot the triangulation and curve, create images for a movie).


    Attributes
    ----------
    Tslices : list of lists
        Tslices is short for time-slices (of the trajectory data), and
        represents the set of point trajectories over time.  The first index
        indexes the time slice, and len(Tslices) = len(Times).  The second
        index indexes the particle id (particle ids are implicit, and a given
        particle will always have its data in the same position in any time
        slice).  So len(Tslices[t]) = number of particles (for any
        t < len(Times)).  The third index indexes the x/y direction choice.
        So, Tslices[t][p][0] gives the x position of the pth particle at time
        slice t, while Tslices[t][p][1] gives the y position.

    Times : list of floats
        The times at which the time slices were taken.  These are needed to
        give quantitative meaning to the Topological Entropy (per unit time).
        Must have len(Times) = len(Tslices). If no Times list is passed on
        initialization, a simple list with equal increments of 1 is generated.

    Domain : list of lists
        This is the rectangular bounding domain for the trajectories.  It has
        the format [[x_min, y_min], [x_max, y_max]].  In the case of periodic
        boundary conditions (PeriodicBC is True), x_min and y_min must be 0. If
        no domain is passed on initialization, one will be generated based on
        the max and min x/y positions of the particles (not ideal for periodic
        boundary case).

    PeriodicBC : bool
        Does the data live on a doubly periodic domain/torus (PeriodicBC is
        True), or does it live on a bounded piece of the plane (PeriodicBC is
        False - default)?  This flag determines which module is used:
        (top_advec_bnd.py or top_advec_pbc.py).

    TA : module name
        This references the appropriate module (top_advec_bnd or top_advec_pbc)
        based on the flag PeriodicBC.  Both modules have identically named
        classes, class methods, and class attributes (enforced by the shared
        parent abstract base class), so, for example, self.TA.triangulation2D()
        will create a triangulation in both cases.

    PlotParameters : PlotParameters object
        This data object contains all the parameters that are needed to specify
        a triangulation/loop plot.  See PlotParameters docs for a complete list
        of the options/parameters.  Use method SetPlotParameters() to a set
        the paramters, ResetPlotParametersDefault() to reset them to default,
        and PrintPlotParameters() to print out the current parameter values.

    Tri : triangulation2D object
        This object keeps track of the triangulation of the points. The methods
        and attributes of triangulation2D are accessed through this object.
        Most of the topological advection algorithm is done behind the scenes
        using this object.

    TriInit : triangulation2D object
        A copy of Tri, which is kept in the initial state (i.e. not evolved
        forward like Tri).  This is used as the basis triangulation for
        initializing topological loops.

    TriEvolved : bool
        A flag that keeps track of the state of Tri. If True, then Tri has been
        evolved forward, accumulated WeightOperators, and is a triangulation
        of the points in the final time slice.

    IsDelaunay : bool
        A flag that keeps track of whether the triangulation Tri is currently
        Delaunay or not.  This is used to inform plotting options.

    TopologicalEntropy : list of 2 floats
        This stores the topological entropy of the trajectory set and the
        error in fitting the log weights to a straight line. [TopEnt, Err]
        This is None until method GetTopologicalEntropy is run.

    TotalWeightOverTime : list of ints
        This records the total weight of a loop initialized in the mesh
        configuration at each time slice.  This data is used to get the
        topological entropy, and is saved if this level of specificity is
        needed.

    LoopData : LoopData object
        This object consists of three attributes: LoopInitial, LoopFinal, and
        LoopEvolved.  The first two are Loop objects from the module TA, which
        represent a topological loop with TriInit as a basis, and a copy of
        this loop that will be evolved forward.  LoopEvolved is a bool which
        indicates whether LoopFinal has been evolved forward yet.

    CurveGenerator : CurveGenerator object
        This object stores representations of geometric curves and has methods
        for generating them.  These geometric curves can then be used to
        initialize a topological loop.

    Notes
    -----
    For periodic boundary conditions (PeriodicBC is True), all particle
    positions must have non-negative x and y values.  That is, the lower left
    corner of the fundamental domain is the origin by convention.  You can
    always shift your data to fit this convention.

    Methods
    -------
    EvolveTri(Delaunay=False)
        Evolve Tri forward to the final time

    GetTopologicalEntropy(frac_start=0.0, ss_indices=None)
        Find the topological entropy of the trajectory set

    """

    def __init__(self, TrajectorySlices, Times = None, Domain=None, PeriodicBC=False):
        self.Tslices = TrajectorySlices
        if Times is None:
            self.Times = [i for i in range(len(self.Tslices))]
        else:
            self.Times = Times
        self._NumTimes = len(self.Times)
        self.Domain = Domain
        self.PeriodicBC = PeriodicBC
        #  use the appropriate module based on periodic BC or regular
        if PeriodicBC:
            self.TA = top_advec_pbc
        else:
            self.TA = top_advec_bnd
        if self.Domain is None:
            if PeriodicBC:
                print("Trajectories live on a doubly periodic domain,"
                      " but no fundamental domain boundary was specifed. \n")
                print("Generating a fundamental domain based on max x and"
                      " y values of the particle trajectories.")
            self.Domain = HF.GetBoundingDomainTraj(self.Tslices,
                                                   PeriodicBC=self.PeriodicBC)
        self.PlotParameters = self.TA.PlotParameters(Bounds=self.Domain)
        #  now initialize a triangulation object
        self.Tri = self.TA.Triangulation2D(self.Tslices[0], self.Domain)
        if not PeriodicBC:
            ExBnd = HF.GetBoundingDomainSlice(self.Tri.pointpos, frac=0.0)
            dx = (ExBnd[1][0] - ExBnd[0][0])/np.sqrt(len(self.Tri.pointpos))
            dy = (ExBnd[1][1] - ExBnd[0][1])/np.sqrt(len(self.Tri.pointpos))
            dz = min(dx, dy)
            ExBnd[0][0] -= dz
            ExBnd[1][0] += dz
            ExBnd[0][1] -= dz
            ExBnd[1][1] += dz
            setattr(self.PlotParameters, "ExpandedBounds", ExBnd)
        #  and make a copy of it to do initial loop evaluations
        self.TriInit = self.Tri.TriCopy()
        self.TriEvolved = False
        self.IsDelaunay = True
        self.TopologicalEntropy = None
        self.TotalWeightOverTime = None
        self.LoopData = None
        self.CurveGenerator = CurveGenerator(self.Domain, self.PeriodicBC)

    def EvolveTri(self, Delaunay=False):
        """Evolve Tri forward to the final time slice.

        This takes Tri (resetting it to be a copy of TriInit if already
        evolved), and evolves it forward using the time slices to the final
        time.

        Parameters
        ----------
        Delaunay : bool
            If True then extra triangulation flips will be used to force the
            triangulation to be Delaunay after each time step.
            The default is False.

        Returns
        -------
        None.
        """
        if self.TriEvolved:
            self._ResetTri()
        for i in range(1, self._NumTimes):
            HF.progressBar(i, self._NumTimes)
            self.Tri.Evolve(self.Tslices[i], Maintain_Delaunay=Delaunay)
        self.TriEvolved = True
        self.IsDelaunay = Delaunay

    def _ResetTri(self):
        self.Tri = self.TriInit.TriCopy()
        self.TriEvolved = False

    def GetTopologicalEntropy(self, frac_start=0.0, ss_indices=None):
        """Find the topological entropy of the trajectory set.

        This evolves Tri forward to the final time slice (if not already
        evolved), creates a mesh initialized loop, operates on this loop with
        the accumulated weight operators and gets the total weights at each
        time step, then finds the best fit of log weights vs. time to a line,
        the slope of which is the topological entropy.

        Parameters
        ----------
        frac_start : float
            The fraction of the total time to start the fitting at (must be
            between 0 and 1). The weights can sometimes have initial transitory
            behavior before exponentially increasing, and we can change
            frac_start to exclude this from the fit. The default is 0.0.

        ss_indices : list of 2 int
            The start and stop indices for the fitting. The default is None. If
            provied ss_indices over-ride the start and stop indices calculated
            with frac_start.

        Returns
        -------
        TE : float
            The topological entropy estimate
        TE_err : float
            The error from the linear best fit
        WeightsM : list of ints
            The list of total weight at each time slice
        """
        if not self.TriEvolved:
            self.EvolveTri()
        loopM = self.TA.Loop(self.TriInit, mesh=True)
        WeightsM = self.Tri.OperatorAction(loopM, num_times=self._NumTimes)
        LogWeightsM = [np.log(w) for w in WeightsM]
        iend = len(LogWeightsM)
        istart = int(iend*frac_start)
        if ss_indices is not None:
            istart, iend = ss_indices
        TE, TE_err = self._GetSlopeFit(LogWeightsM, istart, iend)
        self.TopologicalEntropy = [TE, TE_err]
        self.TotalWeightOverTime = WeightsM
        return TE, TE_err, WeightsM

    def _GetSlopeFit(self, LWeightsIn, istart, iend):
        def linear_func(x, a, b):
            return a*x+b
        #  fitting to a linear function ax+b
        popt, pcov = curve_fit(linear_func, self.Times[istart:iend],
                               LWeightsIn[istart:iend])
        perr = np.sqrt(np.diag(pcov))
        return [popt[0], perr[0]]

    def ClearLoop(self):
        self.LoopData = None

    def ClearCurves(self):
        self.CurveGenerator.ClearCurves()

    def InitializeLoopWithCurves(self):
        if len(self.CurveGenerator.Curves) > 0:
            loop = self.TA.Loop(self.TriInit,
                                curves=self.CurveGenerator.Curves)
            self.LoopData = _LoopData(topadvec_in=self, LoopInitial=loop)

    def EvolveLoop(self):
        if not self.TriEvolved:
            self.EvolveTri()
        if not self.LoopData.LoopEvolved:
            self.Tri.OperatorAction(self.LoopData.LoopFinal, option=1)
            self.LoopData.LoopEvolved = True

    def SetPlotParameters(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self.PlotParameters, key, value)

    def ResetPlotParametersDefault(self):
        self.PlotParameters = self.TA.PlotParameters(Bounds=self.Domain)

    def PrintPlotParameters(self):
        for key, value in asdict(self.PlotParameters).items():
            if not (key == "conversion_factor" or key == "max_weight"):
                print(f"{key}: {value}")

    #  to change plotting parameters, before using this function, set the
    #  prameters in the PlotParameters attribute (a data object)
    def Plot(self, PlotLoop=True, Initial=False):
        setattr(self.PlotParameters, "Delaunay", self.IsDelaunay)
        if not PlotLoop:
            if Initial:
                self.TriInit.Plot(LoopIn=None, PP=self.PlotParameters)
            else:
                self.Tri.Plot(LoopIn=None, PP=self.PlotParameters)
        else:
            if self.LoopData is not None:
                if not self.LoopData.LoopInitial.Shear:
                    if Initial:
                        self.TriInit.Plot(LoopIn=self.LoopData.LoopInitial,
                                          PP=self.PlotParameters)
                    else:
                        self.EvolveLoop()  # does nothing if already evolved
                        self.Tri.Plot(LoopIn=self.LoopData.LoopFinal,
                                      PP=self.PlotParameters)
                else:
                    print("Currently don't support plotting loops"
                          " represented with shear coordinates")
            else:
                print("Need to create a loop")

    def MovieFigures(self, PlotLoop=True, Delaunay=True,
                     ImageFolder="MovieImages/", ImageName="EvolvingLoop",
                     filetype=".png"):
        setattr(self.PlotParameters, "Delaunay", Delaunay)
        if self.LoopData is not None and PlotLoop:
            self._ResetTri()
            loop = self.LoopData.LoopInitial.LoopCopy()
            if not os.path.exists(ImageFolder):
                os.makedirs(ImageFolder)
            fname = ImageFolder + ImageName + HF.CounterToStr(0) + filetype
            setattr(self.PlotParameters, "filename", fname)
            self.Tri.Plot(LoopIn=loop, PP=self.PlotParameters)
            startind, stopind = 0, 0
            for i in range(1, self._NumTimes):
                startind = stopind+1
                HF.progressBar(i, self._NumTimes)
                self.Tri.Evolve(self.Tslices[i], Maintain_Delaunay=Delaunay)
                stopind = len(self.Tri.WeightOperatorList)-1
                self.Tri.OperatorAction(loop, index=[startind, stopind],
                                        option=1)
                fname = ImageFolder + ImageName + HF.CounterToStr(i) + filetype
                setattr(self.PlotParameters, "filename", fname)
                self.Tri.Plot(LoopIn=loop, PP=self.PlotParameters)
            self.TriEvolved = True
            self.IsDelaunay = Delaunay
            setattr(self.PlotParameters, "filename", None)
        elif self.LoopData is None and PlotLoop:
            print("Need to create an initial loop first")
        else:
            self._ResetTri()
            if not os.path.exists(ImageFolder):
                os.makedirs(ImageFolder)
            fname = ImageFolder + ImageName + HF.CounterToStr(0) + filetype
            setattr(self.PlotParameters, "filename", fname)
            self.Tri.Plot(LoopIn=None, PP=self.PlotParameters)
            for i in range(1, self._NumTimes):
                HF.progressBar(i, self._NumTimes)
                self.Tri.Evolve(self.Tslices[i], Maintain_Delaunay=Delaunay)
                fname = ImageFolder + ImageName + HF.CounterToStr(i) + filetype
                setattr(self.PlotParameters, "filename", fname)
                self.Tri.Plot(LoopIn=None, PP=self.PlotParameters)
            self.TriEvolved = True
            self.IsDelaunay = Delaunay
            setattr(self.PlotParameters, "filename", None)


class _LoopData:

    def __init__(self, topadvec_in, LoopInitial):
        self.LoopInitial: topadvec_in.TA.Loop = LoopInitial
        self.LoopFinal: topadvec_in.TA.Loop = LoopInitial.LoopCopy()
        self.LoopEvolved: bool = False

class CurveGenerator:

    def __init__(self, Domain, PeriodicBC):
        self.Domain = Domain
        self.PeriodicBC = PeriodicBC
        self.Curves = []

    def ClearCurves(self):
        self.Curves = []

    def AddCircle(self, center, radius, NumPoints=100):
        self.AddEllipse(center, radius, radius, NumPoints=NumPoints)

    def AddEllipse(self, center, a, b, phi=0, NumPoints=100):
        theta = np.linspace(0, 2*np.pi, num=NumPoints, endpoint=False)
        points = np.array([center[0] + a*np.cos(theta)*np.cos(phi)
                           - b*np.sin(theta)*np.sin(phi),
                           center[1] + a*np.cos(theta)*np.sin(phi)
                           + b*np.sin(theta)*np.cos(phi)]).T
        self.AddClosedCurve(points)

    def AddRectangle(self, center, w, h, phi=0):
        points = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
        points = np.array([center[0] + points[:, 0]*np.cos(phi)
                           - points[:, 1]*np.sin(phi),
                          center[1] + points[:, 0]*np.sin(phi)
                          + points[:, 1]*np.cos(phi)]).T
        self.AddClosedCurve(points)

    def AddSquare(self, center, L, phi=0):
        self.AddRectangle(center, L, L, phi)

    def AddVerticalLine(self, x_val):
        if x_val < self.Domain[0][0] or x_val > self.Domain[1][0]:
            print("Curve is not contained in the domain ", self.Domain)
            return []
        else:
            delta = 1e-6*(self.Domain[1][1] - self.Domain[0][1])
            points = [[x_val, self.Domain[0][1] + delta],
                      [x_val, self.Domain[1][1] - delta]]
            if self.PeriodicBC:
                self.Curves.append([points, False, [False, False], 1.0])
            else:
                self.Curves.append([points, False, [True, True], 0.5])

    def AddHorizontalLine(self, y_val):
        if y_val < self.Domain[0][1] or y_val > self.Domain[1][1]:
            print("Curve is not contained in the domain ", self.Domain)
            return []
        else:
            delta = 1e-6*(self.Domain[1][0] - self.Domain[0][0])
            points = [[self.Domain[0][0] + delta, y_val],
                      [self.Domain[1][0] - delta, y_val]]
            if self.PeriodicBC:
                self.Curves.append([points, False, [False, False], 1.0])
            else:
                self.Curves.append([points, False, [True, True], 0.5])

    def AddLineSegment(self, pt1, pt2):
        points = [pt1, pt2]
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
            self.Curves.append([points, True, [False, False], 1.0])
            #  point_set, is_closed, end_pts_pin, wadd

    def ContainedInDomain(self, points):
        x_max = np.max(points[:, 0])
        x_min = np.min(points[:, 0])
        y_max = np.max(points[:, 1])
        y_min = np.min(points[:, 1])
        if x_max > self.Domain[1][0] or x_min < self.Domain[0][0]:
            return False
        elif y_max > self.Domain[1][1] or y_min < self.Domain[0][1]:
            return False
        else:
            return True
