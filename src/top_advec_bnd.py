"""Module for topological advection with boundary.

This module contains child classes (derived from top_advec_base.py) that are
tailored to the case where particles remain in a bounded region
of the plane. The topological advection algorithm takes trajectories of point
particles in 2D and determines how this motion affects the state of material
curves in the surrounding medium.  Curves are encoded topologically as 'loops'
with a triangulation of the points acting as a basis for the loops.  As the
points move, the triangulation is updated, and operators which act on loops
are accumulated.

Classes
-------
simplex2D
    Class representing a triangle / 2D simplex

Loop
    Class representing a topological loop or set of loops.

WeightOperator
    Class representing an operator that acts on loops.

PlotParameters:
    Data class for grouping plot parameters

triangulation2D
    Class representing a triangulation of data points in a 2D domain.
    With methods for evolving the triangulation forward due to moving points,
    intializing loops, evolving loops, and plotting.
"""

import math
import copy
from operator import itemgetter
from dataclasses import dataclass
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib import rcParams
from top_advec_base import (simplex2D_Base, Loop_Base, WeightOperator_Base,
                            triangulation2D_Base, PlotParameters)
import helper_fns as HF


# simplex2D class ############################################################
class simplex2D(simplex2D_Base):
    """Class representing a triangle / 2D simplex.

        (bounded domain version)

        (used in a 2D triangulation object)

    Attributes
    ----------
    points : list of 3 ints
        List of the 3 vertex point IDs. These IDs will be used to get the
        vertex spatial locations from a master list that is a triangulation2D
        attribue. Note that the order is only unique up to an even
        permutation. The permuation given on initialization is assumed to
        correspond to a set of geometric points that are given in counter
        clock-wise, CCW, order. Moving about this list (and other simplex2D
        attributes) is done via modular arithmatic: (i+1)%3 to move CCW about
        the simplex from index i, and (i+2)%3 to move CW.

    simplices : list of 3 simplex objects
        List of the 3 simplices adjacent to this simplex. Convention: simplex
        at position i in this list is adjacent to this simplex at the edge
        across from self.points[i]

    edgeids : list of 3 ints
        List of the 3 edge ids. Each edge has an identifying integer that is
        used when keeping track of loop coordinates (in loop class).

    SLindex : int
        ID of this simplex in a Simplex List attribute of triangulation2D
        (simplist). This is useful for going back and forth from the local
        view (simplex2D object) and the global view (triangulation2D object)

    IsBoundarySimp: bool
        IsBoundarySimp marks a simplex as being part of the boundary (True)
        if all point are boundary points.  Default is False.  Set when
        generating the triangulation

    Methods
    -------
    LocalID(IDin)
        Given the ID (IDin) of a vertex/point in this triangle, this returns
        the local ID, i.e. the integer i = 0,1,2 s.t. self.points[i] = IDin

    SimpNeighbors(IDin)
        Find the simplices about a given point. In the case of a boundary
        simplex (with None as one/two of the simplex neighbors), the list
        is not necessarily in CCW order.

    EdgeNeighbors(IDin)
        Find the ids of edges about a given point

    SimpLink(S_other)
        Link self with S_other simplex
    """

    def __init__(self, IDlist):
        """Initialize the simplex.

        Parameters
        ----------
        IDlist : list of 3 ints
            List of the 3 point IDs from the master list (part of the
            tranguation class). It is assumed that IDlist already refers to
            points in the proper permutation (list order == ccw geometric
            order). Sets the points attribute.

        Notes
        -----
        Other attributes are set when linking them up in a triangulation2D
        object.
        """
        super().__init__(IDlist)
        self.IsBoundarySimp = False

    def SimpNeighbors(self, IDin):
        """Find the set of simpices which share a point.

        Parameters
        ----------
        IDin : int
            The ID of a vertex/point in this simplex.

        Returns
        -------
        list of simplex2D objects
            the simplices (in CCW cyclical order about the shared point -
            IDin) adjacent to a point (IDin).  In the case of a boundary
            simplex (with None as one/two of the simplex neighbors), the list
            is not necessarily in CCW order.

        Note that this method requires that the simplex is connected up with
        other simplices in a triangulation.
        """
        NeighborList = []
        LocalIDList = []
        stpt = self.LocalID(IDin)
        NeighborList.append(self)
        LocalIDList.append(stpt)
        lsimp = self.simplices[(stpt+1) % 3]
        while (self is not lsimp) and (lsimp is not None):
            NeighborList.append(lsimp)
            lsimp_lid = lsimp.LocalID(IDin)
            LocalIDList.append(lsimp_lid)
            lsimp = lsimp.simplices[(lsimp_lid+1) % 3]
        if lsimp is None:  # this deals with the boundary simplex case
            rsimp = self.simplices[(stpt+2) % 3]
            while (self is not rsimp) and (rsimp is not None):
                NeighborList.append(rsimp)
                rsimp_lid = rsimp.LocalID(IDin)
                LocalIDList.append(rsimp_lid)
                rsimp = rsimp.simplices[(rsimp_lid+2) % 3]
        return NeighborList, LocalIDList

    def EdgeNeighbors(self, IDin):
        """Find the edges which share a point.

        Parameters
        ----------
        IDin : int
            The ID of a vertex/point in this simplex.

        Returns
        -------
        list of ints
            the edge ids (in CCW cyclical order about the shared point -
            IDin) adjacent to a point (IDin).  In the case of a boundary
            simplex (with None as one/two of the simplex neighbors), the list
            is not necessarily in CCW order.

        Note that this method requires that the simplex is connected up with
        other simplices in a triangulation.
        """
        EdgeList = []
        stpt = self.LocalID(IDin)
        lsimp = self.simplices[(stpt+1) % 3]
        EdgeList.append(self.edgeids[(stpt+1) % 3])
        while (self is not lsimp) and (lsimp is not None):
            lsimp_lid = lsimp.LocalID(IDin)
            EdgeList.append(lsimp.edgeids[(lsimp_lid+1) % 3])
            lsimp = lsimp.simplices[(lsimp_lid+1) % 3]
        if lsimp is None:  # this deals with the boundary simplex case
            rsimp = self.simplices[(stpt+2) % 3]
            while (self is not rsimp) and (rsimp is not None):
                rsimp_lid = rsimp.LocalID(IDin)
                EdgeList.append(rsimp.edgeids[(rsimp_lid+2) % 3])
                rsimp = rsimp.simplices[(rsimp_lid+2) % 3]
        return EdgeList

    def SimpLink(self, S_other):
        """Links this simplex with S_other (and vice versa).

        Used during an edge flip operation to ensure the newly created
        simplices are integrated into the triangulation

        Parameters
        ----------
        S_other : Simplex2D object
            The simplex to link with self
        """
        if S_other is not None:
            # Note that we don't need to do anything for linking a new
            # simplex to None ... it starts out that way
            locid_s1 = 0
            for i in range(3):
                if not self.points[i] in S_other.points:
                    self.simplices[i] = S_other
                    locid_s1 = i
                    break
            locid_s2 = (S_other.points.index(
                self.points[(locid_s1 + 1) % 3]) + 1) % 3
            S_other.simplices[locid_s2] = self
# End of simplex2D class #####################################################


# Loop Class #################################################################
class Loop(Loop_Base):
    """Class representing a topological loop or set of loops.

    The coordinate system (basis) for this representation is fixed by a
    particular triangulation2D object.

    Attributes
    ----------
    weightlist : list of ints/real numbers
        List of intersection coordinates for the loop.  Each location in the
        list corresponds to an edge in the triangulation2D object (list index
        is edge id), and the list value at this index is the number of
        transverse intersections of our geometric loop (pulled 'tight' for
        minimal intersection number) with the associated triangulation edge.
        A list of shear coordinates is also possible.

    Shear : bool
        Flag to denote the type of representation: False for regular
        intersection coordinates, and True for shear coordinates.  Shear
        coordinates allow us to create a 'mesh' of bands along each edge,
        which is helpful for topological entropy calculations

    Methods
    -------
    GetWeightTotal()
        Returns the sum of the intersection coordinates (weights), which is a
        good proxy for the length of the loop.

    ProjectivizeWeights()
        Divides all the weights by the max weight value.
    """

    def __init__(self, tri, rbands=None, curves=None, Shear=False, mesh=False):
        """Initialize Loop.

        Parameters
        ----------
        tri : triangulation2D object (really child class of triangulation2D)
            tri is used as a basis with which the input curve options (rbands,
            curves, and mesh) are turned into intersection coordinates.
            Several tri methods are used to initialize weightlist.

        rbands : list of lists (of ints)
            rbands is a collection (outer list) of bands (inner lists). Each
            band is a list of point ids.  This list represents a CCW oriented
            loop in of the points in tri, with the loop going around a point
            on the side that makes a 'large' (>pi) angle with the lines
            connecting this point to the 2 adjacent points in the band.  Bands
            equvalent up to cyclic shifts of the points in the list. This is
            often used with just two point ids to create a band the goes from
            one point to another.

        curves : list of lists
            curves is a collection (outer list) of curves (inner lists). Each
            curve is a list with four elements: The first is a list of point
            positions (each point position is a list: [x,y]), which define a
            sequence of lines connected end to end. The second is a boolean
            (is_closed) which if signals that the point position list should
            wrap-around (True), or be considered a curve with end- points
            (False). The third is a pair (list) of 2 booleans which determine
            if we put a cap (circle around the closest point) around end 1 and
            end 2 of the curve. This is only relevant if the loop is not
            closed.  The final element is a number (wadd, can be int or float)
            that is the weight we add to crossed edges.

        Shear : bool
            Flag to denote the type of representation: False for regular
            intersection coordinates, and True for shear coordinates.

        mesh : bool
            If True, rbands and curves will be ignored, and the loop will be
            initialized in the 'mesh' configuration, where each edge has a
            segment of the loop connecting adjacent points. Also sets
            Shear = True if True.

        Notes
        -----
            rbands is already a topological specification of a loop (points in
            tri), while curves are a geometric specification of a loop.
            Generally, avoid loops which trasversely intersect each other, as
            the way the 'X' is reconnected is dependent on the local details
            of the triangulation.  However, intersections will still result
            in valid loop coordinates.
            Mesh is useful for picking up mixing thoughout the entire domain.
        """
        self.weightlist = [0 for i in range(tri._totalnumedges)]
        self.Shear = Shear
        if mesh:
            for i in range(len(self.weightlist)):
                self.weightlist[i] = -1.0
                # represents bands pinned to adjacent
                # points in this triangulation.
            self.Shear = True  # mesh must be evaluated with shear coordinates
        else:
            if not self.Shear:
                if rbands is not None:
                    tri._BandWeightInitialize(rbands, LoopIn=self)
                if curves is not None:
                    tri._CurveWeightInitialize(curves, LoopIn=self)
            else:
                RegLoop = Loop(tri)
                if rbands is not None:
                    tri._BandWeightInitialize(rbands, LoopIn=RegLoop)
                if curves is not None:
                    tri._CurveWeightInitialize(curves, LoopIn=RegLoop)
                #  This first creates a regular loop (regular coordinates),
                #  then feeds this into the triangulation object to get the
                #  shear coordinates
                tri._ShearWeightInitialize(RegLoop, self)

    def GetWeightTotal(self):
        if not self.Shear:
            return sum(self.weightlist)
        else:
            WT = 0
            for i in range(len(self.weightlist)):
                WT += abs(self.weightlist[i])
            return WT

    def ProjectivizeWeights(self):
        mwv = max(max(self.weightlist), abs(min(self.weightlist)))
        self.weightlist = [x/mwv for x in self.weightlist]

    def LoopCopy(self):
        return copy.deepcopy(self)
# End of Loop Class ##########################################################


# WeightOperator Class #######################################################
class WeightOperator(WeightOperator_Base):
    """Class representing an operator that acts on loops.

        It is generated every time a triangulation flip occurs during the
        evolution of a triangulation object, and holds the information needed
        to update the weightlist of a loop.

    Attributes
    ----------
    eids : list of 5 ints
        The edge ids of the 5 elements in a loop weightlist whose
        corresponding triangulation edges surround a triangulation flip.
        For the quadrilateral whose diagonal edge will be flipped, the order
        of the edge ids in eids is: central diagonal first, then CCW perimeter
        edges, starting with an edge that, along with its quadrilateral
        opposite, form a "Z" with the central diagonal edge.

    time : real number
        the global time at which the flip happens

    Methods
    -------
    Update(LoopIn, Reverse = False)
        Updates the weightlist attribute of LoopIn
    """

    def __init__(self, IndexSet, TimeIn=None):
        """Initialize WeightOperator.

        Parameters
        ----------
        IndexSet : list of (5) ints
            IndexSet contains the edge ids needed for updating a loop
            weightlist during a triangulation flip.

        TimeIn : real number
            the global time at which the flip happens
        """
        self.eids = IndexSet
        self.time = TimeIn

    def Update(self, LoopIn, Reverse=False):
        """Update the weightlist attribute of LoopIn.

        Parameters
        ----------
        LoopIn : Loop Object
            This is the loop whose weightlist will be updated (in place).

        Reverse : bool
            Reverse = True is used to evolve loops backwards in time.
            Default is False.
        """
        WL = [LoopIn.weightlist[x] for x in self.eids]
        if not LoopIn.Shear:
            LoopIn.weightlist[self.eids[0]] = (max(WL[1] + WL[3],
                                                   WL[2] + WL[4]) - WL[0])
            # The main equation for updating intersection coordinates
            # Note that it workds equally well forward/backward in time
        else:
            # For Shear weights, the surrounding quadrilateral weights
            # are also modified
            Diag = WL[0]
            if not Diag == 0:
                LoopIn.weightlist[self.eids[0]] = -Diag
                if Diag > 0:
                    if not Reverse:
                        LoopIn.weightlist[self.eids[2]] += Diag
                        LoopIn.weightlist[self.eids[4]] += Diag
                    else:
                        LoopIn.weightlist[self.eids[1]] += Diag
                        LoopIn.weightlist[self.eids[3]] += Diag
                else:
                    if not Reverse:
                        LoopIn.weightlist[self.eids[1]] += Diag
                        LoopIn.weightlist[self.eids[3]] += Diag
                    else:
                        LoopIn.weightlist[self.eids[2]] += Diag
                        LoopIn.weightlist[self.eids[4]] += Diag
# End of WeightOperator Class ################################################


# PlotParameters Class ######################################################
@dataclass
class PlotParameters(PlotParameters):
    """Parameters for plotting.

    Class containing all of the parameters used in plotting the
    triangulation and loops, and their default values.

    Attributes
    ----------
    filename : str
        The filename (including local path) to save the figure as.
        If None (default), then then the figure is printed to screen.

    triplot : bool
        Flag - prints the background triangulation if True (default) and
        excludes it if False.

    Delaunay : bool
        Flag - if True then uses Voronoi-based control points to draw the
        train-track representation of the loops.  If False (default), then
        triangle centers are used as control points.

    DelaunayAdd : bool
        Flag - A different Voronoi-based control point plotting system for
        the train-tracks.  This represents the train-track weights as line
        widths, which join naturally at train-track switch locations. This
        is only relevant if Delaunay is True.

    Bounds : list of lists
        Bounds has the format [[x_min, y_min],[x_max, y_max]], and determines
        the bounding box for plotting.  This is usually set automatically.

    FigureSizeX : float
        The width of the image in inches.  The height is automatically
        calculated based on Bounds.

    dpi : int
        The dots per inch.  Increase to increase the resolution and size of
        resulting image file.

    ptlabels : bool
        If True, the integer label for each point is plotted next to the
        point. False is default.  Mainly used for visually finding groups of
        points to encircle with a band.

    markersize : float
        Sets the markersize of the points.

    linewidth_tri : float
        The line width of the background triangulation.

    linecolor_tri : str
        The color of the triangulation lines. Default is 'g' (green).

    color_weights : bool
        If True, then the individual segments of the train-track will be
        colored based on their weights.  This is one way to encode weight
        information in the plots.  Default is False.

    log_color : bool
        If True these colors will be assigned using the log of the weights. If
        False (default), the weights them-selves will determine the color
        scale

    color_map : str
        The color map to be used, default is 'inferno_r'.

    linewidth_tt : float
        The line width of the train-track.  If DelaunayAdd is True, then this
        is the maximum line-width

    linecolor_tt : str
        The line color of the train-track. Default is 'r' (red).

    alpha_tt : float
        The opacity of the train-track.  Default is 1.0 (completely
        opaque/not transparent).

    frac : float
        For plotting with the Delaunay flag, this determined how curved the
        train-tracks appear.  A value of 1.0 is maximally curvy (no linear
        segments), while a value of 0.0 would be just straight lines on
        following the Voronoi skeleton.  Default is 0.9

    tt_lw_min_frac : float
        The minimum fraction of linewidth_tt that will be represented.  This
        means that all train-track segments with weights below this fraction
        of the maximum weight will be represented as this fraction of
        linewidth_tt.  All segments with larger weight will have a line width
        that linear in this range.

    boundary_points : bool
        If true, this sets a larger boundary (in ExpandedBounds), which
        included the boundary control points.  Default is False.

    ExpandedBounds : list
        A larger boundary that includes the control points.
    """

    #  main flags/choices
    Delaunay: bool = False
    DelaunayAdd: bool = False
    #  initial setup
    Bounds: list = None
    FigureSizeX: float = 8
    dpi: int = 200
    ptlabels: bool = False
    markersize: float = 2.0
    linewidth_tri: float = 0.5
    linecolor_tri: str = 'g'
    color_weights: bool = False
    log_color: bool = True
    color_map: str = 'inferno_r'
    #  train track specifications
    linewidth_tt: float = 1.0
    linecolor_tt: str = 'r'
    alpha_tt: float = 1.0
    #  Delaunay
    frac: float = 0.9
    #  DelaunayAdd
    tt_lw_min_frac: float = 0.05
    _conversion_factor: float = None  # internal only
    _max_weight: int = None  # internal only
    boundary_points: bool = False
    ExpandedBounds: list = None
# End of PlotParameters Class ###############################################


# triangulation2D Class ######################################################
class triangulation2D(triangulation2D_Base):
    """The central class in the overal Topological Advection algorithm.

    This class represents a triangulation of data points in a 2D
    domain.  It has methods for evolving the triangulation due to the
    motion of data points, acting as a basis for encoding loops,
    accumulating weight operators, and plotting.  This abstract base
    class then has different child classes for different situations
    (doubly periodic boundary conditions, given boundary, etc.).

    Note: Only the main attribues and methods are listed here.

    Note: The triangulation is initialized as a Delaunay triangulation.

    Attributes
    ----------
    pointlist : list
        A list of simplex objects.  Object at index i has the point with
        point id i in its point list.  Allows for O(1) lookup of points in the
        triangulation.  Note, not every simplex is in this list.

    pointpos : list
        A list of the [x,y] positions for the points at the current time

    pointposfuture : list
        List of the [x,y] positions at the next time step.  Used with the
        Evolution method.

    simplist : list
        List of all of the simplices that make up the triangulation.
        Individual simplices have an id (SLindex) that indicates their
        location in this list.

    WeightOperatorList : list
        List of WeightOperator objects.  As the triangulation is evolved
        forward due to point motions, retriangulations with edge flips are
        needed.  For each flip, we record the data needed to evolve a loop
        forward.  This list is ordered (increasing) in time.

    Domain : list
        The rectangular domain that bounds the points.  This is used to set
        the boundary control points.  The format is:
        [[x_min, y_min],[x_max, y_max]]


    Methods
    -------
    Evolve(ptlist, Maintain_Delaunay = False)
        This evolves the triangulation forward due to the motion of the points
        - new point positions in ptlist. Options for evolution via collapse
        events or to maintain a Delaunay triangulation.  For every edge flip
        needed, a WeightOperator is added to the WeightOperator list.

    OperatorAction(LoopIn, index = None, Reverse = False, option = 3)
        This evolves forward an individual loop object (i.e. updates its
        weightlist due to the action of the WeightOperators in
        WeightOperatorList).

    Plot(LoopIn = None, PP: PlotParameters = PlotParameters())
        This plots the triangulation and loop.  See PlotParameters data class
        documentation for details on the many options.

    TriCopy(EvolutionReset = True)
        This returns a copy of this triangulation2D object.
    """

    def __init__(self, ptlist, Domain=None, empty=False):
        """Triangulation Initialization.

        Parameters
        ----------
        ptlist : list
            ptlist is the list of [x,y] positions for the points at the
            initial time.

        Domain : list
            Domain = [[x_min,y_min],[x_max,y_max]]. The data points must be
            contained within this rectangular boundary at all times.  If
            None (Default), then this will be calculated from the max/min
            x/y values of the input ptlist.  Domain is used to generate the
            boundary control points.

        empty : bool
            Used for creating an empty object, which is then used for object
            copying.  Default is False
        """
        self._extrapoints = []
        self._extranum = 0
        self.Domain = Domain
        if not empty:
            self._SetControlPoints(ptlist)
        super().__init__(ptlist, empty)
        for simp in self.simplist:
            boundary_simp = True
            for j in range(3):
                if simp.points[j] < self._ptnum - self._extranum:
                    boundary_simp = False
                    break
            if boundary_simp:
                simp.IsBoundarySimp = True

    def _SetControlPoints(self, ptlist):
        #  This sets the boundary control points based on the specified Domain
        #  Two concentric rectangles are created with linear point density
        #  equal to that of the ptlist data.
        if self.Domain is None:
            self.Domain = HF.GetBoundingDomainSlice(ptlist, frac=0.1)
        #  now find number of points along x and y boundary lines
        npts = len(ptlist)
        Deltax = (self.Domain[1][0] - self.Domain[0][0])
        Deltay = (self.Domain[1][1] - self.Domain[0][1])
        a_ratio = Deltax/Deltay
        npts_x = int(np.sqrt(npts*a_ratio))
        npts_y = int(npts/npts_x)
        dx = Deltax/(npts_x-1)
        dy = Deltay/(npts_y-1)
        delta = 1e-6*min(dx, dy)
        x_set = np.linspace(self.Domain[0][0], self.Domain[1][0],
                            num=npts_x, endpoint=True)
        #  note: the arrays with a sqrt are used to force the outer rectangle
        #  to break degeneracy and be convex.
        x_sety = np.sqrt(1.0 - np.abs(np.linspace(-1.0, 1.0, num=npts_x,
                                                  endpoint=True)))*delta
        y_set = np.linspace(self.Domain[0][1], self.Domain[1][1], num=npts_y,
                            endpoint=True)
        y_setx = np.sqrt(1.0-np.abs(np.linspace(-1.0, 1.0, num=npts_y,
                                                endpoint=True)))*delta
        x_set2 = np.linspace(self.Domain[0][0]-dx/2, self.Domain[1][0]+dx/2,
                             num=npts_x+1, endpoint=True)
        x_sety2 = np.sqrt(1.0-np.abs(np.linspace(-1.0, 1.0, num=npts_x+1,
                                                 endpoint=True)))*delta
        y_set2 = np.linspace(self.Domain[0][1]-dy/2, self.Domain[1][1]+dy/2,
                             num=npts_y+1, endpoint=True)
        y_setx2 = np.sqrt(1.0-np.abs(np.linspace(-1.0, 1.0, num=npts_y+1,
                                                 endpoint=True)))*delta
        Top1 = [[x_set[i], self.Domain[1][1] + x_sety[i]]
                for i in range(len(x_set))]
        Bot1 = [[x_set[i], self.Domain[0][1] - x_sety[i]]
                for i in range(len(x_set))]
        Top2 = [[x_set2[i], self.Domain[1][1] + dy/2 + x_sety2[i]]
                for i in range(len(x_set2))]
        Bot2 = [[x_set2[i], self.Domain[0][1] - dy/2 - x_sety2[i]]
                for i in range(len(x_set2))]
        Left1 = [[self.Domain[0][0] - y_setx[i], y_set[i]]
                 for i in range(1, len(y_set)-1)]
        Right1 = [[self.Domain[1][0] + y_setx[i], y_set[i]]
                  for i in range(1, len(y_set)-1)]
        Left2 = [[self.Domain[0][0] - dx/2 - y_setx2[i], y_set2[i]]
                 for i in range(1, len(y_set2)-1)]
        Right2 = [[self.Domain[1][0] + dx/2 + y_setx2[i], y_set2[i]]
                  for i in range(1, len(y_set2)-1)]
        self._extrapoints = (Top1 + Bot1 + Top2 + Bot2
                             + Left1 + Right1 + Left2 + Right2)
        self._extranum = len(self._extrapoints)
        self._ptnum = len(ptlist) + self._extranum

    def _LoadPos(self, ptlist):
        #  Combining the bounding simplex points with the regular points'
        #  positions (adding the extra point to the end rather than the
        #  begining is dicated by plotting concerns)
        self.pointpos = ptlist + self._extrapoints

    def _SetInitialTriangulation(self):
        #  This takes the set of points (including boundary control points),
        #  calculates the Delaunay Triangulation, creates and links the
        #  simplices, creates the pointlist, and sets the simplex edge ids
        temppoints = np.array(self.pointpos)
        #  create the initial Delaunay triangulation.  The option forces the
        #  creation of simplices for degenerate points by applying a random
        #  perturbation.
        temptri = Delaunay(temppoints, qhull_options="QJ Pp")
        numsimp = temptri.simplices.shape[0]
        self.simplist = []
        #  first create the list of simplex2D objects
        #  not linked together yet - need to create every object first
        for i in range(numsimp):
            self.simplist.append(simplex2D(temptri.simplices[i].tolist()))
            self.simplist[-1].SLindex = i
        #  now create the links
        for i in range(numsimp):
            linklist = temptri.neighbors[i].tolist()
            for j in range(len(linklist)):
                #  if -1 then the simplex already points to None
                #  true for neighbors of boundary simplices
                if not linklist[j] == -1:
                    self.simplist[i].simplices[j] = self.simplist[linklist[j]]
        self._SetPointList()
        self._SetEdgeIds()

    def _SetPointList(self):
        #  creates the pointlist with links to individual simplices
        self.pointlist = [None for i in range(self._ptnum)]
        #  Go through each simplex and add that simplex to each slot in the
        #  pointlist that corresponds to an included point if the slot
        #  contains None (possibly more efficient way to do this)
        for simp in self.simplist:
            for j in range(3):
                if self.pointlist[simp.points[j]] is None:
                    self.pointlist[simp.points[j]] = simp

    def _SetEdgeIds(self):
        #  Assign each edge an index.
        #  The index is just taken from an incremental counter.
        edgecounter = 0
        for simp in self.simplist:
            for j in range(3):
                if simp.edgeids[j] is None:
                    simp.edgeids[j] = edgecounter
                    if not simp.simplices[j] is None:
                        pt = simp.points[(j+1) % 3]
                        Lid = (simp.simplices[j].LocalID(pt)+1) % 3
                        simp.simplices[j].edgeids[Lid] = edgecounter
                    edgecounter += 1
        self._totalnumedges = edgecounter

    def Evolve(self, ptlist, Maintain_Delaunay=False):
        """Evolve the triangulation forward.

        Main method for evolving the state of the triangulation forward in
        time. This assumes that the starting triangulation is good
        (no negative areas).

        Parameters
        ----------
        ptlist : list
            The new time-slice data; the list of [x,y] positions for the
            points at the next time-step.

        Maintain_Delaunay : bool
            If Maintain_Delaunay is True (False is default), then after all
            of the collapse events are acounted for, extra edge flips will be
            used to ensure that the triangulation is Delaunay.
        """
        #  Overview: load the new positions, find the events, deal with the
        #  events, update the current position, and maintain delaunay if
        #  needed.
        self._LoadNewPos(ptlist)
        EventLists = self._GetEvents()
        #  GEvolve deals with the events in CollapseEventList
        #  and CrossingList (if periodic boundaries) in order
        self._GEvolve(EventLists)
        self._UpdatePtPos()
        self._atstep += 1
        if Maintain_Delaunay:
            self.MakeDelaunay()
            #  after the atstep increment so that the operators
            #  will have the correct time-stamp.

    def _LoadNewPos(self, ptlist):
        self._LoadPos(ptlist)

    def _GetEvents(self):
        #  Note that this wraps GetCollapseEvents so that it has the
        #  same signature at for the TopAdv_PBC case
        return self._GetCollapseEvents()

    def _GetCollapseEvents(self):
        """Find triangle collapse events.

        This finds all of the events where a triangle will go through zero
        area in the course of the points evolving from this time to the next
        time-step.

        Returns
        -------
        list
            A list of the simplices that collapse and the time of their
            collapse (bundled as a list of two items).  This list is sorted
            in decending order so that removing from the end (smallest times
            first) inccurs the smallest computational cost.
        """
        collapsesimplist = []
        if self._Vec:
            AZT_bool, AZT_time = self._AreaZeroTimeMultiple()
            collapsesimplist = [[self.simplist[i], AZT_time[i]]
                                for i in range(len(self.simplist))
                                if AZT_bool[i]]
        else:
            for simp in self.simplist:
                AZT_bool, AZT_time = self._AreaZeroTimeSingle(simp)
                if AZT_bool:
                    collapsesimplist.append([simp, AZT_time])
        collapsesimplist.sort(key=itemgetter(1), reverse=True)
        return collapsesimplist

    def _AreaZeroTimeSingle(self, SimpIn, Tin=0):
        """Calculate collapse time.

        Finds whether (and when) a triangle (simp) goes through zero area.

        Parameters
        ----------
        simp : simplex2D object
            The simplex to consider.

        Tin : float
            The lower bound on the time window to consider.

        Returns
        -------
        list
            Returns a pair [IsSoln, TimeOut], where IsSoln is a boolean that
            is True if the first time at which the area goes through zero is
            between Tin and 1, and False if not. For IsSoln == True,
            TimeOut gives this time.
        """
        #  Get the start and end x,y coordinate for each of the three points
        ptlist = SimpIn.points
        IP0x, IP0y = self.pointpos[ptlist[0]]
        IP1x, IP1y = self.pointpos[ptlist[1]]
        IP2x, IP2y = self.pointpos[ptlist[2]]
        FP0x, FP0y = self._pointposfuture[ptlist[0]]
        FP1x, FP1y = self._pointposfuture[ptlist[1]]
        FP2x, FP2y = self._pointposfuture[ptlist[2]]
        return HF.AreaZeroTimeBaseSingle(IP0x, IP0y, IP1x, IP1y, IP2x, IP2y,
                                         FP0x, FP0y, FP1x, FP1y, FP2x, FP2y,
                                         Tin)

    def _AreaZeroTimeMultiple(self, Tin=0):
        """Vectorized calculation of collapse times.

        Goes through every simplex and looks for whether the area zero time
        is between Tin and 1.  Similar to AreaZeroTimeSingle, but wrapping up
        the info in numpy arrays to get vectorization and jit boost.

        Parameters
        ----------
        Tin : float
            The lower bound on the time window to consider.

        Returns
        -------
        list
            list of booleans, indicating whether the simplex (with this
            index) collapsed within the interval.

        list
            list of floats, giving the collapse times.
        """
        nsimps = len(self.simplist)
        pts0 = np.array([self.simplist[i].points[0] for i in range(nsimps)])
        pts1 = np.array([self.simplist[i].points[1] for i in range(nsimps)])
        pts2 = np.array([self.simplist[i].points[2] for i in range(nsimps)])
        npptpos = np.array(self.pointpos)
        npptposf = np.array(self._pointposfuture)
        IP0x, IP0y = npptpos[pts0, 0], npptpos[pts0, 1]
        IP1x, IP1y = npptpos[pts1, 0], npptpos[pts1, 1]
        IP2x, IP2y = npptpos[pts2, 0], npptpos[pts2, 1]
        FP0x, FP0y = npptposf[pts0, 0], npptposf[pts0, 1]
        FP1x, FP1y = npptposf[pts1, 0], npptposf[pts1, 1]
        FP2x, FP2y = npptposf[pts2, 0], npptposf[pts2, 1]
        return HF.AreaZeroTimeBaseVec(IP0x, IP0y, IP1x, IP1y, IP2x, IP2y,
                                      FP0x, FP0y, FP1x, FP1y, FP2x, FP2y, Tin)

    def _GEvolve(self, EventLists):
        """Process the events in EventLists.

        Processes an ordered list of events (collapse) and does edge flips to
        update the triangulation.  Also adds in new events as needed. Finished
        when there are no more events in the time interval, and the
        triangulation is consistent with the new set of points.
        """
        delta = 1e-10
        while len(EventLists) > 0:
            CollSimp, currenttime = EventLists[-1]
            #  deal with simplex collapse events here
            newsimps, delsimp = self._SFix(CollSimp, currenttime)
            #  returns ... [[leftsimp,rightsimp],topsimp (old)]
            del EventLists[-1]  # get rid of the evaluated event
            #  Find the time of zero area for simplex event
            collapsed, collapse_time = self._AreaZeroTimeSingle(
                delsimp, currenttime - delta)
            if collapsed:
                #  and delete it if needed
                HF.BinarySearchDel(EventLists, [delsimp, collapse_time])
            #  Go through the newsimps list and see if each object goes
            #  through zero area in the remaining time (if so, add to
            #  EventList with the calulated time to zero area)
            for simp in newsimps:
                collapsed, collapse_time = self._AreaZeroTimeSingle(
                    simp, currenttime - delta)
                if collapsed:
                    #  insert in the event list at the correct spot
                    HF.BinarySearchIns(EventLists, [simp, collapse_time])

    def _SFix(self, SimpIn, tcollapse):
        #  Fixing a simplex and the surrounding affected simplices. This
        #  returns the two new simplices, so that they can be possibly added
        #  to the local event list, also the bad simplex so it can be removed
        #  (if needed from the local event list)

        #  `colind` is the local index of the offending point during the
        #  area collapse
        colind = self._CollapsePt(SimpIn, tcollapse)
        Topsimp = SimpIn.simplices[colind]
        edge_id = SimpIn.edgeids[colind]
        globaltime = self._atstep + tcollapse
        newsimps = self._EdgeFlip([SimpIn, Topsimp], edge_id, globaltime)
        #  EdgeFlip does most of the work in flipping the edge and
        #  cleaning up linking
        return [newsimps, Topsimp]
        #  return the two new simplices, so that they can be checked to see
        #  if they need to be included in any update to the local event list.
        #  Also return the bad simplex to remove any instance from the
        #  event list.

    def _CollapsePt(self, SimpIn, tcol):
        #  This returns the point (internal id) that passes through its
        #  opposite edge during an area collapse event known to occur
        #  at t = tcol
        #  first get the positions of the 3 points at the time of collapse
        colpos = self._GetSimpCurrentLoc(SimpIn, tcol)
        d0 = ((colpos[2][0] - colpos[0][0])*(colpos[1][0] - colpos[0][0]) +
              (colpos[2][1] - colpos[0][1])*(colpos[1][1] - colpos[0][1]))
        #  This is the dot product of (z2-z0) and (z1-z0)
        if d0 < 0:
            return 0
        else:
            d1 = ((colpos[2][0] - colpos[1][0])*(colpos[0][0]-colpos[1][0]) +
                  (colpos[2][1] - colpos[1][1])*(colpos[0][1] - colpos[1][1]))
            if d1 < 0:
                return 1
            else:
                return 2
            #  Note: don't need to calculate the last dot product.
            #  If the first two are >0, this must be <0

    def _GetSimpCurrentLoc(self, SimpIn, teval):
        #  This returns the linearly interpolated positions of each point in
        #  SimpIn at the time 0 < teval < 1.
        ptlist = SimpIn.points
        return [self._GetCurrentLoc(pt_ind, teval) for pt_ind in ptlist]

    def _GetCurrentLoc(self, pt_ind, teval):
        posi = self.pointpos[pt_ind]
        posf = self._pointposfuture[pt_ind]
        return [((posf[k]-posi[k])*teval + posi[k]) for k in range(2)]

    def _EdgeFlip(self, AdjSimps, EdgeShare, TimeIn=None):
        """Flip an edge in the triangulation.

        EdgeFlip locally re-triangulates the triangulation by removing an
        edge that divides two adjacent triangles in a quadrilateral, and
        replaces it with the other diagonal of this quadrilateral.  This
        removes the old simplices, creates new ones, and links them up in the
        triangulation. EdgeFlip also creates the appropriate WeightOperator
        object and adds it to the WeightOperator list.

        Parameters
        ----------
        AdjSimps : list of 2 simplex2D objects
            These are the two simplices that share the edge to be flipped

        EdgeShare : int
            The edge id of the edge to be flipped.  While this can almost
            always be found from AdjSimps, the redundancy helps in certain
            cases.

        TimeIn : float
            This is the time when the event occured that required a flip.
            It is added as part of the data in the WeightOperator object.

        Returns
        -------
        list of 2 simplex2D objects
            The two new simplices.  Returned so that the calling function
        """
        Simp, Topsimp = AdjSimps
        #  local ids
        bptlid = Simp.edgeids.index(EdgeShare)
        bpt = Simp.points[bptlid]
        rptlid = (bptlid+1) % 3
        lptlid = (bptlid+2) % 3
        rpt = Simp.points[rptlid]
        lpt = Simp.points[lptlid]
        tptuid = Topsimp.edgeids.index(EdgeShare)
        tpt = Topsimp.points[tptuid]
        lptuid = (tptuid+1) % 3
        rptuid = (tptuid+2) % 3
        #  create the new simplices
        rsimp = simplex2D([bpt, rpt, tpt])  # new right simplex
        lsimp = simplex2D([bpt, tpt, lpt])  # new left simplex
        #  create the list of edge ids for the weight operator
        WeightIDs = [EdgeShare, Topsimp.edgeids[lptuid],
                     Topsimp.edgeids[rptuid], Simp.edgeids[rptlid],
                     Simp.edgeids[lptlid]]
        #  create the weight operater and append to the list
        self.WeightOperatorList.append(WeightOperator(WeightIDs, TimeIn))
        #  create the links these simplices have to other simplices
        rsimp.SimpLink(Topsimp.simplices[lptuid])
        lsimp.SimpLink(Topsimp.simplices[rptuid])
        rsimp.SimpLink(Simp.simplices[lptlid])
        lsimp.SimpLink(Simp.simplices[rptlid])
        rsimp.SimpLink(lsimp)
        #  reassign the weight ids
        rsimp.edgeids[0] = WeightIDs[1]
        rsimp.edgeids[1] = WeightIDs[0]
        rsimp.edgeids[2] = WeightIDs[4]
        lsimp.edgeids[0] = WeightIDs[2]
        lsimp.edgeids[1] = WeightIDs[3]
        lsimp.edgeids[2] = WeightIDs[0]
        #  replace the two bad simplices in the simplex list
        #  with the two new ones
        Simpindex = Simp.SLindex
        self.simplist[Simpindex] = lsimp
        lsimp.SLindex = Simpindex
        Topsimpindex = Topsimp.SLindex
        self.simplist[Topsimpindex] = rsimp
        rsimp.SLindex = Topsimpindex
        #  look through the simplex point list to see if either of the
        #  bad simplices were there and if so, replace.
        if self.pointlist[bpt] is Simp:
            self.pointlist[bpt] = rsimp
        if (self.pointlist[rpt] is Simp) or (self.pointlist[rpt] is Topsimp):
            self.pointlist[rpt] = rsimp
        if self.pointlist[tpt] is Topsimp:
            self.pointlist[tpt] = lsimp
        if (self.pointlist[lpt] is Simp) or (self.pointlist[lpt] is Topsimp):
            self.pointlist[lpt] = lsimp
        #  Delete all the references to simplices in both of the bad simplices
        for i in range(3):
            Simp.simplices[i] = None
            Topsimp.simplices[i] = None
        return [lsimp, rsimp]

    def MakeDelaunay(self):
        """Flip edges until the triangulation is Delaunay.

        MakeDelaunay takes the current triangulation and, through a series
        of edge flips, changes it into the Delaunay triangulation for this
        point configuration.  This function changes the underlying
        triangulation
        """
        IsD, InteriorEdge, EdgeBSimps = None, None, None
        if self._Vec:  # vectorized version (marginal improvements)
            IsD, InteriorEdge, EdgeBSimps = self._IsDelaunay()
        else:
            EdgeBSimps = [None for i in range(self._totalnumedges)]
            EdgeUsed = [False for i in range(self._totalnumedges)]
            IsD = [False for i in range(self._totalnumedges)]
            for simp in self.simplist:
                if not simp.IsBoundarySimp:
                    for j in range(3):
                        if not simp.simplices[j].IsBoundarySimp:
                            edgeid = simp.edgeids[j]
                            if not EdgeUsed[edgeid]:
                                EdgeUsed[edgeid] = True
                                EdgeBSimps[edgeid] = [
                                    [simp, simp.simplices[j]], edgeid, True]
                                IsD[edgeid] = self._IsLocallyDelaunay(
                                    [simp, simp.simplices[j]])
            InteriorEdge = EdgeUsed

        EdgeList = [EdgeBSimps[i] for i in range(self._totalnumedges)
                    if not IsD[i] and InteriorEdge[i]]
        EdgeList_Epos = [None for i in range(self._totalnumedges)]
        for i in range(len(EdgeList)):
            EdgeList_Epos[EdgeList[i][1]] = i
        #  go through the edge list and start flipping edges
        while len(EdgeList) > 0:
            EdgeSimps, edge_id, checked = EdgeList.pop()
            EdgeList_Epos[edge_id] = None
            if (not EdgeSimps[0].IsBoundarySimp and
                    not EdgeSimps[1].IsBoundarySimp):
                Flip = True
                if not checked:
                    Flip = not self._IsLocallyDelaunay(EdgeSimps)
                if Flip:
                    LRsimps = self._EdgeFlip(EdgeSimps, edge_id, self._atstep)
                    for i in range(2):  # Left and right simplices
                        loc = LRsimps[i].edgeids.index(edge_id)
                        lrsimp = LRsimps[i]
                        for j in range(2):  # upper and lower simplices
                            eid = lrsimp.edgeids[(loc+1+j) % 3]
                            if InteriorEdge[eid]:
                                adjsimp = lrsimp.simplices[(loc+1+j) % 3]
                                ELinsert = [[lrsimp, adjsimp], eid, False]
                                if EdgeList_Epos[eid] is None:
                                    EdgeList_Epos[eid] = len(EdgeList)
                                    EdgeList.append(ELinsert)
                                else:
                                    EdgeList[EdgeList_Epos[eid]] = ELinsert

    def _IsDelaunay(self):
        """Is the triangulation Delaunay.

        IsDelaunay outputs an array (length = number of edges) of booleans,
        which indicate if the quadrilateral with the ith edge as a diagonal is
        Delaunay. Also outputs an array of the pairs of simplices which bound
        each edge. This calls IsDelaunayBase (from a helper function module)
        for a jit speed-up.
        """
        Ax, Ay, Bx, By, Cx, Cy, Dx, Dy = [np.zeros(self._totalnumedges)
                                          for i in range(8)]
        EdgeUsed = [False for i in range(self._totalnumedges)]
        BoundingSimps = [None for i in range(self._totalnumedges)]
        for simp in self.simplist:
            if not simp.IsBoundarySimp:
                for j in range(3):
                    if not simp.simplices[j].IsBoundarySimp:
                        edgeid = simp.edgeids[j]
                        if not EdgeUsed[edgeid]:
                            EdgeUsed[edgeid] = True
                            Apt = simp.points[(j+2) % 3]
                            Ax[edgeid], Ay[edgeid] = self.pointpos[Apt]
                            Bpt = simp.points[j]
                            Bx[edgeid], By[edgeid] = self.pointpos[Bpt]
                            Cpt = simp.points[(j+1) % 3]
                            Cx[edgeid], Cy[edgeid] = self.pointpos[Cpt]
                            adjsimp = simp.simplices[j]
                            BoundingSimps[edgeid] = [[simp, adjsimp],
                                                     edgeid, True]
                            adjsimp_loc_id = adjsimp.edgeids.index(edgeid)
                            Dpt = adjsimp.points[adjsimp_loc_id]
                            Dx[edgeid], Dy[edgeid] = self.pointpos[Dpt]
        return (HF.IsDelaunayBaseWMask(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy,
                                       np.array(EdgeUsed)), EdgeUsed,
                BoundingSimps)

    def _IsLocallyDelaunay(self, AdjSimps):
        """Is the quadrilateral Delaunay.

        Given the two adjacent simplices, this determine if the
        configuration is locally Delaunay.  Returns True or False.
        """
        simp1, simp2 = AdjSimps
        locid = simp1.simplices.index(simp2)
        Apt = simp1.points[(locid+2) % 3]
        Ax, Ay = self.pointpos[Apt]
        Bpt = simp1.points[locid]
        Bx, By = self.pointpos[Bpt]
        Cpt = simp1.points[(locid+1) % 3]
        Cx, Cy = self.pointpos[Cpt]
        locid2 = simp2.simplices.index(simp1)
        Dpt = simp2.points[locid2]
        Dx, Dy = self.pointpos[Dpt]
        ADx = Ax - Dx
        ADy = Ay - Dy
        BDx = Bx - Dx
        BDy = By - Dy
        CDx = Cx - Dx
        CDy = Cy - Dy
        AD2 = ADx*ADx + ADy*ADy
        BD2 = BDx*BDx + BDy*BDy
        CD2 = CDx*CDx + CDy*CDy
        detvals = (ADx*(BDy*CD2 - CDy*BD2) - ADy*(BDx*CD2 - CDx*BD2)
                   + AD2*(BDx*CDy - CDx*BDy))
        return (detvals < 0)

    def OperatorAction(self, LoopIn, index=None, Reverse=False,
                       option=3, num_times=None):
        """Flip operator acting on a Loop.

        OperatorAction takes the accumulated operator list stored in
        WeightOperatorList and operates sucessively on the given Loop

        Parameters
        ----------
        LoopIn : Loop Object
            The weightlist of this loop will be modified in place

        index : list of 2 ints
            the start and stop index can also be specified to break this up
            into stages (only used for option 1 and 2). Default is None

        Reverse : bool
            Reverse does the operator actions in reverse order (i.e. for
            loops in the final triangulation)

        option : int {1,2,3}
            option 1 just changes the data in the loop object
            option 2 also accumulates a weight list with the total weights
            after each operator has acted on the loop, and gives the global
            time of the operator action.
            option 3 (the default) returns a weight list which has the weights
            at the end of each time step (the intervals between each use of
            the Evolve method). This weight list does not have the time listed,
            as this is only know externally. this is most useful for producing
            a list that we can directly tie to an external list of times.
            This is what we need for extracting the topological entropy
            (hence the default option).

        Return
        ------
        No return - option 1
        list of floats - option 2 and 3 (list is weightlist)
        """
        startind = 0
        endind = len(self.WeightOperatorList)-1
        if index is not None:
            startind = index[0]
            endind = index[1]
        if option == 1:
            if not Reverse:
                for i in range(startind, endind+1):
                    self.WeightOperatorList[i].Update(LoopIn)
            else:
                for i in range(endind, startind-1, -1):
                    self.WeightOperatorList[i].Update(LoopIn, Reverse)
        elif option == 2:
            WeightList = []
            if not Reverse:
                WeightList.append([LoopIn.GetWeightTotal(), 0])
                for i in range(startind, endind+1):
                    self.WeightOperatorList[i].Update(LoopIn)
                    WeightList.append([LoopIn.GetWeightTotal(),
                                       self.WeightOperatorList[i].time])
            else:
                finaltime = math.ceil(self.WeightOperatorList[i].time)
                WeightList.append([LoopIn.GetWeightTotal(), finaltime])
                for i in range(endind, startind-1, -1):
                    self.WeightOperatorList[i].Update(LoopIn, Reverse)
                    WeightList.append([LoopIn.GetWeightTotal(),
                                       self.WeightOperatorList[i].time])
            return WeightList
        elif option == 3:
            WeightList = []
            if not Reverse:
                prevtime = 0
                for i in range(len(self.WeightOperatorList)):
                    thistime = math.ceil(self.WeightOperatorList[i].time)
                    if thistime > prevtime:
                        prevtime = thistime
                        currentweight = LoopIn.GetWeightTotal()
                        while len(WeightList) < thistime:
                            WeightList.append(currentweight)
                    self.WeightOperatorList[i].Update(LoopIn)
                WeightList.append(LoopIn.GetWeightTotal())
            else:
                endtime = math.ceil(self.WeightOperatorList[-1].time)
                prevtime = endtime
                for i in range(len(self.WeightOperatorList)-1, -1, -1):
                    thistime = math.floor(self.WeightOperatorList[i].time)
                    if thistime < prevtime:
                        prevtime = thistime
                        currentweight = LoopIn.GetWeightTotal()
                        while len(WeightList) < endtime-thistime:
                            WeightList.append(currentweight)
                    self.WeightOperatorList[i].Update(LoopIn, Reverse)
                WeightList.append(LoopIn.GetWeightTotal())
            if num_times is not None:
                while len(WeightList) < num_times:
                    WL_last = WeightList[-1]
                    WeightList.append(WL_last)
            return WeightList
        else:
            print("Need to choose one of the options 1, 2, or 3")

    def _BandWeightInitialize(self, rbands, LoopIn):
        """Initialize a loop with band data.

        This initializes the edge weights in `LoopIn` that correspond to a
        given band (or set of bands) in `rbands`.

        Parameters
        ----------
        rbands : list
            Each element in the list represents a band, and consists of two
            items: the list of points which define a band (see Loop class
            documentation), and the weight to add to the loop weightlist.

        LoopIn : Loop object
            The weightlist of `LoopIn` will be modified to represent this
            additional set of bands being added in.
        """
        for band_data in rbands:
            band, wadd = band_data
            numpoints = len(band)
            AreAdjacent, CurveLeft = [], []
            for k in range(numpoints):
                AreAdjacent.append(
                    self._ArePointsAdjacent(band[k], band[(k+1) % numpoints]))
                triplepts = [band[(k+numpoints-1) % numpoints],
                             band[k], band[(k+1) % numpoints]]
                CurveLeft.append(self._DoesCurveLeft(triplepts))
            for j in range(numpoints):
                Bool1 = [CurveLeft[j], AreAdjacent[j],
                         CurveLeft[(j+1) % numpoints]]
                Bool2 = [AreAdjacent[(j+numpoints-1) % numpoints],
                         CurveLeft[j], AreAdjacent[j]]
                triplepts = [band[(j+numpoints-1) % numpoints], band[j],
                             band[(j+1) % numpoints]]
                self._AddWeightsAlongLine(
                    [band[j], band[(j+1) % numpoints]], Bool1, LoopIn, wadd)
                self._AddWeightsAroundPoint(triplepts, Bool2, LoopIn, wadd)

    def _ArePointsAdjacent(self, pt1, pt2):
        AreAdjacent = False
        goodind = (len(self.pointlist) - self._extranum)-1
        IsBndryEdge = False
        if pt1 > goodind:
            if pt2 > goodind:
                IsBndryEdge = True
            else:
                temppt = pt1
                pt1 = pt2
                pt2 = temppt
        if not IsBndryEdge:
            LRsimps = []
            LRsimpspos = []
            simpposcounter = 0
            StartSimp = self.pointlist[pt1]
            locid = StartSimp.LocalID(pt1)
            if pt2 in StartSimp.points:
                AreAdjacent = True
                LRsimps.append(StartSimp)
                LRsimpspos.append(simpposcounter)
            simpposcounter += 1
            NextSimp = StartSimp.simplices[(locid+1) % 3]
            locid = NextSimp.LocalID(pt1)
            while NextSimp is not StartSimp and len(LRsimpspos) < 2:
                if pt2 in NextSimp.points:
                    AreAdjacent = True
                    LRsimps.append(NextSimp)
                    LRsimpspos.append(simpposcounter)
                simpposcounter += 1
                NextSimp = NextSimp.simplices[(locid+1) % 3]
                locid = NextSimp.LocalID(pt1)
        if AreAdjacent:
            if LRsimpspos[1] == LRsimpspos[0]+1:
                return [AreAdjacent, [LRsimps[1], LRsimps[0]]]
            else:
                return [AreAdjacent, LRsimps]
        else:
            return [AreAdjacent, None]

    def _DoesCurveLeft(self, pttriple):
        pt1, pt2, pt3 = pttriple
        pos1 = self.pointpos[pt1]
        pos2 = self.pointpos[pt2]
        pos3 = self.pointpos[pt3]
        crossP = ((pos3[0] - pos2[0])*(pos1[1] - pos2[1]) -
                  (pos3[1] - pos2[1])*(pos1[0] - pos2[0]))
        return crossP >= 0

    def _IsLeft(self, linepts, ptin):
        #  this determines if the given point (ptin) is to the left of line
        #  that goes from the first to second point in linepts.
        #  Used in determining the edges crossed in an initial band
        pttriple = [ptin, linepts[0], linepts[1]]
        return self._DoesCurveLeft(pttriple)

    def _AddWeightsAlongLine(self, linepoints, Boolin, LoopIn, wadd=1.0):
        #  This takes the two points in linepoints and adds a weight of one
        #  (or non-default value) to any edges that are crossed by the line.
        pt1, pt2 = linepoints
        if Boolin[1][0]:
            #  case of adjacent points (i.e. the line between the
            #  points is an edge). Only if the curvelefts'
            #  (Boolin[0], Boolin[2]) are opposite one another,
            #  do we add a weight
            if Boolin[0] is not Boolin[2]:
                pt1rtlocid = Boolin[1][1][1].LocalID(pt1)
                edgeindex = Boolin[1][1][1].edgeids[(pt1rtlocid+1) % 3]
                LoopIn.weightlist[edgeindex] += wadd
        else:
            #  determine the direction (which simplex) to set out
            #  from that has pt1 as a point.
            stlocid, StartSimp = self._SimpInDir([pt1, pt2])
            endlocid, EndSimp = self._SimpInDir([pt2, pt1])
            if pt2 not in StartSimp.points:
                edgeindex = StartSimp.edgeids[stlocid]
                LoopIn.weightlist[edgeindex] += wadd
                leftpoint = StartSimp.points[(stlocid+2) % 3]
                CurrentSimp = StartSimp.simplices[stlocid]
                leftptloc = CurrentSimp.LocalID(leftpoint)
                while CurrentSimp is not EndSimp:
                    ptcompare = CurrentSimp.points[(leftptloc+2) % 3]
                    indexadd = 0
                    if not self._IsLeft(linepoints, ptcompare):
                        indexadd = 1
                    edgeindex = CurrentSimp.edgeids[(leftptloc+indexadd) % 3]
                    LoopIn.weightlist[edgeindex] += wadd
                    leftpoint = CurrentSimp.points[(leftptloc+indexadd+2) % 3]
                    CurrentSimp = CurrentSimp.simplices[
                        (leftptloc + indexadd) % 3]
                    leftptloc = CurrentSimp.LocalID(leftpoint)

    def _SimpInDir(self, linepoints):
        #  this returns the simplex (and local point id) that contains the
        #  first of linepoints, and has the line (to the second point)
        #  passing through it
        pt1, pt2 = linepoints
        StartSimp = self.pointlist[pt1]
        locpt = StartSimp.LocalID(pt1)
        ptright = StartSimp.points[(locpt+1) % 3]
        ptleft = StartSimp.points[(locpt+2) % 3]
        while not ((not self._IsLeft([pt1, pt2], ptright)) and
                   self._IsLeft([pt1, pt2], ptleft)):
            StartSimp = StartSimp.simplices[(locpt+1) % 3]
            locpt = StartSimp.LocalID(pt1)
            ptright = StartSimp.points[(locpt+1) % 3]
            ptleft = StartSimp.points[(locpt+2) % 3]
        return locpt, StartSimp

    def _AddWeightsAroundPoint(self, pttriple, Boolin, LoopIn, wadd=1):
        #  This takes the central point in pttriple and adds in the weight
        #  of wadd to each of the radial edges starting from the edge that
        #  is part of the simplex bisected by pt1 and pt2, to the edge that
        #  is part of the simplex bisected by pt2 and pt3
        pt1, pt2, pt3 = pttriple
        indadd = 1
        if not Boolin[1]:
            indadd = 2  # curve right triggers this
        stlocid, StartSimp = None, None
        if Boolin[0][0]:
            if not Boolin[1]:
                StartSimp = Boolin[0][1][0]
            else:
                StartSimp = Boolin[0][1][1]
            stlocid = StartSimp.LocalID(pt2)
        else:
            stlocid, StartSimp = self._SimpInDir([pt2, pt1])
        endlocid, EndSimp = None, None
        if Boolin[2][0]:
            if not Boolin[1]:
                EndSimp = Boolin[2][1][0]
            else:
                EndSimp = Boolin[2][1][1]
            endlocid = EndSimp.LocalID(pt2)
        else:
            endlocid, EndSimp = self._SimpInDir([pt2, pt3])
        edgeindex = StartSimp.edgeids[(stlocid+indadd) % 3]
        LoopIn.weightlist[edgeindex] += wadd
        CurrentSimp = StartSimp.simplices[(stlocid+indadd) % 3]
        ptloc = CurrentSimp.LocalID(pt2)
        while CurrentSimp is not EndSimp:
            edgeindex = CurrentSimp.edgeids[(ptloc+indadd) % 3]
            LoopIn.weightlist[edgeindex] += wadd
            CurrentSimp = CurrentSimp.simplices[(ptloc+indadd) % 3]
            ptloc = CurrentSimp.LocalID(pt2)

    def _CurveWeightInitialize(self, curves, LoopIn):
        """Initialize a loop with curve data.

        This initializes the edge weights in `LoopIn` that correspond to a
        given set of curves in `curves`.

        Parameters
        ----------
        curves : list
            Each element in the list represents a curve, and consists of four
            items: the list of point positions [[x_0,y_0],[x_1,y_1],...],
            whether the curve is closed (bool), whether the end points are
            pinned [bool,bool], and finally, the weight to add to the loop
            weightlist.

        LoopIn : Loop object
            The weightlist of LoopIn will be modified to represent this
            additional set of curves being added in.
        """
        for curve in curves:
            point_set, is_closed, end_pts_pin, wadd = curve
            edge_list = self._Get_Edges(point_set, is_closed, end_pts_pin)
            for edge in edge_list:
                LoopIn.weightlist[edge] += wadd

    def _Simp_Hop(self, pt_in, simp, line_big, edge_prev=None, next_edge=None):
        #  see if pt_in is in simp.  If not, find edge intersection and then
        #  get adj simp, find ref point and dx,dy for shift that matches edge
        #  from simp. Then calls self recursively. Stops when simp is found
        #  with this point interior.  returns list of pairs [simp, edge] along
        #  path
        #  first see if pt_in is in the simp
        if self._Tri_Contains(pt_in, simp):
            return [[simp, None]]
        else:
            vertices = np.array([copy.copy(self.pointpos[p])
                                 for p in simp.points])
            next_id = None
            if next_edge is not None:
                i = simp.edgeids.index(next_edge)
                Line_simp = [vertices[(i+1) % 3], vertices[(i+2) % 3]]
                if HF.IsIntersection(line_big, Line_simp):
                    next_id = i
            else:
                if edge_prev is None:
                    for i in range(3):
                        Line_simp = [vertices[(i+1) % 3], vertices[(i+2) % 3]]
                        if HF.IsIntersection(line_big, Line_simp):
                            next_id = i
                            break
                else:
                    for i in range(3):
                        if not simp.edgeids[i] == edge_prev:
                            Line_simp = [vertices[(i+1) % 3],
                                         vertices[(i+2) % 3]]
                            if HF.IsIntersection(line_big, Line_simp):
                                next_id = i
                                break
            if next_id is None:
                return None
            edge = simp.edgeids[next_id]
            next_simp = simp.simplices[next_id]
            return [[simp, edge]] + self._Simp_Hop(pt_in, next_simp, line_big,
                                                   edge_prev=edge)

    def _Tri_Contains(self, pt, simp):
        # Determines whether the triangle (simp) contains pt.
        vertices = np.array([copy.copy(self.pointpos[p]) for p in simp.points])
        trial_pt = np.array(pt)
        for i in range(3):
            c_i = HF.Curl(vertices[(i+1) % 3]-vertices[i],
                          trial_pt - vertices[i])
            if c_i < 0.0:
                return False
        return True

    def _Get_Edges(self, points, closed=True, end_pts_pin=[False, False]):
        #  Finds the list of edges crossed by the curve represented by
        #  the list of points (points).
        tree = KDTree(self.pointpos)
        _, nn = tree.query(points, k=1)
        simp_in = [self._Find_Simp(points[i], nn[i]) for i in range(len(nn))]
        edge_list = []
        ncl = 0
        if not closed:
            ncl = -1
        for i in range(len(points)+ncl):
            line_big = [points[i], points[(i+1) % len(points)]]
            simp_chain = self._Simp_Hop(points[(i+1) % len(points)],
                                        simp_in[i], line_big)
            edge_list += [simp_chain[k][1] for k in range(len(simp_chain)-1)]
        HF.Reduce_List(edge_list)
        if not closed and not end_pts_pin == [False, False]:
            temp_edge_list = []
            if end_pts_pin[0]:
                st_pt = simp_in[0].edgeids.index(edge_list[0])
                temp_edge_list += simp_in[0].EdgeNeighbors(
                    simp_in[0].points[st_pt])
            temp_edge_list += edge_list
            if end_pts_pin[1]:
                end_pt = simp_in[-1].edgeids.index(edge_list[-1])
                temp_edge_list += simp_in[-1].EdgeNeighbors(
                    simp_in[-1].points[end_pt])
            edge_list = temp_edge_list + edge_list[::-1]
        return edge_list

    def _Find_Simp(self, pt_in, nn_id):
        #  Finds the simplex which contains the input point
        #  uses the nearest neighbor point (nn_id)
        simp_set, l_id_set = self.pointlist[nn_id].SimpNeighbors(nn_id)
        for i in range(len(simp_set)):
            simp = simp_set[i]
            l_id = l_id_set[i]
            edge = simp.edgeids[l_id]
            line_big = [copy.copy(self.pointpos[nn_id]), pt_in]
            simp_chain = self._Simp_Hop(pt_in, simp, line_big, next_edge=edge)
            if simp_chain is not None:
                return simp_chain[-1][0]
        return None

    def _ShearWeightInitialize(self, RegLoop, LoopIn):
        """Initialize a loop a shear coordinate version of another loop.

        This takes the regular edge weights (for some band) encoded in
        `RegLoop`, and uses the triangulation connectivity to initialize
        `LoopIn`, which represents the band in shear coordinates.

        Parameters
        ----------
        RegLoop : Loop object
            A loop already initialized with regular coordinates.

        LoopIn : Loop object
            The loop that will be initialized with shear coordinates
        """
        for simp in self.simplist:
            for i in range(3):
                #  LoopIn must be initialized to all zeros (this catches
                #  the second time through)
                if LoopIn.weightlist[simp.edgeids[i]] == 0:
                    #  if the value for the regular loop is zero here, then
                    #  the shear coordinates should be zero
                    if not RegLoop.weightlist[simp.edgeids[i]] == 0:
                        WA = RegLoop.weightlist[simp.edgeids[(i+1) % 3]]
                        WB = RegLoop.weightlist[simp.edgeids[(i+2) % 3]]
                        xsimp = simp.simplices[i]
                        Lid = xsimp.LocalID(simp.points[(i+2) % 3])
                        WC = RegLoop.weightlist[xsimp.edgeids[Lid]]
                        WD = RegLoop.weightlist[xsimp.edgeids[(Lid+1) % 3]]
                        LoopIn.weightlist[simp.edgeids[i]] = (-WA+WB-WC+WD)//2

    #  Plotting
    def _PlotPrelims(self, PP: PlotParameters):
        """Preliminary setup for plotting.

        Handles the initial setup for the figure

        Parameters
        ----------
        PP : PlotParameters object
            For this method, the relevant PlotParameters attributes are:

        Bounds : list of lists
            Bounds has the format [[x_min, y_min],[x_max, y_max]], and
            determines the bounding box for plotting. This is usually set
            automatically.

        FigureSizeX : float
            The width of the image in inches.  The height is automatically
            calculated based on Bounds.

        dpi : int
            The dots per inch.  Increase to increase the resolution and size of
            resulting image file.

        boundary_points : bool
            If true, this sets a larger boundary (in ExpandedBounds), which
            included the boundary control points.  Default is False.

        ExpandedBounds : list
            A larger boundary that includes the control points.

        Returns
        -------
        fig : matplotlib fig object
            Not currently used, but might be used in the future to
            add another subplot

        ax : matplotlib axis object
            This is used to add features to the current plot
        """
        szx = PP.FigureSizeX
        szy = szx
        if PP.boundary_points and PP.ExpandedBounds is not None:
            szy = (szx*(PP.ExpandedBounds[1][1] - PP.ExpandedBounds[0][1]) /
                   (PP.ExpandedBounds[1][0] - PP.ExpandedBounds[0][0]))
            szy += 1.0/PP.dpi*(int(szy*PP.dpi) % 2)  # szy*dpi must be even
        elif not PP.boundary_points and PP.Bounds is not None:
            szy = (szx*(PP.Bounds[1][1] - PP.Bounds[0][1]) /
                   (PP.Bounds[1][0] - PP.Bounds[0][0]))
            szy += 1.0/PP.dpi*(int(szy*PP.dpi) % 2)  # szy*dpi must be even
        fig = plt.figure(figsize=(szx, szy), dpi=PP.dpi, frameon=False)
        ax = fig.gca()
        rcParams['savefig.pad_inches'] = 0
        # to speed up plotting ... set smaller if needing higher quality
        # rcParams['path.simplify_threshold'] = 1.0
        ax.autoscale(tight=True)
        if PP.boundary_points and PP.ExpandedBounds is not None:
            ax.set_xlim((PP.ExpandedBounds[0][0], PP.ExpandedBounds[1][0]))
            ax.set_ylim((PP.ExpandedBounds[0][1], PP.ExpandedBounds[1][1]))
        elif not PP.boundary_points and PP.Bounds is not None:
            ax.set_xlim((PP.Bounds[0][0], PP.Bounds[1][0]))
            ax.set_ylim((PP.Bounds[0][1], PP.Bounds[1][1]))
        ax.set_aspect('equal')
        ax.tick_params(axis='x', which='both', bottom=False, top=False,
                       labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False,
                       labelleft=False)
        fig.tight_layout(pad=0)
        bbox = ax.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted())
        xlim = ax.get_xlim()
        PP._conversion_factor = (xlim[1]-xlim[0])/bbox.width/72
        return fig, ax

    def _TriangulationPlotBase(self, ax, PP: PlotParameters):
        """Plot the triangulation.

        Plots the underlying triangulation

        Parameters
        ----------
        ax : matplotlib axis object
            The figure axis to add elements to

        PP : PlotParameters object
            For this method, the relevant PlotParameters attributes are:

        linewidth_tri : float
            The line width of the background triangulation.

        linecolor_tri : str
            The color of the triangulation lines. Default is 'g' (green).
        """
        if not PP.boundary_points:
            xpoints = [x[0] for x in self.pointpos[:len(self.pointpos) -
                       self._extranum]]
            #  note that we exclude the bounding points
            ypoints = [x[1] for x in self.pointpos[:len(self.pointpos) -
                       self._extranum]]
            #  make sure that the list of triangles (triplets of points) do
            #  not include the excluded boundary
            triangles = [x.points for x in self.simplist if
                         (len(set(x.points).intersection(
                             [(len(self.pointpos)-y) for y in
                              range(1, self._extranum+1)])) == 0)]
            ax.triplot(xpoints, ypoints, triangles, c=PP.linecolor_tri,
                       lw=PP.linewidth_tri, zorder=1)
        else:
            xpoints = [x[0] for x in self.pointpos]
            ypoints = [x[1] for x in self.pointpos]
            triangles = [x.points for x in self.simplist]
            ax.triplot(xpoints, ypoints, triangles, c=PP.linecolor_tri,
                       lw=PP.linewidth_tri, zorder=1)

    def _PointPlotBase(self, ax, PP: PlotParameters):
        """Plot the points.

        Plots the data points.

        Parameters
        ----------
        ax : matplotlib axis object
            The figure axis to add elements to

        PP : PlotParameters object
            For this method, the relevant PlotParameters attributes are:

        ptlabels : bool
            If True, the integer label for each point is plotted next to the
            point. False is default.  Mainly used for visually finding groups
            of points to encircle with a band.

        markersize : float
            Sets the markersize of the points.
        """
        ptlen = len(self.pointpos)
        if not PP.boundary_points:
            ptlen -= self._extranum
        xpoints = [x[0] for x in self.pointpos[:ptlen]]
        ypoints = [x[1] for x in self.pointpos[:ptlen]]
        if PP.ptlabels:
            for k in range(len(xpoints)):
                ax.annotate(k, (xpoints[k], ypoints[k]))
        ax.scatter(xpoints, ypoints, marker='o', s=PP.markersize,
                   c='k', zorder=2)

    def _TTPlotBase(self, ax, LoopIn, PP: PlotParameters):
        """Plot the loop.

        Plots the train-track representation of the loop

        Parameters
        ----------
        ax : matplotlib axis object
            The figure axis to add elements to

        LoopIn : Loop object
            The data in LoopIn determine the train-track

        PP : PlotParameters object
            For this method, the relevant PlotParameters attributes are:

        Delaunay : bool
            Flag - if True then uses Voronoi-based control points to draw the
            train-track representation of the loops.  If False (default), then
            triangle centers are used as control points.

        DelaunayAdd : bool
            Flag - A different Voronoi-based control point plotting system for
            the train-tracks.  This represents the train-track weights as line
            widths, which join naturally at train-track switch locations. This
            is only relevant if Delaunay is True.

        color_weights : bool
            If True, then the individual segments of the train-track will be
            colored based on their weights.  This is one way to encode weight
            information in the plots.  Default is False.

        log_color : bool
            If True these colors will be assigned using the log of the
            weights. If False (default), the weights them-selves will
            determine the color scale.

        color_map : str
            The color map to be used, default is 'inferno_r'.

        linewidth_tt : float
            The line width of the train-track.  If DelaunayAdd is True, then
            this is the maximum line-width

        linecolor_tt : str
            The line color of the train-track. Default is 'r' (red).

        alpha_tt : float
            The opacity of the train-track.  Default is 1.0 (completely
            opaque/not transparent).

        frac : float
            For plotting with the Delaunay flag, this determined how curved
            the train-tracks appear.  A value of 1.0 is maximally curvy (no
            linear segments), while a value of 0.0 would be just straight
            lines on following the Voronoi skeleton.  Default is 0.9

        tt_lw_min_frac : float
            The minimum fraction of linewidth_tt that will be represented.
            This means that all train-track segments with weights below this
            fraction of the maximum weight will be represented as this
            fraction of linewidth_tt.  All segments with larger weight will
            have a line width that linear in this range.
        """
        #  keeps track of segments that have been plotted
        #  (so as to not plot an element twice)
        EdgePlotted = [False for i in range(self._totalnumedges)]
        ttpatches, cweights, line_widths = [], [], []
        if not PP.Delaunay:   # regular case, works for any triangulation
            for simp in self.simplist:
                if not simp.IsBoundarySimp:
                    new_ttpatches, new_cweights = self._GeneralSimplexTTPlot(
                        simp, LoopIn, EdgePlotted, PP)
                    ttpatches += new_ttpatches
                    if PP.color_weights:
                        cweights += new_cweights
        else:  # looks nicer, but only works for a Delaunay triangulation
            if not PP.DelaunayAdd:
                for simp in self.simplist:
                    if not simp.IsBoundarySimp:
                        new_ttpatches, new_cweights = (
                            self._DelaunaySimplexTTPlot(simp, LoopIn,
                                                        EdgePlotted, PP))
                        ttpatches += new_ttpatches
                        if PP.color_weights:
                            cweights += new_cweights
            else:
                PP._max_weight = max(LoopIn.weightlist)
                for simp in self.simplist:
                    if not simp.IsBoundarySimp:
                        new_ttpatches, new_cweights, new_l_widths = (
                            self._DelaunaySimplexTTPlot_exp(simp, LoopIn, PP))
                        ttpatches += new_ttpatches
                        line_widths += new_l_widths
                        if PP.color_weights:
                            cweights += new_cweights
        Pcollection = PatchCollection(ttpatches, fc="none", alpha=PP.alpha_tt,
                                      capstyle='butt', joinstyle='round',
                                      zorder=3)
        if not PP.DelaunayAdd:
            Pcollection.set_linewidth(PP.linewidth_tt)
        else:
            Pcollection.set_linewidth(line_widths)
        if not PP.color_weights:
            Pcollection.set_edgecolor(PP.linecolor_tt)
        else:
            if PP.log_color:
                Pcollection.set_array(np.log(cweights))
            else:
                Pcollection.set_array(cweights)
            Pcollection.set_cmap(PP.color_map)
        ax.add_collection(Pcollection)

    def _GeneralSimplexTTPlot(self, simp, LoopIn, EdgePlotted,
                              PP: PlotParameters):
        #  plot the segments of train tracks that are determined from a
        #  given simplex
        patches_out, weights_out = [], []
        W = [LoopIn.weightlist[eid] for eid in simp.edgeids]  # edge weights
        delta = 1e-10
        if sum(W) > delta:  # if there are any weights to plot
            #  locations of the three simplex vertices
            vertpts = np.array([self.pointpos[pts] for pts in simp.points])
            #  local id of the extra point in each of the 3 surrounding
            #  simplices
            exlids = [simp.simplices[i].edgeids.index(simp.edgeids[i])
                      for i in range(3)]
            #  locations of the extra point in each of the 3 surrounding
            #  simplices
            exvertpts = np.array(
                [self.pointpos[simp.simplices[i].points[exlids[i]]]
                 for i in range(3)])
            #  now let's get the simplex geometric centers and
            #  edge halfwaypoints
            SimpCenter = HF.GetCenter(vertpts.tolist())
            AdjSimpCenters = [HF.GetCenter(
                [vertpts[(1+i) % 3, :], exvertpts[i, :],
                 vertpts[(2+i) % 3, :]]) for i in range(3)]
            EdgeHalf = np.array(
                [HF.GetCenter([vertpts[(1+i) % 3], vertpts[(2+i) % 3]])
                 for i in range(3)])
            #  now the points that are halfway between the edge
            #  centers and the simpcenter
            CenterEdgeHalf = np.array(
                [HF.GetCenter([SimpCenter, EdgeHalf[i, :]]) for i in range(3)])
            #  now the points that are halfway between the edge centers
            #  and the adjacent simplex centers
            AdjEdgeHalf = np.array(
                [HF.GetCenter([AdjSimpCenters[i], EdgeHalf[i]])
                 for i in range(3)])
            #  check that the quadratic Bezier control triangle doesn't
            #  contain a vertex.
            #  If so, we modify the control points
            for i in range(3):
                side = 2  # default is left
                C1 = HF.Curl(AdjEdgeHalf[i, :] - EdgeHalf[i, :],
                             EdgeHalf[i, :] - CenterEdgeHalf[i, :])
                if C1 > 0:
                    side = 1  # right
                Line1 = [CenterEdgeHalf[i, :], AdjEdgeHalf[i, :]]
                Line2 = [vertpts[(i+side) % 3, :], EdgeHalf[i, :]]
                t1, t2 = HF.GetIntersectionTimes(Line1, Line2)
                if t2 < 0:  # need to modify
                    alpha = -t2/(1-t2)
                    CenterEdgeHalf[i, :] = ((1-alpha)*CenterEdgeHalf[i, :]
                                            + alpha*EdgeHalf[i, :])
                    AdjEdgeHalf[i, :] = ((1-alpha)*AdjEdgeHalf[i, :]
                                         + alpha*EdgeHalf[i, :])
            #  the interior weights
            Wp = [(W[(k+1) % 3] + W[(k+2) % 3] - W[k]) / 2 for k in range(3)]
            for i in range(3):
                if not EdgePlotted[simp.edgeids[i]]:
                    if W[i] > delta:
                        patches_out.append(
                            HF.BezierQuad(CenterEdgeHalf[i, :], EdgeHalf[i, :],
                                          AdjEdgeHalf[i, :]))
                        if PP.color_weights:
                            weights_out.append(W[i])
                    EdgePlotted[simp.edgeids[i]] = True
                if Wp[i] > delta:
                    patches_out.append(
                        HF.BezierQuad(CenterEdgeHalf[(i+1) % 3, :],
                                      SimpCenter,
                                      CenterEdgeHalf[(i+2) % 3, :]))
                    if PP.color_weights:
                        weights_out.append(Wp[i])
        return patches_out, weights_out

    def _DelaunaySimplexTTPlot(self, simp, LoopIn, EdgePlotted, PP):
        #  used in other function to plot the segments of train tracks that
        #  are determined from a given simplex this version assumes the
        #  triangulation is Delaunay, and uses the dual Voroni Centers as
        #  control points
        patches_out, weights_out = [], []
        W = [LoopIn.weightlist[eid] for eid in simp.edgeids]  # edge weights
        delta = 1e-10
        if sum(W) > delta:  # if there are any weights to plot
            #  locations of the three simplex vertices
            vertpts = np.array([self.pointpos[pts] for pts in simp.points])
            #  local id of the extra point in each of the 3
            #  surrounding simplices
            exlids = [simp.simplices[i].edgeids.index(simp.edgeids[i])
                      for i in range(3)]
            #  locations of the extra point in each of the 3
            #  surrounding simplices
            exvertpts = np.array(
                [self.pointpos[simp.simplices[i].points[exlids[i]]]
                 for i in range(3)])
            #  now let's get the simplex Voronoi centers and halfwaypoints
            SimpVCenter = HF.GetCircumCircleCenter(vertpts.tolist())
            AdjSimpVCenters = [
                HF.GetCircumCircleCenter([vertpts[(1+i) % 3, :],
                                          exvertpts[i, :],
                                          vertpts[(2+i) % 3, :]])
                for i in range(3)]
            #  halfway between Voronoi centers
            HalfVCs = [HF.GetCenter([SimpVCenter, AdjSimpVCenters[i]])
                       for i in range(3)]
            #  now the points that partway (frac - default = 0.5) from
            #  Center voroni to HalfVCs
            FracControlPts_In = np.array(
                [HF.LinFuncInterp(SimpVCenter, HalfVCs[i], PP.frac)
                 for i in range(3)])
            FracControlPts_Out = np.array(
                [HF.LinFuncInterp(AdjSimpVCenters[i], HalfVCs[i], PP.frac)
                 for i in range(3)])
            #  the interior weights
            Wp = [(W[(k+1) % 3]+W[(k+2) % 3] - W[k]) / 2 for k in range(3)]
            for i in range(3):
                if not EdgePlotted[simp.edgeids[i]]:
                    if W[i] > delta:
                        patches_out.append(
                            HF.BezierLinear(FracControlPts_In[i, :],
                                            FracControlPts_Out[i, :]))
                        if PP.color_weights:
                            weights_out.append(W[i])
                    EdgePlotted[simp.edgeids[i]] = True
                if Wp[i] > delta:
                    patches_out.append(
                        HF.BezierQuad(FracControlPts_In[(i+1) % 3, :],
                                      SimpVCenter,
                                      FracControlPts_In[(i+2) % 3, :]))
                    if PP.color_weights:
                        weights_out.append(Wp[i])
        return patches_out, weights_out

    def _DelaunaySimplexTTPlot_exp(self, simp, LoopIn, PP):
        #  used in other function to plot the segments of train tracks that
        #  are determined from a given simplex this version assumes the
        #  triangulation is Delaunay, and uses the dual Voroni Centers as
        #  control points this is an experimental version, where I work on
        #  ideas before incorporating them into the main plotting
        patches_out, weights_out, line_weights_out = [], [], []
        W = [LoopIn.weightlist[eid] for eid in simp.edgeids]  # edge weights
        delta = 1e-10
        if sum(W) > delta:  # if there are any weights to plot
            #  locations of the three simplex vertices
            vertpts = np.array([self.pointpos[pts] for pts in simp.points])
            #  local id of the extra point in each of the 3
            #  surrounding simplices
            exlids = [simp.simplices[i].edgeids.index(simp.edgeids[i])
                      for i in range(3)]
            #  locations of the extra point in each of the 3
            #  surrounding simplices
            exvertpts = np.array(
                [self.pointpos[simp.simplices[i].points[exlids[i]]]
                 for i in range(3)])
            #  now let's get the simplex Voronoi centers and halfwaypoints
            SimpVCenter = HF.GetCircumCircleCenter(vertpts.tolist())
            AdjSimpVCenters = [
                HF.GetCircumCircleCenter([vertpts[(1+i) % 3, :],
                                          exvertpts[i, :],
                                          vertpts[(2+i) % 3, :]])
                for i in range(3)]
            #  halfway between Voronoi centers
            HalfVCs = [HF.GetCenter([SimpVCenter, AdjSimpVCenters[i]])
                       for i in range(3)]
            #  now the points that partway (frac - default = 0.5)
            #  from Center voroni to HalfVCs
            FracControlPts_In = np.array(
                [HF.LinFuncInterp(SimpVCenter, HalfVCs[i], PP.frac)
                 for i in range(3)])
            # FracControlPts_Out = np.array(
            #    [HF.LinFuncInterp(AdjSimpVCenters[i], HalfVCs[i], PP.frac)
            #     for i in range(3)])
            # the interior weights
            Wp = [(W[(k+1) % 3] + W[(k+2) % 3] - W[k]) / 2 for k in range(3)]
            W_scaled = []
            Wp_scaled = []
            for i in range(3):
                if W[i] <= PP._max_weight*PP.tt_lw_min_frac:
                    W_scaled.append(PP.linewidth_tt*PP.tt_lw_min_frac)
                else:
                    W_scaled.append(PP.linewidth_tt*(W[i]/PP._max_weight))
                if Wp[i] <= PP._max_weight*PP.tt_lw_min_frac:
                    Wp_scaled.append(PP.linewidth_tt*PP.tt_lw_min_frac)
                else:
                    Wp_scaled.append(PP.linewidth_tt*(Wp[i]/PP._max_weight))
            #  now find the modified control points
            rmp90 = np.array([[0, -1], [1, 0]])
            rmm90 = np.array([[0, 1], [-1, 0]])
            FCP_m_center = FracControlPts_In - np.array(SimpVCenter)
            FCP_m_center_mag = np.hypot(FCP_m_center[:, 0], FCP_m_center[:, 1])
            displace_r = np.array(
                [(W_scaled[i] - Wp_scaled[(i+1) % 3]) / 2 *
                 PP._conversion_factor for i in range(3)])
            displace_l = np.array(
                [(W_scaled[i] - Wp_scaled[(i+2) % 3]) / 2 *
                 PP._conversion_factor for i in range(3)])
            FCP_m_center_rotp = np.array([np.dot(rmp90, FCP_m_center[i, :])
                                          for i in range(3)])
            FCP_m_center_rotm = np.array([np.dot(rmm90, FCP_m_center[i, :])
                                          for i in range(3)])
            scaling_l = displace_l/FCP_m_center_mag
            scaling_r = displace_r/FCP_m_center_mag
            delta_vec_l = np.array([FCP_m_center_rotp[i]*scaling_l[i]
                                    for i in range(3)])
            delta_vec_r = np.array([FCP_m_center_rotm[i]*scaling_r[i]
                                    for i in range(3)])
            FCP_mod_l = delta_vec_l + FracControlPts_In
            FCP_mod_r = delta_vec_r + FracControlPts_In
            Center_mod_l = delta_vec_l + np.array(SimpVCenter)
            Center_mod_r = delta_vec_r + np.array(SimpVCenter)
            HalfVCs_mod_l = delta_vec_l + np.array(HalfVCs)
            HalfVCs_mod_r = delta_vec_r + np.array(HalfVCs)
            center_m = np.array(
                [HF.GetIntersectionPoint(
                    [FCP_mod_r[(i+2) % 3], Center_mod_r[(i+2) % 3]],
                    [FCP_mod_l[(i+1) % 3], Center_mod_l[(i+1) % 3]])
                    for i in range(3)])
            control_points = np.array(
                [[HalfVCs_mod_r[(i+2) % 3], FCP_mod_r[(i+2) % 3], center_m[i],
                  FCP_mod_l[(i+1) % 3], HalfVCs_mod_l[(i+1) % 3]]
                 for i in range(3)])
            for i in range(3):
                if Wp[i] > delta:
                    patches_out.append(
                        HF.BezierCustom(
                            control_points[i, 0, :], control_points[i, 1, :],
                            control_points[i, 2, :], control_points[i, 3, :],
                            control_points[i, 4, :]))
                    if PP.color_weights:
                        weights_out.append(Wp[i])
                    line_weights_out.append(Wp_scaled[i])
        return patches_out, weights_out, line_weights_out

    def TriCopy(self, EvolutionReset=True):
        """Create a copy of this triangulation2D object.

        Custom, as a deepcopy is not sufficient.

        Parameters
        ----------
        EvolutionReset : bool
            If True (default), then the WeightOperatorList is reset to
            be an empty list.  i.e. the memory of any past evolution
            is ignored.

        Returns
        -------
        triangulation2D object
            Returns a copy of this triangulation2D object
        """
        #  create an empty triangulation object (to be returned at the end)
        TriC = triangulation2D([], None, empty=True)
        if not EvolutionReset:
            TriC._atstep = self._atstep
        TriC._extranum = self._extranum
        TriC.Domain = self.Domain
        TriC._ptnum = self._ptnum
        TriC._extrapoints = copy.deepcopy(self._extrapoints)
        TriC.pointpos = copy.deepcopy(self.pointpos)
        TriC._pointposfuture = copy.deepcopy(self._pointposfuture)
        TriC._Vec = self._Vec
        for i in range(len(self.simplist)):
            TriC.simplist.append(simplex2D(self.simplist[i].points))
            TriC.simplist[-1].edgeids = copy.copy(self.simplist[i].edgeids)
            TriC.simplist[-1].SLindex = i
            TriC.simplist[-1].IsBoundarySimp = self.simplist[i].IsBoundarySimp
        #  now create the links
        for i in range(len(self.simplist)):
            for j in range(3):
                if not self.simplist[i].simplices[j] is None:
                    TriC.simplist[i].simplices[j] = TriC.simplist[
                        self.simplist[i].simplices[j].SLindex]
        #  now fill the pointlist
        TriC.pointlist = []
        for i in range(len(self.pointlist)):
            TriC.pointlist.append(TriC.simplist[self.pointlist[i].SLindex])
        TriC._totalnumedges = self._totalnumedges
        if not EvolutionReset:
            for i in range(len(self.WeightOperatorList)):
                TriC.WeightOperatorList.append(
                    WeightOperator(copy.copy(self.WeightOperatorList[i].es)))
        return TriC
