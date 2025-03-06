from abc import ABC, abstractmethod
import copy
import math
import HelperFns as HF
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter
from dataclasses import dataclass

# simplex2D_Base class #######################################################
class simplex2D_Base(ABC):
    """Class representing a triangle / 2D simplex 
        (used in a 2D triangulation object)

    Attributes
    ----------
    _count : int
        a class-level count of the number of simpices created

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

    Methods
    -------
    LocalID(IDin)
        Given the ID (IDin) of a vertex/point in this triangle, this returns 
        the local ID, i.e. the integer i = 0,1,2 s.t. self.points[i] = IDin

    SimpNeighbors(IDin)
        Find the simplices about a given point

    SimpLink(S_other)
        Link self with S_other simplex
    """
    _count = 0
    
    def __init__(self, IDlist):  
        simplex2D_Base._count += 1
        self.points = []
        for i in range(0,len(IDlist)):
            self.points.append(IDlist[i])
        self.simplices = [None,None,None]
        self.edgeids = [None,None,None]
        self.SLindex = None
    
    def __del__(self):
        simplex2D_Base._count -= 1
    
    def __eq__(self, other):
        return self is other  # all comparisions by object id, not value.
  
    def LocalID(self, IDin):
        """Returns the local id of a given point id

        Parameters
        ----------
        IDin : int
            The ID of a vertex/point in this simplex.

        Returns
        -------
        int
            the local ID corresponding to IDin. i.e. the integer i = 0,1,2 
            s.t. self.points[i] = IDin
        """
        try:
            return self.points.index(IDin)
        except Exception as e:
            print(e)
            return None

    @abstractmethod
    def SimpNeighbors(self, IDin):
        """Finds the simpices which share a point. Abstract method"""
        pass

    @abstractmethod
    def EdgeNeighbors(self, IDin):
        """Finds the edges which share a point. Abstract method"""
        pass

    @abstractmethod
    def SimpLink(self,S_other):
        """Links this simplex with S_other (and vice versa)"""
        pass
    
# End simplex2D_Base class ################################################### 


# Loop Class #################################################################
class Loop:
    """Class representing a topological loop or set of loops.  The coordinate 
    system (basis) for this representation is fixed by a particular 
    triangulation2D object.

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
    def __init__(self, tri, rbands = None, curves = None, 
                 Shear = False, mesh = False):
        """
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
        self.weightlist = [0 for i in range(tri.totalnumedges)]
        self.Shear = Shear
        if mesh:
            for i in range(len(self.weightlist)):
                self.weightlist[i] = -1.0
                # represents bands pinned to adjacent 
                # points in this triangulation.
            self.Shear = True # mesh must be evaluated with shear coordinates
        else:
            if not self.Shear:
                if rbands is not None:
                    tri.BandWeightInitialize(rbands, LoopIn = self)
                if curves is not None:
                    tri.CurveWeightInitialize(curves, LoopIn = self)
            else:
                RegLoop = Loop(tri)
                if rbands is not None:
                    tri.BandWeightInitialize(rbands, LoopIn = RegLoop)
                if curves is not None:
                    tri.CurveWeightInitialize(curves, LoopIn = RegLoop)
                #  This first creates a regular loop (regular coordinates), 
                #  then feeds this into the triangulation object to get the
                #  shear coordinates
                tri.ShearWeightInitialize(RegLoop, self)
                       
    def GetWeightTotal(self):
        if not self.Shear:
            return sum(self.weightlist)
        else:
            WT = 0
            for i in range(len(self.weightlist)):
                WT += abs(self.weightlist[i])
            return WT

    def ProjectivizeWeights(self):
        mwv = max(max(self.weightlist),abs(min(self.weightlist)))
        self.weightlist = [x/mwv for x in self.weightlist]

    def LoopCopy(self):
        return copy.deepcopy(self)
# End of Loop Class ##########################################################


# WeightOperator Class #######################################################
class WeightOperator:
    """Class representing an operator that acts on loops.  It is generated
        every time a triangulation flip occurs during the evolution of a 
        triangulation object, and holds the information needed to update the
        weightlist of a loop.

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
    def __init__(self, IndexSet, TimeIn = None):
        """
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
    
    #This updates the given weightlist
    def Update(self, LoopIn, Reverse = False):
        """
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
        else:  # For Shear weights, the surrounding quadrilateral weights 
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

# PrintParameters Class ######################################################
@dataclass
class PrintParameters:
    """Class containing all of the parameters used in printing the 
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
        (Not used in the periodic boundary version).  If true, this sets a 
        larger boundary (calculated automatically) which included the boundary
        control points.  Default is False.
    """
    #main flags/choices
    filename: str = None
    triplot: bool = True
    Delaunay: bool = False
    DelaunayAdd: bool = False
    #initial setup
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
    #train track specifications
    linewidth_tt: float = 1.0
    linecolor_tt: str = 'r'
    alpha_tt: float = 1.0
    #Delaunay
    frac: float = 0.9
    #DelaunayAdd
    tt_lw_min_frac: float = 0.05
    conversion_factor: float = None #internal only
    max_weight: int = None #internal only


# triangulation2D_Base Class #################################################

class triangulation2D_Base(ABC):
    """The central class in the overal Topological Advection algorithm, this
    class represents a triangulation of data points in a 2D domain.  It has
    methods for evolving the triangulation due to the motion of data points,
    acting as a basis for encoding loops, accumulating weight operators, and
    plotting.  This abstract base class then has different child classes for
    different situations (doubly periodic boundary conditions, given 
    boundary, etc.).

    Note: Only the main attribues and methods are listed here.
    Note: The triangulation is initialized as a Delaunay triangulation.

    Attributes:
    -----------
    
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

    
    Methods:
    --------

    Evolve(ptlist, Maintain_Delaunay = False)
        This evolves the triangulation forward due to the motion of the points
        - new point positions in ptlist. Options for evolution via collapse 
        events or to maintain a Delaunay triangulation.  For every edge flip
        needed, a WeightOperator is added to the WeightOperator list.

    OperatorAction(LoopIn, index = None, Reverse = False, option = 3)
        This evolves forward an individual loop object (i.e. updates its 
        weightlist due to the action of the WeightOperators in 
        WeightOperatorList).

    BandWeightInitialize(rbands, LoopIn)
        This initializes the loop weights in LoopIn using the set of bands in
        rbands. A band represent a closed curve topologically (with a set of 
        point ids).

    CurveWeightInitialize(curves, LoopIn)
        This initializes the loop weights in LoopIn using the set of curves in
        curves.  A curve represents a geometric curve (with a set of [x,y] 
        positions).

    ShearWeightInitialize(RegLoop, LoopIn):
        This modifies the weightlist of LoopIn to represent the loop given in
        RegLoop, but in shear coordinates.

    Plot(LoopIn = None, PP: PrintParameters = PrintParameters())
        This plots the triangulation and loop.  See PrintParameters data class
        documentation for details on the many options.
    """

    def __init__(self, ptlist, empty = False):
        """
        Parameters
        ----------
        ptlist : list
            ptlist is the list of [x,y] positions for the points at the 
            initial time.

        empty : bool
            Used for creating an empty object, which is then used for object 
            copying.  Defaul is False
        """
        
        self.atstep = 0
        self.pointlist = None
        self.pointpos = None
        self.pointposfuture = None
        self.simplist = []
        self.totalnumedges = 0
        self.WeightOperatorList = []
        self.Vec = True
        # for small number of points, the non-vectorized version
        # of a few functions will be faster
        if len(ptlist) < 10:
            self.Vec = False
        if not empty:
            self.LoadPos(ptlist)
            self.SetInitialTriangulation()

    @abstractmethod
    def LoadPos(self, ptlist):
        pass

    @abstractmethod
    def SetInitialTriangulation(self):
        pass

    def Evolve(self, ptlist, Maintain_Delaunay = False):
        """
        Evolve. Main method for evolving the state of the triangulation 
        forward in time.  This assumes that the starting triangulation is good
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
        self.LoadNewPos(ptlist)
        EventLists = self.GetEvents() 
        #  GEvolve deals with the events in CollapseEventList 
        #  and CrossingList (if periodic boundaries) in order
        self.GEvolve(EventLists)
        self.UpdatePtPos()
        self.atstep += 1
        if Maintain_Delaunay:
            self.MakeDelaunay()
            #  after the atstep increment so that the operators
            #  will have the correct time-stamp.

    @abstractmethod
    def LoadNewPos(self, ptlist):
        pass

    @abstractmethod
    def GetEvents(self):
        pass

    def GetCollapseEvents(self):
        """
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
        if self.Vec:  
            AZT_bool, AZT_time = self.AreaZeroTimeMultiple()
            collapsesimplist = [[self.simplist[i],AZT_time[i]] 
                                for i in range(len(self.simplist)) 
                                if AZT_bool[i]]    
        else:
            for simp in self.simplist:
                AZT = self.AreaZeroTimeSingle(simp)
                if AZT[0]:
                    collapsesimplist.append([simp,AZT[1]])  
        collapsesimplist.sort(key=itemgetter(1), reverse=True)
        return collapsesimplist

    @abstractmethod
    def AreaZeroTimeMultiple(self, Tin = 0):
        """
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
        pass

    @abstractmethod
    def AreaZeroTimeSingle(self, simp, Tin = 0):
        """
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
        pass

    def UpdatePtPos(self):
        self.pointpos = self.pointposfuture

    @abstractmethod
    def GEvolve(self, EventLists):
        """
        Processes an ordered list of events (collapse, and crossing - if 
        periodic boundary conditions) and does edge flips to update the 
        triangulation.  Also adds in new events as needed. Finished when 
        there are no more events in the time interval, and the triangulation
        is consistent with the new set of points.
        """
        pass

    def SFix(self, SimpIn, tcollapse):
        #  Fixing a simplex and the surrounding affected simplices. This returns the two 
        #  new simplices, so that they can be possibly added to the local event 
        #  list, also the bad simplex so it can be removed (if needed from the 
        #  local event list)
        
        #  `colind` is the local index of the offending point during the 
        #  area collapse
        colind = self.CollapsePt(SimpIn, tcollapse) 
        Topsimp = SimpIn.simplices[colind]
        edge_id = SimpIn.edgeids[colind]
        globaltime = self.atstep + tcollapse
        newsimps = self.EdgeFlip([SimpIn,Topsimp], edge_id, globaltime)  
        #  EdgeFlip does most of the work in flipping the edge and 
        #  cleaning up linking
        return [newsimps,Topsimp]
        #  return the two new simplices, so that they can be checked to see 
        #  if they need to be included in any update to the local event list.
        #  Also return the bad simplex to remove any instance from the 
        #  event list.
        
    def CollapsePt(self, SimpIn, tcol):
        #  This returns the point (internal id) that passes through its 
        #  opposite edge during an area collapse event known to occur 
        #  at t = tcol
        #  first get the positions of the 3 points at the time of collapse
        colpos = self.GetSimpCurrentLoc(SimpIn, tcol)
        d0 = ((colpos[2][0] - colpos[0][0])*(colpos[1][0] - colpos[0][0]) +
              (colpos[2][1] - colpos[0][1])*(colpos[1][1] - colpos[0][1]))
        #  This is the dot product of (z2-z0) and (z1-z0)
        if d0 < 0:  return 0
        else:
            d1 = ((colpos[2][0] - colpos[1][0])*(colpos[0][0]-colpos[1][0]) + 
                  (colpos[2][1] - colpos[1][1])*(colpos[0][1] - colpos[1][1]))
            if d1 < 0:  return 1
            else:  return 2  
            #  Note: don't need to calculate the last dot product.  
            #  If the first two are >0, this must be <0
    
    @abstractmethod
    def GetSimpCurrentLoc(self, SimpIn, tcol):
        """
        this returns the linearly interpolated positions of the three points
        in SimpIn at time tcol
        """
        pass

    @abstractmethod
    def EdgeFlip(self, AdjSimps, EdgeShare, TimeIn = None):
        """
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
        pass
    
    @abstractmethod
    def MakeDelaunay(self):
        """
        MakeDelaunay takes the current triangulation and, through a series
        of edge flips, changes it into the Delaunay triangulation for this 
        point configuration.  This function changes the underlying 
        triangulation
        """
        pass
    
    def OperatorAction(self, LoopIn, index = None, Reverse = False, 
                       option = 3, num_times = None):
        """
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
                for i in range(startind,endind+1):
                    self.WeightOperatorList[i].Update(LoopIn)
            else:
                for i in range(endind,startind-1,-1):
                    self.WeightOperatorList[i].Update(LoopIn, Reverse)    
        elif option == 2:
            WeightList = []
            if not Reverse:
                WeightList.append([LoopIn.GetWeightTotal(),0])
                for i in range(startind,endind+1):
                    self.WeightOperatorList[i].Update(LoopIn)
                    WeightList.append([LoopIn.GetWeightTotal(), 
                                       self.WeightOperatorList[i].time])
            else:
                finaltime = math.ceil(self.WeightOperatorList[i].time)
                WeightList.append([LoopIn.GetWeightTotal(), finaltime])
                for i in range(endind,startind-1,-1):
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
                for i in range(len(self.WeightOperatorList)-1,-1,-1):
                    thistime = math.floor(self.WeightOperatorList[i].time)
                    if thistime < prevtime:
                        prevtime = thistime
                        currentweight = LoopIn.GetWeightTotal()
                        while len(WeightList) < endtime-thistime:
                            WeightList.append(currentweight)
                    self.WeightOperatorList[i].Update(LoopIn,Reverse)            
                WeightList.append(LoopIn.GetWeightTotal())
            if num_times is not None:
                while len(WeightList) < num_times:
                    WL_last = WeightList[-1]
                    WeightList.append(WL_last)
            return WeightList
        else:
            print("Need to choose one of the options 1, 2, or 3")

    def BandWeightInitialize(self,rbands, LoopIn):
        """
        This initializes the edge weights in `LoopIn` that correspond to a
        given band (or set of bands) in `rbands`.

        Parameters
        ----------
        rbands : list
            Each element in the list represents a band, and consists of two 
            items: the list of points which define a band (see Loop class 
            documentation), and the weight to add to the loop weightlist.

        LoopIn : Loop object
            The weightlist of LoopIn will be modified to represent this 
            additional set of bands being added in.
        """
        for band_data in rbands:
            band, wadd = band_data
            numpoints = len(band)
            AreAdjacent, CurveLeft = [], []
            for k in range(numpoints):
                AreAdjacent.append(
                    self.ArePointsAdjacent(band[k], band[(k+1)%numpoints]))
                triplepts = [band[(k+numpoints-1)%numpoints], 
                             band[k], band[(k+1)%numpoints]]
                CurveLeft.append(self.DoesCurveLeft(triplepts))
            for j in range(numpoints):
                Bool1 = [CurveLeft[j], AreAdjacent[j], 
                         CurveLeft[(j+1)%numpoints]]
                Bool2 = [AreAdjacent[(j+numpoints-1)%numpoints], 
                         CurveLeft[j], AreAdjacent[j]]
                triplepts = [band[(j+numpoints-1)%numpoints], band[j], 
                             band[(j+1)%numpoints]]                
                self.AddWeightsAlongLine(
                    [band[j],rbands[i][(j+1)%numpoints]], Bool1, LoopIn, wadd)
                self.AddWeightsAroundPoint(triplepts, Bool2, LoopIn, wadd)

    @abstractmethod
    def ArePointsAdjacent(self,pt1,pt2):
        pass

    @abstractmethod
    def DoesCurveLeft(self,pttriple):
        pass
    
    @abstractmethod
    def AddWeightsAlongLine(self,linepoints, Boolin, LoopIn, wadd = 1.0):
        pass

    @abstractmethod
    def SimpInDir(self,linepoints):    
        pass

    @abstractmethod
    def AddWeightsAroundPoint(self, pttriple, Boolin, LoopIn, wadd = 1.0):
        pass

    def CurveWeightInitialize(self, curves, LoopIn):
        """
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
            edge_list = self.Get_Edges(point_set, is_closed, end_pts_pin)
            for edge in edge_list:
                LoopIn.weightlist[edge] += wadd

    @abstractmethod
    def Get_Edges(self, points, closed = True, end_pts_pin = True):
        #  This finds the set of edges (in order) which correspond to a curve.
        pass

    def ShearWeightInitialize(self, RegLoop, LoopIn):
        """
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
                        WA = RegLoop.weightlist[simp.edgeids[(i+1)%3]]
                        WB = RegLoop.weightlist[simp.edgeids[(i+2)%3]]
                        xsimp = simp.simplices[i]
                        Lid = xsimp.LocalID(simp.points[(i+2)%3])
                        WC = RegLoop.weightlist[xsimp.edgeids[Lid]]
                        WD = RegLoop.weightlist[xsimp.edgeids[(Lid+1)%3]]
                        LoopIn.weightlist[simp.edgeids[i]] = (-WA+WB-WC+WD)//2

    def Plot(self, LoopIn = None, PP: PrintParameters = PrintParameters()):
        """
        Plots the points, triangulation, and loops with a large variety
        of options specified in PrintParameters (see the documentation for 
        PrintParameters data class for more details).

        Parameters
        ----------
        LoopIn : Loop object
            If a loop object is passed, then the train-track associated with
            this loop will be included in the plot. Default is None.

        PP : PrintParameters data object
            All of the options for customizing the plot are wrapped up in
            this data object.  The top-level attributes of PP that are used
            in Plot are:

        filename : str
            The filename (including local path) to save the figure as.
            If None (default), then then the figure is printed to screen.

        triplot : bool
            Flag - prints the background triangulation if True (default) and 
            excludes it if False.
        """
        # the preliminary plotting settings
        fig, ax = self.PlotPrelims(PP)
        # the underlying triangulation
        if PP.triplot:  self.TriangulationPlotBase(ax, PP)
        self.PointPlotBase(ax, PP)  # the points
        #only plot the traintrack if a Loop is given
        if LoopIn is not None:  self.TTPlotBase(ax, LoopIn, PP)
        if PP.filename is None:  plt.show()
        else:  plt.savefig(PP.filename)
        plt.close()

    @abstractmethod
    def PlotPrelims(self, PP: PrintParameters):
        """
        Handles the initial setup for the figure

        Parameters
        ----------
        PP : PrintParameters object
            For this method, the relevant PrintParameters attributes are:

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

        Returns
        -------
        fig : matplotlib fig object
            Not currently used, but might be used in the future to
            add another subplot

        ax : matplotlib axis object
            This is used to add features to the current plot
        """
        pass
    
    @abstractmethod
    def TriangulationPlotBase(self, ax, PP: PrintParameters):
        """
        Plots the underlying triangulation

        Parameters
        ----------
        ax : matplotlib axis object
            The figure axis to add elements to

        PP : PrintParameters object
            For this method, the relevant PrintParameters attributes are:

        linewidth_tri : float
            The line width of the background triangulation.

        linecolor_tri : str
            The color of the triangulation lines. Default is 'g' (green).    
        """
        pass
        
    @abstractmethod
    def PointPlotBase(self, ax, PP: PrintParameters):
        """
        Plots the points

        Parameters
        ----------
        ax : matplotlib axis object
            The figure axis to add elements to

        PP : PrintParameters object
            For this method, the relevant PrintParameters attributes are:

        ptlabels : bool
            If True, the integer label for each point is plotted next to the 
            point. False is default.  Mainly used for visually finding groups 
            of points to encircle with a band.

        markersize : float
            Sets the markersize of the points.
        """
        pass

    @abstractmethod
    def TTPlotBase(self, ax, LoopIn, PP: PrintParameters):
        """
        Plots the train-track representation of the loop

        Parameters
        ----------
        ax : matplotlib axis object
            The figure axis to add elements to

        LoopIn : Loop object
            The data in LoopIn determine the train-track

        PP : PrintParameters object
            For this method, the relevant PrintParameters attributes are:

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
        pass

    @abstractmethod
    def TriCopy(self, EvolutionReset = True):
        """
        Creates a copy of this triangulation2D object.  Custom, as a 
        deepcopy is not sufficient.

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
        pass