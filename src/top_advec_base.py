"""The base module for the topological advection algorithm.

This module contains the abstract base classes for the topological advection
algorithm, and is meant to be a template for concrete versions of these
classes.  The topological advection algorithm takes trajectories of point
particles in 2D and determines how this motion affects the state of material
curves in the surrounding medium.  Curves are encoded topologically as 'loops'
with a triangulation of the points acting as a basis for the loops.  As the
points move, the triangulation is updated, and operators which act on loops
are accumulated.

Classes
-------
Simplex2D_Base
    Base class representing a triangle / 2D simplex

Loop_Base
    Base class representing a topological loop or set of loops.

WeightOperator_Base
    Base class representing an operator that acts on loops.

PlotParameters:
    Data class for grouping plot parameters

Triangulation2D_Base
    Base class representing a triangulation of data points in a 2D domain.
    With methods for evolving the triangulation forward due to moving points,
    intializing loops, evolving loops, and plotting.
"""


import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from dataclasses import dataclass


# Simplex2D_Base class #######################################################
class Simplex2D_Base(ABC):
    """Class representing a triangle / 2D simplex.

        (used in a 2D triangulation object)

    Attributes
    ----------
    points : list of 3 ints
        List of the 3 vertex point IDs. These IDs will be used to get
        the vertex spatial locations from a master list that is a
        Triangulation2D attribue. Note that the order is only unique up
        to an even permutation. The permuation given on initialization
        is assumed to correspond to a set of geometric points that are
        given in counter clock-wise, CCW, order. Moving about this list
        (and other Simplex2D attributes) is done via modular
        arithmatic: (i+1)%3 to move CCW about the simplex from index i,
        and (i+2)%3 to move CW.

    simplices : list of 3 simplex objects
        List of the 3 simplices adjacent to this simplex. Convention:
        simplex at position i in this list is adjacent to this simplex
        at the edge across from self.points[i]

    edgeids : list of 3 ints
        List of the 3 edge ids. Each edge has an identifying integer
        that is used when keeping track of loop coordinates (in loop
        class).

    SLindex : int
        ID of this simplex in a Simplex List attribute of
        Triangulation2D (simplist). This is useful for going back and
        forth from the local view (Simplex2D object) and the global
        view (Triangulation2D object)

    Methods
    -------
    LocalID(IDin)
        Given the ID (IDin) of a vertex/point in this triangle, this
        returns the local ID, i.e. the integer i = 0,1,2 s.t.
        self.points[i] = IDin

    SimpNeighbors(IDin)
        Find the simplices about a given point

    EdgeNeighbors(IDin)
        Find the ids of edges about a given point

    SimpLink(S_other)
        Link self with S_other simplex
    """

    _count = 0

    def __init__(self, IDlist):
        Simplex2D_Base._count += 1
        self.points = []
        for i in range(len(IDlist)):
            self.points.append(IDlist[i])
        self.simplices = [None, None, None]
        self.edgeids = [None, None, None]
        self.SLindex = None

    def __del__(self):
        Simplex2D_Base._count -= 1

    def __eq__(self, other):
        return self is other  # all comparisions by object id, not value.

    def LocalID(self, IDin):
        """Return the local id of a given point id.

        Parameters
        ----------
        IDin : int
            The ID of a vertex/point in this simplex.

        Returns
        -------
        int
            the local ID corresponding to IDin. i.e. the integer
            i = 0,1,2  s.t. self.points[i] = IDin
        """
        try:
            return self.points.index(IDin)
        except Exception as e:
            print(e)
            return None

    @abstractmethod
    def SimpNeighbors(self, IDin):
        """Find the simpices which share a point. Abstract method."""
        pass

    @abstractmethod
    def EdgeNeighbors(self, IDin):
        """Find the edges which share a point. Abstract method."""
        pass

    @abstractmethod
    def SimpLink(self, S_other):
        """Links this simplex with S_other (and vice versa)."""
        pass
# End Simplex2D_Base class ###################################################


# Loop Class #################################################################
class Loop_Base(ABC):
    """Class representing a topological loop or set of loops.

    The coordinate system (basis) for this representation is fixed by a
    particular Triangulation2D object.

    Attributes
    ----------
    weightlist : list/dict
        representation of a loop via intersection/shear coordinates.

    Methods
    -------
    GetWeightTotal()
        Return the sum of the intersection coordinates (weights),
        which is a good proxy for the length of the loop.
    """

    @abstractmethod
    def GetWeightTotal(self):
        """Return the sum of the intersection coordinates (weights).

        This is a good proxy for the length of the loop.
        """
        pass

    @abstractmethod
    def LoopCopy(self):
        """Make a copy of the loop and return it."""
        pass
# End of Loop Class ##########################################################


# WeightOperator Class #######################################################
class WeightOperator_Base(ABC):
    """Class representing an operator that acts on loops.

        It is generated every time a triangulation flip occurs during
        the evolution of a triangulation object, and holds the
        information needed to update the weightlist of a loop.

    Methods
    -------
    Update(LoopIn)
        Update the weightlist attribute of LoopIn.
    """

    @abstractmethod
    def Update(self, LoopIn):
        """Update the weightlist attribute of LoopIn.

        Parameters
        ----------
        LoopIn : Loop Object
            This is the loop whose weightlist will be updated
            (in place).
        """
        pass
# End of WeightOperator Class ################################################


# PlotParameters Class ######################################################
@dataclass
class PlotParameters:
    """Data class for grouping the print parameters.

    Contains all of the parameters used in printing the
    triangulation and loops, and their default values.

    Attributes
    ----------
    filename : str
        The filename (including local path) to save the figure as.
        If None (default), then then the figure is printed to screen.

    triplot : bool
        Flag - prints the background triangulation if True (default)
        and excludes it if False.

    Note
    ____
    Many other parameters will be added in the child classes

    """

    filename: str = None
    triplot: bool = True
# End PlotParameters Class ##################################################


# Triangulation2D_Base Class #################################################
class Triangulation2D_Base(ABC):
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
        A list of simplex objects.  Object at index i has the point
        with point id i in its point list.  Allows for O(1) lookup of
        points in the triangulation.  Note, not every simplex is in
        this list.

    pointpos : list
        A list of the [x,y] positions for the points at the current
        time

    pointposfuture : list
        List of the [x,y] positions at the next time step.  Used with
        the Evolution method.

    simplist : list
        List of all of the simplices that make up the triangulation.
        Individual simplices have an id (SLindex) that indicates their
        location in this list.

    WeightOperatorList : list
        List of WeightOperator objects.  As the triangulation is
        evolved forward due to point motions, retriangulations with
        edge flips are needed.  For each flip, we record the data
        needed to evolve a loop forward.  This list is ordered
        (increasing) in time.


    Methods
    -------
    Evolve(ptlist, Maintain_Delaunay = False)
        This evolves the triangulation forward due to the motion of the
        points - new point positions in ptlist. Options for evolution
        via collapse events or to maintain a Delaunay triangulation.
        For every edge flip needed, a WeightOperator is added to the
        WeightOperator list.

    OperatorAction(LoopIn, index = None, Reverse = False, option = 3)
        This evolves forward an individual loop object (i.e. updates
        its weightlist due to the action of the WeightOperators in
        WeightOperatorList).

    Plot(LoopIn = None, PP: PlotParameters = PlotParameters())
        This plots the triangulation and loop.  See PlotParameters
        data class documentation for details on the many options.

    TriCopy(EvolutionReset = True)
        This returns a copy of this Triangulation2D object.
    """

    def __init__(self, ptlist, empty=False):
        """Triangulation Initialization.

        Parameters
        ----------
        ptlist : list
            ptlist is the list of [x,y] positions for the points at the
            initial time.

        empty : bool
            Used for creating an empty object, which is then used for
            object copying.  Default is False
        """
        self._atstep = 0
        self.pointlist = None
        self.pointpos = None
        self._pointposfuture = None
        self.simplist = []
        self._totalnumedges = 0
        self.WeightOperatorList = []
        self._Vec = True
        # for small number of points, the non-vectorized version
        # of a few functions will be faster
        if len(ptlist) < 10:
            self._Vec = False
        if not empty:
            self._LoadPos(ptlist)
            self._SetInitialTriangulation()

    @abstractmethod
    def _LoadPos(self, ptlist):
        pass

    @abstractmethod
    def _SetInitialTriangulation(self):
        pass

    @abstractmethod
    def Evolve(self, ptlist):
        """Evolve the triangulation forward.

        Main method for evolving the state of the triangulation
        forward in time.  This assumes that the starting triangulation
        is good (no negative areas).

        Parameters
        ----------
        ptlist : list
            The new time-slice data; the list of [x,y] positions for
            the points at the next time-step.
        """
        #  Overview: load the new positions, find the events, deal with the
        #  events, update the current position, and maintain delaunay if
        #  needed.
        pass

    @abstractmethod
    def _LoadNewPos(self, ptlist):
        pass

    @abstractmethod
    def _GetEvents(self):
        pass

    @abstractmethod
    def _GetCollapseEvents(self):
        """Find triangle collapse events.

        This finds all of the events where a triangle will go through
        zero area in the course of the points evolving from this time
        to the next time-step.

        Returns
        -------
        list
            A list of the simplices that collapse and the time of their
            collapse (bundled as a list of two items).  This list is
            sorted in decending order so that removing from the end
            (smallest times first) inccurs the smallest computational
            cost.
        """
        pass

    @abstractmethod
    def _AreaZeroTimeMultiple(self, Tin=0):
        """Vectorized calculation of collapse times.

        Goes through every simplex and looks for whether the area zero
        time is between Tin and 1.  Similar to AreaZeroTimeSingle, but
        wrapping up the info in numpy arrays to get vectorization and
        jit boost.

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
    def _AreaZeroTimeSingle(self, simp, Tin=0):
        """Calculate collapse time.

        Finds whether (and when) a triangle (simp) goes through zero
        area.

        Parameters
        ----------
        simp : Simplex2D object
            The simplex to consider.

        Tin : float
            The lower bound on the time window to consider.

        Returns
        -------
        list
            Returns a pair [IsSoln, TimeOut], where IsSoln is a boolean
            that is True if the first time at which the area goes
            through zero is between Tin and 1, and False if not. For
            IsSoln == True, TimeOut gives this time.
        """
        pass

    def _UpdatePtPos(self):
        self.pointpos = self._pointposfuture

    @abstractmethod
    def _GEvolve(self, EventLists):
        """Process the events in EventLists.

        Processes an ordered list of events (collapse, and crossing -
        if periodic boundary conditions) and does edge flips to update
        the triangulation.  Also adds in new events as needed. Finished
        when there are no more events in the time interval, and the
        triangulation is consistent with the new set of points.
        """
        pass

    @abstractmethod
    def _SFix(self, SimpIn, tcollapse):
        #  Fixing a simplex and the surrounding affected simplices.
        #  This returns the two new simplices, so that they can be
        #  possibly added to the local event list, also the bad simplex
        #  so it can be removed (if needed from the local event list)
        pass

    @abstractmethod
    def _CollapsePt(self, SimpIn, tcol):
        #  This returns the point (internal id) that passes through its
        #  opposite edge during an area collapse event known to occur
        #  at t = tcol
        pass

    @abstractmethod
    def _GetSimpCurrentLoc(self, SimpIn, tcol):
        """Get the location of simplex points at given time.

        This returns the linearly interpolated positions of the three
        points in SimpIn at time tcol.
        """
        pass

    @abstractmethod
    def _EdgeFlip(self, AdjSimps, EdgeShare, TimeIn=None):
        """Flip an edge in the triangulation.

        EdgeFlip locally re-triangulates the triangulation by removing
        an edge that divides two adjacent triangles in a quadrilateral,
        and replaces it with the other diagonal of this quadrilateral.
        This removes the old simplices, creates new ones, and links
        them up in the triangulation. EdgeFlip also creates the
        appropriate WeightOperator object and adds it to the
        WeightOperator list.

        Parameters
        ----------
        AdjSimps : list of 2 Simplex2D objects
            These are the two simplices that share the edge to be
            flipped

        EdgeShare : int
            The edge id of the edge to be flipped.  While this can
            almost always be found from AdjSimps, the redundancy helps
            in certain cases.

        TimeIn : float
            This is the time when the event occured that required a
            flip. It is added as part of the data in the WeightOperator
            object.

        Returns
        -------
        list of 2 Simplex2D objects
            The two new simplices.  Returned so that the calling
            function
        """
        pass

    @abstractmethod
    def MakeDelaunay(self):
        """Flip edges until the triangulation is Delaunay.

        MakeDelaunay takes the current triangulation and, through a
        series of edge flips, changes it into the Delaunay
        triangulation for this point configuration.  This function
        changes the underlying triangulation
        """
        pass

    @abstractmethod
    def OperatorAction(self, LoopIn, index=None, Reverse=False, option=3):
        """Flip operator acting on a Loop.

        OperatorAction takes the accumulated operator list stored in
        WeightOperatorList and operates sucessively on the given Loop

        Parameters
        ----------
        LoopIn : Loop Object
            The weightlist of this loop will be modified in place

        index : list of 2 ints
            the start and stop index can also be specified to break
            this up into stages (only used for option 1 and 2). Default
            is None

        Reverse : bool
            Reverse does the operator actions in reverse order (i.e.
            for loops in the final triangulation)

        option : int {1,2,3}
            option 1 just changes the data in the loop object.
            option 2 also accumulates a weight list with the total
            weights after each operator has acted on the loop, and
            gives the global time of the operator action.
            option 3 (the default) returns a weight list which has the
            weights at the end of each time step (the intervals between
            each use of the Evolve method). This weight list does not
            have the time listed, as this is only know externally. this
            is most useful for producing a list that we can directly
            tie to an external list of times. This is what we need for
            extracting the topological entropy (hence the default
            option).
        """
        pass

    @abstractmethod
    def _BandWeightInitialize(self, rbands, LoopIn):
        """Initialize a loop with band data.

        This initializes the edge weights in `LoopIn` that correspond
        to a given band (or set of bands) in `rbands`.

        Parameters
        ----------
        rbands : list
            Each element in the list represents a band, and consists of
            two items: the list of points which define a band (see Loop
            class documentation), and the weight to add to the loop
            weightlist.

        LoopIn : Loop object
            The weightlist of `LoopIn` will be modified to represent
            this additional set of bands being added in.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def Plot(self, LoopIn=None, PP: PlotParameters = PlotParameters()):
        """General Plotting function.

        Plots the points, triangulation, and loops with a large variety
        of options specified in PlotParameters (see the documentation for
        PlotParameters data class for more details).

        Parameters
        ----------
        LoopIn : Loop object
            If a loop object is passed, then the train-track associated with
            this loop will be included in the plot. Default is None.

        PP : PlotParameters data object
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
        #  the preliminary plotting settings
        fig, ax = self._PlotPrelims(PP)
        #  the underlying triangulation
        if PP.triplot:
            self._TriangulationPlotBase(ax, PP)
        #  the points
        self._PointPlotBase(ax, PP)
        #  only plot the traintrack if a Loop is given
        if LoopIn is not None:
            self._TTPlotBase(ax, LoopIn, PP)
        if PP.filename is None:
            plt.show()
        else:
            plt.savefig(PP.filename)
        plt.close()

    @abstractmethod
    def _PlotPrelims(self, PP: PlotParameters):
        """Preliminary setup for plotting.

        Handles the initial setup for the figure.

        Parameters
        ----------
        PP : PlotParameters object

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
    def _TriangulationPlotBase(self, ax, PP: PlotParameters):
        """Plot the triangulation.

        Plots the underlying triangulation

        Parameters
        ----------
        ax : matplotlib axis object
            The figure axis to add elements to

        PP : PlotParameters object
        """
        pass

    @abstractmethod
    def _PointPlotBase(self, ax, PP: PlotParameters):
        """Plot the points.

        Plots the data points.

        Parameters
        ----------
        ax : matplotlib axis object
            The figure axis to add elements to

        PP : PlotParameters object
        """
        pass

    @abstractmethod
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
        """
        pass

    @abstractmethod
    def TriCopy(self, EvolutionReset=True):
        """Create a copy of this Triangulation2D object.

        Custom, as a deepcopy is not sufficient.

        Parameters
        ----------
        EvolutionReset : bool
            If True (default), then the WeightOperatorList is reset to
            be an empty list.  i.e. the memory of any past evolution
            is ignored.

        Returns
        -------
        Triangulation2D object
            Returns a copy of this Triangulation2D object
        """
        pass
