from TopAdvBase import simplex2D_Base, Loop, WeightOperator, triangulation2D_Base, PrintParameters
import HelperFns as HF
import numpy as np
import math
import copy
from operator import itemgetter, attrgetter
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.style as mplstyle
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib import rcParams
from dataclasses import dataclass

# simplex2D class ########################################################
class simplex2D(simplex2D_Base):
    """Class representing a triangle / 2D simplex 
        (periodic boundaries version)
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

    relptregion : list of lists
        relptregion records the relative position of points.  If the edge 
        connecting two points of this simplex crosses the torus boundary, then
        we record the x,y integer displacement (how many copies of the domain 
        over in the x and y direction) of the second point from the 
        perspective of the first (with the first and second point ordering 
        determined by the CCW simplex ordering, and the unique edge indexed as
        usual - i.e. associated with the index of the point it is across from)
        
            Example: in a fundamental domain of Dx=Dy=1, consider the first 
            point at [0.5,0.7] and second point at [0.5,0.1], where the line
            connecting them goes through the top boundary (remember, the 
            periodic boundary conditions). Then the integer pair describing 
            this edge from this triangle would be [0,1], since we have to go
            one copy up in the y direction to get to the second point. This 
            extra bit of information allows us to consistently define 
            distances between points and find triangle areas.

    Methods
    -------
    LocalID(IDin)
        Given the ID (IDin) of a vertex/point in this triangle, this returns 
        the local ID, i.e. the integer i = 0,1,2 s.t. self.points[i] = IDin

    SimpNeighbors(IDin)
        Given the ID (IDin) of a vertex/point in this triangle, this traverses
        the triangulation about this point (CCW) and returns the list of 
        simplices that surround it in CCW order (as well as a list of local 
        ids for IDin in each of the simplices in the list). Note that this 
        requires that this simplex is connected up with other simplices in a 
        triangulation.

    SimpLink(S_other, S_other_locid, edge_share)
        Link self with S_other simplex. To deal with cases of degenerate 
        triangles (e.g. triangle with same point at different vertices), 
        extra parameters are passed in (S_other_locid and edge_share).
    """
    def __init__(self, IDlist, RelPtPos = [[0,0],[0,0],[0,0]]):
        """
        Parameters
        ----------
        IDlist : list of 3 ints
            List of the 3 point IDs from the master list (part of the 
            tranguation class). It is assumed that IDlist already refers to 
            points in the proper permutation (list order == ccw geometric 
            order). Sets the points attribute.

        RelPtPos : list of lists
            Sets relptregion, which records the relative position of points.
            If the edge connecting two points of this simplex crosses the 
            torus boundary, then we record the x,y integer displacement 
            (how many copies of the domain over in the x and y direction) 
            of the second point from the perspective of the first (with the
            first and second point ordering determined by the CCW simplex 
            ordering, and the unique edge indexed as usual - i.e. associated
            with the index of the point it is across from)
        
            Example: in a fundamental domain of Dx=Dy=1, consider the first 
            point at [0.5,0.7] and second point at [0.5,0.1], where the line
            connecting them goes through the top boundary (remember, the 
            periodic boundary conditions). Then the integer pair describing 
            this edge from this triangle would be [0,1], since we have to go
            one copy up in the y direction to get to the second point. This 
            extra bit of information allows us to consistently define 
            distances between points and find triangle areas.

        Notes
        -----
        Other attributes are set when linking them up in a triangulation2D 
        object.
        """  
        super().__init__(IDlist)
        self.relptregion = copy.deepcopy(RelPtPos) # the relative points regions

    # More complex than sibling class version (simplex2D), as it compares both edges 
    # and simplicesfor a halting criteria in traversing the simplices about a point.  
    # This allows for edge cases of degenerate triangulations, where some edges have 
    # boundary points that are the same point.
    def SimpNeighbors(self, IDin):
        """Finds the simpices which share a point.

        Parameters
        ----------
        IDin : int
            The ID of a vertex/point in this simplex.

        Returns
        -------
        list of simplex2D objects
            The simplices (in CCW cyclical order about the shared point - 
            IDin) adjacent to a point (IDin).

        list of ints
            The local ids of IDin in each simplex in the returned simplex list
        
        Notes
        -----
        This method requires that the simplex is part of a triangulation2D 
        object (so that it has neighboring simplices). The LocalIDList is 
        required for use cases where the NeighborList contains a simplex twice
        (and therefore the simplex contains IDin twice, and we can't just use 
        LocalID method)
        """
        NeighborList = []
        LocalIDList = []
        stpt = self.LocalID(IDin)
        NeighborList.append(self)
        LocalIDList.append(stpt)
        sharededge = self.edgeids[(stpt+1)%3]
        start_sharededge = sharededge
        lsimp = self.simplices[(stpt+1)%3]
        lsimplid = lsimp.edgeids.index(sharededge)
        NeighborList.append(lsimp)
        LocalIDList.append((lsimplid+1)%3)
        sharededge = NeighborList[-1].edgeids[(lsimplid+2)%3]
        lsimp = NeighborList[-1].simplices[(lsimplid+2)%3]
        lsimplid = lsimp.edgeids.index(sharededge)      
        while not (lsimp is NeighborList[1] 
                   and NeighborList[-1] is NeighborList[0] 
                   and start_sharededge == sharededge):
            NeighborList.append(lsimp)
            LocalIDList.append((lsimplid+1)%3)
            sharededge = NeighborList[-1].edgeids[(lsimplid+2)%3]
            lsimp = NeighborList[-1].simplices[(lsimplid+2)%3]                 
            lsimplid = lsimp.edgeids.index(sharededge)
        return NeighborList[:-1], LocalIDList[:-1] #exclude the last element

    def EdgeNeighbors(self, IDin):
        """Finds the edges which share a point.

        Parameters
        ----------
        IDin : int
            The ID of a vertex/point in this simplex.

        Returns
        -------
        list of 
            The edge ids (in CCW cyclical order about the shared point - 
            IDin) adjacent to a point (IDin).

        
        Notes
        -----
        This method requires that the simplex is part of a triangulation2D 
        object (so that it has neighboring simplices).
        """
        NeighborList = []
        LocalIDList = []
        EdgeList = []
        stpt = self.LocalID(IDin)
        NeighborList.append(self)
        sharededge = self.edgeids[(stpt+1)%3]
        EdgeList.append(sharededge)
        start_sharededge = sharededge
        lsimp = self.simplices[(stpt+1)%3]
        lsimplid = lsimp.edgeids.index(sharededge)
        NeighborList.append(lsimp)
        sharededge = NeighborList[-1].edgeids[(lsimplid+2)%3]
        EdgeList.append(sharededge)
        lsimp = NeighborList[-1].simplices[(lsimplid+2)%3]
        lsimplid = lsimp.edgeids.index(sharededge)      
        while not (lsimp is NeighborList[1] 
                   and NeighborList[-1] is NeighborList[0] 
                   and start_sharededge == sharededge):
            NeighborList.append(lsimp)
            sharededge = NeighborList[-1].edgeids[(lsimplid+2)%3]
            EdgeList.append(sharededge)
            lsimp = NeighborList[-1].simplices[(lsimplid+2)%3]                 
            lsimplid = lsimp.edgeids.index(sharededge)
        return EdgeList[:-1] #exclude the last element

    
    def SimpLink(self,S_other, S_other_locid, edge_share):
        """
        Links this simplex with S_other (and vice versa).
        Used during an edge flip operation to ensure the newly created
        simplices are integrated into the triangulation

        Parameters
        ----------
        S_other : Simplex2D object
            The simplex to link with self

        S_other_locid : int
            The local index of the shared edge in S_other

        edge_share : int
            The index of the shared edge between self and S_other
            
        Notes
        -----
        This version uses the extra information (new simplex local id, shared
        edge id) to avoid referencing the points (on the torus, cases exist 
        where two adjacent simplices share all three points)
        """
        edge_locid = self.edgeids.index(edge_share)
        self.simplices[edge_locid] = S_other
        S_other.simplices[S_other_locid] = self
# End of simplex2D_PBC class #################################################


# PrintParameters Class ######################################################
@dataclass
class PrintParameters(PrintParameters):
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
    """    
    pass
    #already have everything we need in the parent class
# End of PrintParameters Class ###############################################


# triangulation2D Class ######################################################
#This is the triangulation class, the central class in the overall algorithm. It is initialized using a Delaunay triangulation
class triangulation2D(triangulation2D_Base):
    """The central class in the overal Topological Advection algorithm, this
    class represents a triangulation of data points in a 2D domain.  It has
    methods for evolving the triangulation due to the motion of data points,
    acting as a basis for encoding loops, accumulating weight operators, and
    plotting.  This child of the parent triangulation2D_Base class works for
    doubly periodic boundary conditions (i.e. Torus).

    Note: Only the main attribues and methods are listed here.
    Note: The triangulation is initialized as a Delaunay triangulation.

    Attributes:
    -----------
    
    pointlist : list
        A list of simplex objects.  Object at index i has the point with 
        point id i in its point list.  Allows for O(1) lookup of points in the
        triangulation.  Note, not every simplex is in this list.

    FDsizes : list
        A list that sets the fundamental domain of the torus.  FDsize =
        [x_width, y_height].  By convention x_min, y_min = 0, and 
        x_max, y_max = FDsizes.  FDsizes are used for the moduluar 
        arithmatic that enforces the periodic boundaries.

    pointpos : list
        A list of the [x,y] positions for the points at the current time
        Note that all points must be within the fundamental domain.

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

    Plot(LoopIn = None, PP: PrintParameters = PrintParameters())
        This plots the triangulation and loop.  See PrintParameters data class
        documentation for details on the many options.

    TriCopy(EvolutionReset = True)
        This returns a copy of this triangulation2D object.
    """
    
    #The constructor for triangulation2D.  ptlist is the list of [x,y] positions for the points at the initial time.
    #Reminder that the input points are just in the fundamental domain.  We also have the size of the fundamental domain as [0,Dx) and [0,Dy) in the x and y directions repectively. Domain = [[x_min,y_min],[x_max,y_max]], for PBC x_min & y_min must be = 0
    #Important that the points are in this domain.  We also pass in Dx and Dy.  There are no control points, as this will be a triangulation without boundary.
     
    def __init__(self, ptlist, Domain = None, empty = False):
        """
        Parameters
        ----------
        ptlist : list
            ptlist is the list of [x,y] positions for the points at the 
            initial time.

        Domain : list
            Domain = [[x_min,y_min],[x_max,y_max]].  Both x_min and y_min 
            are equal to zero by convention, and we will save just the 
            width and height in FDsizes.  If Domain is None (default, but
            it is highly recommended to pass in a Domain), then the FDsizes
            will be determined from max x/y pt values.

        empty : bool
            Used for creating an empty object, which is then used for object 
            copying.  Default is False
        """
        if not empty: 
            if not Domain[0] == [0,0]:
                print("Domain must be of the form [[0,0], [x_width, y_height]]")
            self.FDsizes = Domain[1]  #[Dx,Dy]
        else:
            Domain_approx = HF.GetBoundingDomainSlice(ptlist, frac = 0.0)
            self.FDsizes = Domain_approx[1]
        self._dpindices = ((0,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0))
        self._ptnum = len(ptlist)
        #  These are two lists that give the relative copy of the FD that the
        #  final position is in relative to the initial position (initial pos 
        #  is thought of as being in the FD). The first one is relative to the
        #  initial positions and does not change in a given time-step (though 
        #  of course is re-filled at the beginning of each time-step).  The
        #  second one starts off as identical, but is updated as points cross 
        #  the boundary of the FD. To be more specific, as a point crosses the
        #  FD, we still think of it as being in the FD, but the copy of the FD
        #  that the final position is in (relative to this now-moved point 
        #  position) is now different.  The Crossing event list events will 
        #  update this list.
        #  FuturePosRelCopyLoc holds the indices that need to be put into 
        #  dpindices to get the shifts
        self._FuturePosRelCopyLoc = []   
        # UpdatedFuturePosRelCopyLoc directly holds the shifts (updated),
        # - this makes it easier to update
        self._UpdatedFuturePosRelCopyLoc = []  
        super().__init__(ptlist, empty)

    def _LoadPos(self,ptlist):
        #  enforce the fundamental domain boundaries
        self.pointpos = [ [pt[i]%self.FDsizes[i] for i in range(2)] for pt in ptlist]

    def _SetInitialTriangulation(self):
        #  Now we add on the 8 copies of the points in the fundamental domain,
        #  displaced to the surrounding rectangles.  The convention will be 
        #  that we start with the FD, then the copy down and to the left of 
        #  the fundamental domain, and work our way around the fundamental 
        #  domain counter clock-wise.
        temppoints = np.array(self.pointpos)
        temppoints2 = np.array(self.pointpos)
        for i in range(1,len(self._dpindices)):
            temppoints = np.append(temppoints, temppoints2 + np.array([self._dpindices[i][0]*self.FDsizes[0], self._dpindices[i][1]*self.FDsizes[1]]), axis=0)
        #  now we have 9x the number of actual points (+ the 8 neighboring copies)
        # slight shear to break degeneracy in consistent way in the case of one point
        if self._ptnum == 1:
            shear_val = 0.0001*self.FDsizes[0]
            for i in range(1,8):
                if i in [1,2,3]:
                    temppoints[i][0] += shear_val
                elif i in [5,6,7]:
                    temppoints[i][0] -= shear_val  
        #  create the initial Delaunay triangulation.  The option forces the 
        #  creation of simplices for degenerate points by applying a random
        #  perturbation.
        temptri = Delaunay(temppoints,qhull_options="QJ Pp") 
        #  the copy of the FD that each point in a triangle is in
        ptcopypos = []  
        #  convention: 0 for FD, 1 BL, 2 BC, 3 BR, 4 MR, 5 TR, 6 TC, 7 TL, 8 ML
        #the point in the FD that each point in a triangle maps to
        ptfdID = []  
        #  gives 1 if all 3 points are in the fundamental domain, 0 if at least
        #  one point in in the FD, and -1 if they are all outside the FD
        triloc, edgesimps, FDsimps = [], [], []
        for i in range(temptri.simplices.shape[0]):
            tempptlist = temptri.simplices[i].tolist()
            fdID, copypos = [], []
            for j in range(3):
                fdID.append(tempptlist[j]%self._ptnum)
                copypos.append(tempptlist[j]//self._ptnum)
            ptfdID.append(fdID)
            ptcopypos.append(copypos)
            InFD = [copypos[k] == 0 for k in range(3)]
            if all(InFD):  #  all points are in the FD
                triloc.append(1)
                FDsimps.append(i)
            elif any(InFD):  #  one or two points in the FD
                triloc.append(0)
                edgesimps.append(i)
            else:  #  no points in the FD
                triloc.append(-1)
        #  now we need to create a list of equivalent shifted triangles 
        #  (only ones that straddle the FD edge are needed)
        Equiv_Tri = [[] for i in range(temptri.simplices.shape[0])]
        for i in range(len(edgesimps)-1):
            for j in range(i+1,len(edgesimps)):
                ptidsi = temptri.simplices[edgesimps[i]]
                triptsi = [np.array(temptri.points[x]) for x in ptidsi]
                ptidsj = temptri.simplices[edgesimps[j]]
                triptsj = [np.array(temptri.points[x]) for x in ptidsj] 
                for l in range(3):
                    triptsj_roll = np.roll(triptsj,l,axis=0)
                    vec = [triptsj_roll[m]-triptsi[m] for m in range(3)]
                    diff_mag = 0
                    for m in range(3):
                        diff_mag += np.linalg.norm(vec[m]-vec[(m+1)%3])
                    if diff_mag < 0.000000001*self.FDsizes[0]:
                        #  if true, then the i and j triangles map to the same
                        #  points in the FD and they are the same triangle
                        Equiv_Tri[edgesimps[i]].append([edgesimps[j],l])
                        Equiv_Tri[edgesimps[j]].append([edgesimps[i],(l+l)%3])
                        break            
        #  Now we create the simplices.  Include all simplices with all pts
        #  in the FD, for ones with one or two pts in the FD, choose the first
        #  one encountered among the equiv. copies.  for each equiv copy not 
        #  chosen (and a border simplex) create the shift that takes the 
        #  internal id to the chosen copy's internal id (0,1,2)
        BigToLittleList = [None for i in range(temptri.simplices.shape[0])]
        LittleToBigList = []
        RelShift = [None for i in range(temptri.simplices.shape[0])]
        for i in FDsimps:
            self.simplist.append(simplex2D( (temptri.simplices[i]%self._ptnum).tolist()))
            LittleToBigList.append(i)
            BigToLittleList[i] = len(self.simplist)-1
            RelShift[i] = [i,0]
        for i in edgesimps:
            if RelShift[i] is None:  #haven't delt with this set, use this one at the canonical copy
                RelShift[i] = [i,0]  
                self.simplist.append(simplex2D( (temptri.simplices[i]%self._ptnum).tolist()))
                LittleToBigList.append(i)
                BigToLittleList[i] = len(self.simplist)-1
                #now deal with the copies
                for copyid in Equiv_Tri[i]:
                    RelShift[copyid[0]] = [i,copyid[1]]
                    BigToLittleList[copyid[0]] = len(self.simplist)-1         
        #  so, say we look at triangle 17.  RelShift[17] gives me two items: 
        #  RelShift[17][0] is the triangle id that corresponds to the 
        #  cannonical triangle among the set of equivalent (via translation) 
        #  FD boundary triangles. here: 11 of {11,17,37}.  RelShift[17][1] is
        #  the shift needed to go between the two triangles.  more 
        #  specifically, the internal id of RelShift[17][1] for triangle 11 
        #  corresponds to the internal id of 0 for triangle 17.

        #  Now we go through each point in the FD.  For each point we loop 
        #  around the simplices that share this point. We link adjacent pairs 
        #  of simplices, add in the relptpos (of the two pts across from this 
        #  pt in the simp), add in the edge id, populate the pointlist, and 
        #  add in the SLindex
        self.pointlist = [None for i in range(self._ptnum)]
        edgecounter = 0
        for pt in range(self._ptnum):
            start_simp_id_big = temptri.vertex_to_simplex[pt]
            start_simp_pt_loc_id_big = temptri.simplices[start_simp_id_big].tolist().index(pt)
            start_simp_pt_loc_id = (RelShift[start_simp_id_big][1]+start_simp_pt_loc_id_big)%3
            start_simp_id = BigToLittleList[start_simp_id_big]
            start_simp = self.simplist[start_simp_id]
            self.pointlist[pt] = start_simp  #  populating the pointlist
            #  set up the relptregion
            ptr = temptri.simplices[start_simp_id_big][(start_simp_pt_loc_id_big+1)%3]
            ptl = temptri.simplices[start_simp_id_big][(start_simp_pt_loc_id_big+2)%3]
            rpl = (np.array(self._dpindices[ptl//self._ptnum]) - np.array(self._dpindices[ptr//self._ptnum])).tolist()
            start_simp.relptregion[start_simp_pt_loc_id] = rpl
            start_simp.SLindex = start_simp_id
            this_simp_id_big = start_simp_id_big
            this_simp_pt_loc_id_big = start_simp_pt_loc_id_big
            this_simp_pt_loc_id = start_simp_pt_loc_id
            this_simp = start_simp
            next_simp_id_big = temptri.neighbors[this_simp_id_big][(this_simp_pt_loc_id_big+1)%3]
            next_simp_pt_loc_id_big = (temptri.neighbors[next_simp_id_big].tolist().index(this_simp_id_big)+1)%3
            next_simp_pt_loc_id = (RelShift[next_simp_id_big][1]+next_simp_pt_loc_id_big)%3
            next_simp_id = BigToLittleList[next_simp_id_big]
            next_simp = self.simplist[next_simp_id]
            #  linking the two simplices and adding in the edge id
            this_simp.simplices[(this_simp_pt_loc_id+1)%3] = next_simp
            next_simp.simplices[(next_simp_pt_loc_id+2)%3] = this_simp
            if this_simp.edgeids[(this_simp_pt_loc_id+1)%3] is None:
                this_simp.edgeids[(this_simp_pt_loc_id+1)%3] = edgecounter
                next_simp.edgeids[(next_simp_pt_loc_id+2)%3] = edgecounter
                edgecounter += 1
            #  now for the while loop:
            while not next_simp_id_big == start_simp_id_big:
                #  get relptregion for next simp
                ptr = temptri.simplices[next_simp_id_big][(next_simp_pt_loc_id_big+1)%3]
                ptl = temptri.simplices[next_simp_id_big][(next_simp_pt_loc_id_big+2)%3]
                rpl = (np.array(self._dpindices[ptl//self._ptnum]) - np.array(self._dpindices[ptr//self._ptnum])).tolist()
                next_simp.relptregion[next_simp_pt_loc_id] = rpl
                next_simp.SLindex = next_simp_id
                #  save next simp as this simp
                this_simp_id_big = next_simp_id_big
                this_simp_pt_loc_id_big = next_simp_pt_loc_id_big
                this_simp_pt_loc_id = next_simp_pt_loc_id
                this_simp = next_simp
                #  find next simp
                next_simp_id_big = temptri.neighbors[this_simp_id_big][(this_simp_pt_loc_id_big+1)%3]
                next_simp_pt_loc_id_big = (temptri.neighbors[next_simp_id_big].tolist().index(this_simp_id_big)+1)%3
                next_simp_pt_loc_id = (RelShift[next_simp_id_big][1]+next_simp_pt_loc_id_big)%3
                next_simp_id = BigToLittleList[next_simp_id_big]
                next_simp = self.simplist[next_simp_id]
                #  linking the two simplices
                this_simp.simplices[(this_simp_pt_loc_id+1)%3] = next_simp
                next_simp.simplices[(next_simp_pt_loc_id+2)%3] = this_simp
                if this_simp.edgeids[(this_simp_pt_loc_id+1)%3] is None:
                    this_simp.edgeids[(this_simp_pt_loc_id+1)%3] = edgecounter
                    next_simp.edgeids[(next_simp_pt_loc_id+2)%3] = edgecounter
                    edgecounter += 1
        self._totalnumedges = edgecounter 

    def _LoadNewPos(self,ptlist):
        #  putting the new point positions in pointposfuture
        self._pointposfuture = ptlist  
        self._FuturePosRelCopyLoc = self._GetNewPosCopyLoc()
        #  this will be updated as the points cross boundaries
        self._UpdatedFuturePosRelCopyLoc = [[self._dpindices[x][0],self._dpindices[x][1]] for x in self._FuturePosRelCopyLoc] 

    def _GetNewPosCopyLoc(self):
        #  Function that takes all of the current and future positions and 
        #  gets the copy of the FD that the future positions are in. Because
        #  the positions are constrained to be in the FD, we must find the 
        #  copy (9 options) that has the smallest distance between the inital
        #  and proposed final positions.
        if self._Vec:
            #  vectorized version
            return self._GetNewPosCopyLocVec() 
        else:
            return self._GetNewPosCopyLocSingle()

    def _GetNewPosCopyLocSingle(self):
        copyloc = []
        Dxh = self.FDsizes[0]/2
        Dyh = self.FDsizes[1]/2 
        for i in range(len(self._pointposfuture)):
            posi = self.pointpos[i]
            posf = self._pointposfuture[i]
            dz = [posf[0] - posi[0], posf[1] - posi[1]]
            dzabs = [abs(dz[0]),abs(dz[1])]
            if dzabs[0] < Dxh: #  x first
                if dzabs[1] < Dyh: copyloc.append(0)
                else:  #  moved out of FD in y dir
                    #  moved out of FD through bottom
                    if dz[1] > 0: copyloc.append(2) 
                    #  moved out through top    
                    else: copyloc.append(6)  
            else: #  moved out of FD in x dir
                if dz[0] > 0: #  moved out of FD to left
                    if dzabs[1] < Dyh: copyloc.append(8)
                    else:  #  moved out of FD in y dir
                        #  moved out of FD through bottom
                        if dz[1] > 0: copyloc.append(1) 
                        else: copyloc.append(7) 
                else:  #  moved out to the right
                    if dzabs[1] < Dyh: copyloc.append(4)
                    else: #  moved out of FD in y dir
                        #  moved out of FD through bottom
                        if dz[1] > 0: copyloc.append(3) 
                        else: copyloc.append(5) 
        return copyloc
                    
    def _GetNewPosCopyLocVec(self):
        #  vectorized version of above (usually faster)
        posix = np.array([self.pointpos[i][0] for i in range(self._ptnum)])
        posiy = np.array([self.pointpos[i][1] for i in range(self._ptnum)])
        posfx = np.array([self._pointposfuture[i][0] for i in range(self._ptnum)])
        posfy = np.array([self._pointposfuture[i][1] for i in range(self._ptnum)])
        Dx, Dy = self.FDsizes
        copyloc = HF.CopyLocations(posix,posiy,posfx,posfy,Dx,Dy)
        return copyloc.astype(int).tolist()

    def _GetEvents(self):
        return [self._GetCollapseEvents(), self._GetPtCrossEvents()]
    
    def _GetPtCrossEvents(self):
        #  We need to find, for each point, the time(s) it crosses any of 
        #  the FD boundary lines and which line it crosses
        CrossList = []
        #  first get the list of future pos copy locations
        Dx, Dy = self.FDsizes
        Lines = [[[0,-Dy],[0,2*Dy]],[[Dx,-Dy],[Dx,2*Dy]],[[-Dx,0],[2*Dx,0]],[[-Dx,Dy],[2*Dx,Dy]]]
        movedir = [[-1,0],[1,0],[0,-1],[0,1]]
        for i in range(len(self._pointposfuture)):
            if not self._FuturePosRelCopyLoc[i] == 0:
                posi = [self.pointpos[i][0],self.pointpos[i][1]]
                posf = [self._pointposfuture[i][0],self._pointposfuture[i][1]]
                posfn = [posf[k]+self.FDsizes[k]*self._dpindices[self._FuturePosRelCopyLoc[i]][k] for k in range(2)]
                newline = [posi,posfn]
                for j in range(len(Lines)):
                    IsInt = HF.IsIntersection(newline,Lines[j],timeinfo = True)
                    if IsInt[0]:
                        CrossList.append([i,IsInt[1],movedir[j]])  
        CrossList.sort(key=itemgetter(1),reverse=True)    
        #  returns the point index, the time of crossing, and the move 
        #  direction (-1,0,1 for both the x and y directions)
        return CrossList
       
    def _AreaZeroTimeSingle(self, SimpIn, Tin = 0):
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
        #  first, by convention, we are going to take a specific copy of 
        #  this simplex ... the one where the first point stored in the 
        #  simplex is considered to be in the fundamental domain.  For 
        #  boundary simplices, this gives us one copy to consider.
        ptlist = SimpIn.points
        rpr = SimpIn.relptregion
        pt1shift = rpr[2]
        pt2shift = [-1*rpr[1][0],-1*rpr[1][1]]
        #  now shift the Initial and Final points appropriately
        Dx, Dy = self.FDsizes
        Initpos = []
        if Tin == 0:
            for i in range(3):
                Initpos.append(self.pointpos[ptlist[i]][0])
                Initpos.append(self.pointpos[ptlist[i]][1])
        else:
            for i in range(3):
                CurrentLoc = self._GetCurrentLoc(ptlist[i],Tin)
                Initpos.append(CurrentLoc[0])
                Initpos.append(CurrentLoc[1])
        Finalpos = []
        for i in range(3):
            Finalpos.append(self._pointposfuture[ptlist[i]][0])
            Finalpos.append(self._pointposfuture[ptlist[i]][1])  
        Initpos[2] += Dx*pt1shift[0]
        Initpos[3] += Dy*pt1shift[1]
        Finalpos[2] += Dx*pt1shift[0]
        Finalpos[3] += Dy*pt1shift[1]
        Initpos[4] += Dx*pt2shift[0]
        Initpos[5] += Dy*pt2shift[1]
        Finalpos[4] += Dx*pt2shift[0]
        Finalpos[5] += Dy*pt2shift[1] 
        #  Finally, we see if any of the final points have crossed a 
        #  boundary, and correct for this (notice that we used the 
        #  updated version)
        cploc = [self._UpdatedFuturePosRelCopyLoc[x] for x in ptlist]
        for i in range(3):
            Finalpos[2*i] += Dx*cploc[i][0]
            Finalpos[2*i+1] += Dy*cploc[i][1]
        AZT =  HF.AreaZeroTimeBaseSingle(Initpos[0], Initpos[1], Initpos[2],
                    Initpos[3], Initpos[4], Initpos[5], Finalpos[0], 
                    Finalpos[1], Finalpos[2], Finalpos[3], Finalpos[4], 
                    Finalpos[5],  0)
        #  now, the returned time is between 0 and 1, while we need a time 
        #  between Tin and 1, so we uniformly contract this time
        if AZT[0] and not Tin == 0:
            AZT[1] = AZT[1]*(1-Tin)+Tin
        return AZT


    # AreaZeroTimeMultiple goes through every simplex and looks for whether the area zero time is between Tin and 1.  Similar to AreaZeroTimeSingle, but wrapping up the info in np arrays to get vectorization and jit boost.
    def _AreaZeroTimeMultiple(self, Tin = 0):
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
        Dx, Dy = self.FDsizes
        nsimps = len(self.simplist)
        pts0 = np.array([self.simplist[i].points[0] for i in range(nsimps)])
        pts1 = np.array([self.simplist[i].points[1] for i in range(nsimps)])
        pts2 = np.array([self.simplist[i].points[2] for i in range(nsimps)])
        npptpos = np.array(self.pointpos)
        npptposf = np.array(self._pointposfuture)
        IP0x, IP0y = npptpos[pts0,0], npptpos[pts0,1]
        IP1x, IP1y = npptpos[pts1,0], npptpos[pts1,1]
        IP2x, IP2y = npptpos[pts2,0], npptpos[pts2,1]
        FP0x, FP0y = npptposf[pts0,0], npptposf[pts0,1]
        FP1x, FP1y = npptposf[pts1,0], npptposf[pts1,1]
        FP2x, FP2y = npptposf[pts2,0], npptposf[pts2,1]
        pt1shiftx = np.array([self.simplist[i].relptregion[2][0] for i in range(nsimps)])
        pt1shifty = np.array([self.simplist[i].relptregion[2][1] for i in range(nsimps)])
        pt2shiftx = np.array([-1*self.simplist[i].relptregion[1][0] for i in range(nsimps)])
        pt2shifty = np.array([-1*self.simplist[i].relptregion[1][1] for i in range(nsimps)])
        IP1x += Dx*pt1shiftx
        IP1y += Dy*pt1shifty
        FP1x += Dx*pt1shiftx
        FP1y += Dy*pt1shifty
        IP2x += Dx*pt2shiftx
        IP2y += Dy*pt2shifty
        FP2x += Dx*pt2shiftx
        FP2y += Dy*pt2shifty
        pt0finalshiftx = np.array([self._dpindices[self._FuturePosRelCopyLoc[
                                   self.simplist[i].points[0]]][0] 
                                   for i in range(nsimps)])
        pt0finalshifty = np.array([self._dpindices[self._FuturePosRelCopyLoc[
                                   self.simplist[i].points[0]]][1]
                                   for i in range(nsimps)])
        pt1finalshiftx = np.array([self._dpindices[self._FuturePosRelCopyLoc[
                                   self.simplist[i].points[1]]][0] 
                                   for i in range(nsimps)])
        pt1finalshifty = np.array([self._dpindices[self._FuturePosRelCopyLoc[
                                   self.simplist[i].points[1]]][1] 
                                   for i in range(nsimps)])
        pt2finalshiftx = np.array([self._dpindices[self._FuturePosRelCopyLoc[
                                   self.simplist[i].points[2]]][0] 
                                   for i in range(nsimps)])
        pt2finalshifty = np.array([self._dpindices[self._FuturePosRelCopyLoc[
                                   self.simplist[i].points[2]]][1] 
                                   for i in range(nsimps)])
        FP0x += Dx*pt0finalshiftx
        FP0y += Dy*pt0finalshifty
        FP1x += Dx*pt1finalshiftx
        FP1y += Dy*pt1finalshifty
        FP2x += Dx*pt2finalshiftx
        FP2y += Dy*pt2finalshifty
        #  vectorized (+jit)
        return HF.AreaZeroTimeBaseVec(IP0x,IP0y,IP1x,IP1y,IP2x,IP2y,FP0x,FP0y,FP1x,FP1y,FP2x,FP2y,Tin)

    def _GEvolve(self, EventLists):
        """
        Processes an ordered list of events (collapse) and does edge flips to 
        update the triangulation.  Also adds in new events as needed. Finished
        when there are no more events in the time interval, and the 
        triangulation is consistent with the new set of points.
        """        
        EventListSimp, EventListCrossing = EventLists
        delta = 1e-10
        while len(EventListSimp)> 0 or len(EventListCrossing) > 0:
            latestSimpEventTime = 1
            latestCrossingEventTime = 1
            if len(EventListSimp)> 0:
                latestSimpEventTime = EventListSimp[-1][1]
            if len(EventListCrossing) > 0:
                latestCrossingEventTime = EventListCrossing[-1][1]
                
            if latestSimpEventTime < latestCrossingEventTime:
                #  here we deal with simplex collapse events
                CollSimp, currenttime = EventListSimp[-1]
                #  deal with simplex collapse events here
                #  returns ... [[leftsimp,rightsimp],topsimp (old)]
                newsimps, delsimp = self._SFix(CollSimp, currenttime)    
                del EventListSimp[-1]  #  get rid of the evaluated event
                #  first find the time of zero area for potential top 
                #  simplex event, and delete it if it is in the eventlist
                collapsed, collapse_time = self._AreaZeroTimeSingle(delsimp, currenttime - delta)
                if collapsed:
                    HF.BinarySearchDel(EventListSimp, [delsimp, collapse_time])
                #  now run through the newsimps list and see if each object 
                #  goes through zero area in the remaining time (if so, add to 
                #  EventList with the calulated time to zero area)
                for simp in newsimps:
                    collapsed, collapse_time = self._AreaZeroTimeSingle(simp, currenttime - delta)
                    if collapsed:  #  insert in the event list at the correct spot
                        HF.BinarySearchIns(EventListSimp,[simp, collapse_time])
            else:
                #  here we deal with the crossing events
                currenttime = latestCrossingEventTime
                ptindex = EventListCrossing[-1][0]
                ptmove = EventListCrossing[-1][2]
                # update self.UpdatedFuturePosRelCopyLoc
                self._UpdatedFuturePosRelCopyLoc[ptindex][0] -= ptmove[0]
                self._UpdatedFuturePosRelCopyLoc[ptindex][1] -= ptmove[1]
                #  We need to update the relative position data in every
                #  simplex that shares this point
                #  first get a list of all the simplices that bound this point
                Sset, SLid = self.pointlist[ptindex].SimpNeighbors(ptindex)
                for simp, locid in zip(Sset,SLid):
                    simp.relptregion[(locid+1)%3][0] += ptmove[0]
                    simp.relptregion[(locid+1)%3][1] += ptmove[1]
                    simp.relptregion[(locid+2)%3][0] -= ptmove[0]
                    simp.relptregion[(locid+2)%3][1] -= ptmove[1]
                #  now we need to delete the crossing event
                del EventListCrossing[-1]

    def _GetSimpCurrentLoc(self, SimpIn, timeIn):
        ptlist = SimpIn.points
        rpr = SimpIn.relptregion
        #now shift the Initial and Final points appropriately
        pos = [self._GetCurrentLoc(x,timeIn,True) for x in ptlist]
        for i in range(2):
            pos[1][i] += self.FDsizes[i]*rpr[2][i]
            pos[2][i] -= self.FDsizes[i]*rpr[1][i]
        return pos

    def _GetCurrentLoc(self, PtInd, timeIn, mod = True):
        #  gets the current position of a given point (pass in a point index)
        #  by taking the linear interpolation from the initial position to 
        #  the final postion, then moding by the boundary size so that the 
        #  point is in the FD.  timeIn is in [0,1]
        posi = self.pointpos[PtInd]
        posf = self._pointposfuture[PtInd]
        posfn = [posf[k]+self.FDsizes[k]*self._dpindices[self._FuturePosRelCopyLoc[PtInd]][k] for k in range(2)]
        if mod:
            return [((posfn[k]-posi[k])*timeIn + posi[k])%self.FDsizes[k] for k in range(2)]
        else:
            return [((posfn[k]-posi[k])*timeIn + posi[k]) for k in range(2)]  


    def _EdgeFlip(self, AdjSimps, EdgeShare, TimeIn = None):
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
        #  first get the local ids of the points not shared by these simplices
        Simp = AdjSimps[0]
        Topsimp = AdjSimps[1]
        bptlid = Simp.edgeids.index(EdgeShare)          
        bpt = Simp.points[bptlid]
        rptlid = (bptlid+1)%3
        lptlid = (bptlid+2)%3
        rpt = Simp.points[rptlid]
        lpt = Simp.points[lptlid]
        tptuid = Topsimp.edgeids.index(EdgeShare)
        tpt = Topsimp.points[tptuid]
        lptuid = (tptuid+1)%3
        rptuid = (tptuid+2)%3
        rslist = [bpt,rpt,tpt]
        lslist = [bpt,tpt,lpt]
        rsimp = simplex2D(rslist)  #  new right simplex
        lsimp = simplex2D(lslist)  #  new left simplex
        #  create the list of edge ids for the weight operator
        WeightIDs = [EdgeShare, Topsimp.edgeids[lptuid], 
                     Topsimp.edgeids[rptuid], Simp.edgeids[rptlid], 
                     Simp.edgeids[lptlid]]
        #  create the weight operater and append to the list
        self.WeightOperatorList.append(WeightOperator(WeightIDs,TimeIn))
        #  need to create the relptregion data for the new simplices
        for k in range(2):
            rsimp.relptregion[0][k] = Topsimp.relptregion[lptuid][k]
            rsimp.relptregion[1][k] = Topsimp.relptregion[rptuid][k] + Simp.relptregion[rptlid][k]
            rsimp.relptregion[2][k] = Simp.relptregion[lptlid][k]
            lsimp.relptregion[0][k] = Topsimp.relptregion[rptuid][k]
            lsimp.relptregion[1][k] = Simp.relptregion[rptlid][k]
            lsimp.relptregion[2][k] = Topsimp.relptregion[lptuid][k] + Simp.relptregion[lptlid][k]    
        #  now create the links these simplices have to other simplices
        #  first determine if this is a case where some of the adjacent 
        #  simplices are either Simp or Topsimp
        if Simp.edgeids[rptlid] == Topsimp.edgeids[lptuid]:
            Topsimp.simplices[rptuid].SimpLink(lsimp,0,Topsimp.edgeids[rptuid])
            Simp.simplices[lptlid].SimpLink(rsimp,2,Simp.edgeids[lptlid])
            rsimp.simplices[0] = lsimp
            lsimp.simplices[1] = rsimp
        elif Simp.edgeids[lptlid] == Topsimp.edgeids[rptuid]:
            Topsimp.simplices[lptuid].SimpLink(rsimp,0,Topsimp.edgeids[lptuid])
            Simp.simplices[rptlid].SimpLink(lsimp,1,Simp.edgeids[rptlid])
            rsimp.simplices[2] = lsimp
            lsimp.simplices[0] = rsimp
        else:
            #  regular case
            Topsimp.simplices[lptuid].SimpLink(rsimp,0,Topsimp.edgeids[lptuid])
            Topsimp.simplices[rptuid].SimpLink(lsimp,0,Topsimp.edgeids[rptuid])
            Simp.simplices[lptlid].SimpLink(rsimp,2,Simp.edgeids[lptlid])
            Simp.simplices[rptlid].SimpLink(lsimp,1,Simp.edgeids[rptlid])
        rsimp.simplices[1] = lsimp
        lsimp.simplices[2] = rsimp
        #  also need to reassign the weight ids
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
        #  look through the simplex point list to see if either of the bad 
        #  simplices were there and replace if so
        if self.pointlist[bpt] is Simp:
            self.pointlist[bpt] = rsimp
        if (self.pointlist[rpt] is Simp) or (self.pointlist[rpt] is Topsimp):
            self.pointlist[rpt] = rsimp
        if self.pointlist[tpt] is Topsimp:
            self.pointlist[tpt] = lsimp
        if (self.pointlist[lpt] is Simp) or (self.pointlist[lpt] is Topsimp):
            self.pointlist[lpt] = lsimp
        #  Next, delete all the references to simplices in both 
        #  of the bad simplices
        for i in range(3):
            Simp.simplices[i] = None
            Topsimp.simplices[i] = None        
        return [lsimp,rsimp]   

    
    def MakeDelaunay(self):
        """
        MakeDelaunay takes the current triangulation and, through a series
        of edge flips, changes it into the Delaunay triangulation for this 
        point configuration.  This function changes the underlying 
        triangulation
        """           
        IsD, EdgeBSimps = None, None
        #  vectorized version (this one only makes marginal improvements)
        if self._Vec:  
            IsD, EdgeBSimps = self._IsDelaunay()
        else:
            EdgeBSimps = [None for i in range(self._totalnumedges)]
            EdgeUsed = [False for i in range(self._totalnumedges)]
            IsD = [False for i in range(self._totalnumedges)]
            for simp in self.simplist:
                for j in range(3):
                    edgeid = simp.edgeids[j]
                    if not EdgeUsed[edgeid]:
                        EdgeUsed[edgeid] = True
                        EdgeBSimps[edgeid] = [[simp,simp.simplices[j]],edgeid, True]
                        IsD[edgeid] = self._IsLocallyDelaunay([simp,simp.simplices[j]],edgeid)
        EdgeList = [EdgeBSimps[i] for i in range(self._totalnumedges) if IsD[i] == False]
        EdgeList_Epos = [None for i in range(self._totalnumedges)]
        for i in range(len(EdgeList)):
            EdgeList_Epos[EdgeList[i][1]] = i
        #  now go through the edge list and start flipping edges
        while len(EdgeList) > 0:
            EdgeSimps, edge_id, checked = EdgeList.pop()
            EdgeList_Epos[edge_id] = None
            Flip = True
            if not checked:
                Flip = not self._IsLocallyDelaunay(EdgeSimps,edge_id)
            if Flip:
                LRsimps = self._EdgeFlip(EdgeSimps, edge_id, self._atstep)
                for i in range(2): # Left and right simplices
                    loc = LRsimps[i].edgeids.index(edge_id)
                    lrsimp = LRsimps[i]
                    for j in range(2): # upper and lower simplices
                        eid = lrsimp.edgeids[(loc+1+j)%3]
                        adjsimp = lrsimp.simplices[(loc+1+j)%3]
                        ELinsert = [[lrsimp,adjsimp], eid, False]
                        if EdgeList_Epos[eid] == None:
                            EdgeList_Epos[eid] = len(EdgeList)
                            EdgeList.append(ELinsert)
                        else:
                            EdgeList[EdgeList_Epos[eid]] = ELinsert    


    def _IsDelaunay(self):
        """IsDelaunay outputs an array (length = number of edges) of booleans,
        which indicate if the quadrilateral with the ith edge as a diagonal is
        Delaunay. Also outputs an array of the pairs of simplices which bound 
        each edge. This calls IsDelaunayBase (from a helper function module)
        for a jit speed-up.
        """        
        Ax, Ay, Bx, By, Cx, Cy, Dx, Dy = [np.empty(self._totalnumedges) for i in range(8)]
        EdgeUsed = [False for i in range(self._totalnumedges)]
        BoundingSimps = [None for i in range(self._totalnumedges)]
        for simp in self.simplist:
            for j in range(3):
                edgeid = simp.edgeids[j]
                if not EdgeUsed[edgeid]:
                    EdgeUsed[edgeid] = True
                    Apt = simp.points[(j+2)%3]
                    Ax[edgeid], Ay[edgeid] = self.pointpos[Apt]
                    Bpt = simp.points[j]
                    Bx[edgeid], By[edgeid] = self.pointpos[Bpt]
                    Cpt = simp.points[(j+1)%3]
                    Cx[edgeid], Cy[edgeid] = self.pointpos[Cpt]
                    adjsimp = simp.simplices[j]
                    BoundingSimps[edgeid] = [[simp,adjsimp],edgeid, True]
                    adjsimp_loc_id = adjsimp.edgeids.index(edgeid)
                    Dpt = adjsimp.points[adjsimp_loc_id]
                    Dx[edgeid], Dy[edgeid] = self.pointpos[Dpt]
                    #now need to modify point positions.  Will use pt A as reference point (in FD)
                    Bx[edgeid] += simp.relptregion[(j+1)%3][0]*self.FDsizes[0]
                    By[edgeid] += simp.relptregion[(j+1)%3][1]*self.FDsizes[1]
                    Cx[edgeid] -= simp.relptregion[j][0]*self.FDsizes[0]
                    Cy[edgeid] -= simp.relptregion[j][1]*self.FDsizes[1]
                    Dx[edgeid] -= adjsimp.relptregion[(adjsimp_loc_id+2)%3][0]*self.FDsizes[0]
                    Dy[edgeid] -= adjsimp.relptregion[(adjsimp_loc_id+2)%3][1]*self.FDsizes[1]
        #return IsDelaunayBase(Ax,Ay,Bx,By,Cx,Cy,Dx,Dy)
        return HF.IsDelaunayBase(Ax,Ay,Bx,By,Cx,Cy,Dx,Dy), BoundingSimps

    
    def _IsLocallyDelaunay(self,AdjSimps,edgeid):
        """Given the two adjacent simplices, this determine if the 
        configuration is locally Delaunay.  Returns True or False.
        """        
        simp = AdjSimps[0]
        locid = simp.edgeids.index(edgeid)
        Apt = simp.points[(locid+2)%3]
        Ax, Ay = self.pointpos[Apt]
        Bpt = simp.points[locid]
        Bx, By = self.pointpos[Bpt]
        Cpt = simp.points[(locid+1)%3]
        Cx, Cy = self.pointpos[Cpt]
        adjsimp = simp.simplices[locid]
        adjsimp_loc_id = adjsimp.edgeids.index(edgeid)
        Dpt = adjsimp.points[adjsimp_loc_id]
        Dx, Dy = self.pointpos[Dpt]
        #  now need to modify point positions.  
        #  Will use pt A as reference point (in FD)
        Bx += simp.relptregion[(locid+1)%3][0]*self.FDsizes[0]
        By += simp.relptregion[(locid+1)%3][1]*self.FDsizes[1]
        Cx -= simp.relptregion[locid][0]*self.FDsizes[0]
        Cy -= simp.relptregion[locid][1]*self.FDsizes[1]
        Dx -= adjsimp.relptregion[(adjsimp_loc_id+2)%3][0]*self.FDsizes[0]
        Dy -= adjsimp.relptregion[(adjsimp_loc_id+2)%3][1]*self.FDsizes[1]
        ADx = Ax - Dx
        ADy = Ay - Dy
        BDx = Bx - Dx
        BDy = By - Dy
        CDx = Cx - Dx
        CDy = Cy - Dy
        AD2 = ADx*ADx + ADy*ADy
        BD2 = BDx*BDx + BDy*BDy
        CD2 = CDx*CDx + CDy*CDy
        detvals = ADx*(BDy*CD2-CDy*BD2) - ADy*(BDx*CD2-CDx*BD2) + AD2*(BDx*CDy-CDx*BDy)
        return (detvals < 0)   

    def _ArePointsAdjacent(self,pt1,pt2):
        AreAdjacent = False
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
        
        NextSimp = StartSimp.simplices[(locid+1)%3]
        locid = NextSimp.LocalID(pt1)
        while not NextSimp is StartSimp and len(LRsimpspos) < 2:
            if pt2 in NextSimp.points:
                AreAdjacent = True
                LRsimps.append(NextSimp)
                LRsimpspos.append(simpposcounter)                
            simpposcounter +=1
            NextSimp = NextSimp.simplices[(locid+1)%3]
            locid = NextSimp.LocalID(pt1)
        if AreAdjacent:
            if LRsimpspos[1] == LRsimpspos[0]+1:
                return [AreAdjacent,[LRsimps[1],LRsimps[0]]]
            else:
                return [AreAdjacent,LRsimps]
        else:
            return [AreAdjacent,None]

    def _DoesCurveLeft(self,pttriple):
        pt1, pt2, pt3 = pttriple
        pos1 = self.pointpos[pt1]
        pos2 = self.pointpos[pt2]
        pos3 = self.pointpos[pt3]
        crossP = (pos3[0] - pos2[0])*(pos1[1] - pos2[1]) - (pos3[1] - pos2[1])*(pos1[0] - pos2[0])
        return crossP >= 0

    #this determines if the given point (ptin) is to the left of line that goes from the first to second point in linepts
    #Used in determining the edges crossed in an initial band
    def IsLeft(self,linepts,ptin):
        pttriple = [ptin,linepts[0], linepts[1]]
        return self._DoesCurveLeft(pttriple)

    #This takes the two points in linepoints and adds a weight of one (or non-default value) to any edges that are crossed
    #by the line.
    def _AddWeightsAlongLine(self,linepoints, Boolin, LoopIn, wadd = 1.0):
        pt1, pt2 = linepoints
        if Boolin[1][0]: #this is the case of adjacent points (i.e. the line between the points is an edge)
            #only if the curvelefts' (Boolin[0], Boolin[2]) are opposite one another, do we add a weight
            if Boolin[0] is not Boolin[2]:
                pt1rtlocid = Boolin[1][1][1].LocalID(pt1)
                edgeindex = Boolin[1][1][1].edgeids[(pt1rtlocid+1)%3]
                LoopIn.weightlist[edgeindex] += wadd
        else:
            #first we need to determine the direction (which simplex) to set out from that has pt1 as a point.
            stlocid, StartSimp = self._SimpInDir([pt1,pt2])
            endlocid, EndSimp = self._SimpInDir([pt2,pt1])
            if not pt2 in StartSimp.points:
                edgeindex = StartSimp.edgeids[stlocid]
                LoopIn.weightlist[edgeindex] += wadd
                leftpoint = StartSimp.points[(stlocid+2)%3]
                CurrentSimp = StartSimp.simplices[stlocid]
                leftptloc = CurrentSimp.LocalID(leftpoint)
                while not CurrentSimp is EndSimp:
                    ptcompare = CurrentSimp.points[(leftptloc+2)%3]
                    indexadd = 0
                    if not self.IsLeft(linepoints,ptcompare):
                        indexadd = 1
                    edgeindex = CurrentSimp.edgeids[(leftptloc+indexadd)%3]
                    LoopIn.weightlist[edgeindex] += wadd
                    leftpoint = CurrentSimp.points[(leftptloc+indexadd+2)%3]
                    CurrentSimp = CurrentSimp.simplices[(leftptloc+indexadd)%3]
                    leftptloc = CurrentSimp.LocalID(leftpoint)

    #this returns the simplex (and local point id) that contains the first of linepoints, 
    #and has the line (to the second point) passing through it
    def _SimpInDir(self,linepoints):
        pt1 = linepoints[0]
        pt2 = linepoints[1]
        StartSimp = self.pointlist[pt1]
        locpt = StartSimp.LocalID(pt1)
        ptright = StartSimp.points[(locpt+1)%3]
        ptleft = StartSimp.points[(locpt+2)%3]
        while not ((not self.IsLeft([pt1,pt2],ptright)) and self.IsLeft([pt1,pt2],ptleft)):
            StartSimp = StartSimp.simplices[(locpt+1)%3]
            locpt = StartSimp.LocalID(pt1)
            ptright = StartSimp.points[(locpt+1)%3]
            ptleft = StartSimp.points[(locpt+2)%3]
        return locpt, StartSimp

    #This takes the central point in pttriple and adds in the weight of wadd to each of the radial edges starting
    #from the edge that is part of the simplex bisected by pt1 and pt2, to the edge that is part of the simplex
    #bisected by pt2 and pt3
    def _AddWeightsAroundPoint(self, pttriple, Boolin, LoopIn, wadd = 1):
        pt1, pt2, pt3 = pttriple
        indadd = 1
        if not Boolin[1]:  indadd = 2 # curve right triggers this
        stlocid, StartSimp = None, None
        if Boolin[0][0]:
            if not Boolin[1]:
                StartSimp = Boolin[0][1][0]
            else:
                StartSimp = Boolin[0][1][1]
            stlocid = StartSimp.LocalID(pt2)
        else:
            stlocid, StartSimp = self._SimpInDir([pt2,pt1])
        endlocid, EndSimp = None, None
        if Boolin[2][0]:
            if not Boolin[1]:
                EndSimp = Boolin[2][1][0]
            else:
                EndSimp = Boolin[2][1][1]
            endlocid = EndSimp.LocalID(pt2)   
        else:
            endlocid, EndSimp = self._SimpInDir([pt2,pt3])
        edgeindex = StartSimp.edgeids[(stlocid+indadd)%3]
        LoopIn.weightlist[edgeindex] += wadd
        CurrentSimp = StartSimp.simplices[(stlocid+indadd)%3]
        ptloc = CurrentSimp.LocalID(pt2)
        while not CurrentSimp is EndSimp:
            edgeindex = CurrentSimp.edgeids[(ptloc+indadd)%3]
            LoopIn.weightlist[edgeindex] += wadd
            CurrentSimp = CurrentSimp.simplices[(ptloc+indadd)%3]
            ptloc = CurrentSimp.LocalID(pt2)    

    #using ref pt, ix, iy, see if point is in simp.  if not, find edge intersection and then 
    # get adj simp, find ref point and dx,dy for shift that matches edge from simp.  Then calls self recursively
    # stops when simp is found with this point interior.  returns list of pairs [simp, edge] along path
    def _Simp_Hop(self, pt_in, simp, line_big, l_id, ix = 0, iy = 0, edge_prev = None, next_edge = None):
        delta = 1e-8
        #first see if pt_in is in the simp
        if self._Tri_Contains(pt_in,simp,l_id, ix, iy):
            return [[simp, None, l_id, ix, iy]]
        else:
            vertices = self.Get_Shifted_Vertices(simp,l_id, ix, iy)
            next_id = None
            if next_edge is not None:
                i = simp.edgeids.index(next_edge)
                Line_simp = [vertices[(i+1)%3],vertices[(i+2)%3]]
                if HF.IsIntersection(line_big,Line_simp):
                    next_id = i
            else: 
                if edge_prev is None:
                    for i in range(3):
                        Line_simp = [vertices[(i+1)%3],vertices[(i+2)%3]]
                        if HF.IsIntersection(line_big,Line_simp):
                            next_id = i
                            break
                else:
                    for i in range(3):
                        if not simp.edgeids[i] == edge_prev:
                            Line_simp = [vertices[(i+1)%3],vertices[(i+2)%3]]
                            if HF.IsIntersection(line_big,Line_simp):
                                next_id = i
                                break
            if next_id is None: return None
            edge = simp.edgeids[next_id]
            next_simp = simp.simplices[next_id]
            next_edge_id = next_simp.edgeids.index(edge)
            #now find the shifted parameters for the next simp s.t. the geometric edge matches up
            v1 = vertices[(next_id+2)%3]
            v2 = vertices[(next_id+1)%3]
            for i in range(3):
                l_id_trial = i
                for ix_trial in range(-1,2,1):
                    for iy_trial in range(-1,2,1):
                        vertices_trial = self.Get_Shifted_Vertices(next_simp,l_id_trial, ix_trial, iy_trial)
                        v1_trial = vertices_trial[(next_edge_id+1)%3]
                        v2_trial = vertices_trial[(next_edge_id+2)%3]
                        diff1 = v1_trial - v1
                        diff2 = v2_trial - v2
                        diff = np.hypot(diff1[0],diff1[1]) + np.hypot(diff2[0],diff2[1])
                        if diff < delta:
                            #have a match
                            return [[simp, edge,l_id, ix, iy]] + self._Simp_Hop(pt_in, next_simp, line_big, l_id_trial, ix_trial, iy_trial, edge)


    def Get_Shifted_Vertices(self, simp, l_id, ix = 0, iy = 0):
        vertices = [copy.copy(self.pointpos[p]) for p in simp.points]
        vertices[l_id][0] += ix*self.FDsizes[0]
        vertices[l_id][1] += iy*self.FDsizes[1]
        vertices[(l_id+1)%3][0] += (simp.relptregion[(l_id+2)%3][0] + ix)*self.FDsizes[0]
        vertices[(l_id+1)%3][1] += (simp.relptregion[(l_id+2)%3][1] + iy)*self.FDsizes[1]
        vertices[(l_id+2)%3][0] += (-simp.relptregion[(l_id+1)%3][0] + ix)*self.FDsizes[0]
        vertices[(l_id+2)%3][1] += (-simp.relptregion[(l_id+1)%3][1] + iy)*self.FDsizes[1]
        return np.array(vertices)
        
    
    def _Tri_Contains(self,pt,simp,l_id, ix = 0, iy = 0):
        vertices = self.Get_Shifted_Vertices(simp,l_id, ix, iy)
        trial_pt = np.array(pt)
        for i in range(3):
            c_i = HF.Curl(vertices[(i+1)%3]-vertices[i], trial_pt - vertices[i])
            if c_i < 0.0:
                return False
        return True

    def _Get_Edges(self, points, closed = True, end_pts_pin = [False,False] ):
        tree = KDTree(self.pointpos)
        _,nn = tree.query(points, k=1)
        simp_in = [self._Find_Simp(points[i],nn[i]) for i in range(len(nn))]
        edge_list = [] 
        ncl = 0
        if not closed: ncl = -1
        for i in range(len(points)+ncl):
            line_big = [points[i], points[(i+1)%len(points)]]
            simp_chain = self._Simp_Hop(points[(i+1)%len(points)], simp_in[i][0], line_big, simp_in[i][1], simp_in[i][2], simp_in[i][3])
            edge_list += [simp_chain[k][1] for k in range(len(simp_chain)-1)]
        HF.Reduce_List(edge_list)
        if not closed and not end_pts_pin == [False, False]:
            temp_edge_list = []
            if end_pts_pin[0]:
                st_pt = simp_in[0][0].edgeids.index(edge_list[0])
                temp_edge_list += simp_in[0][0].EdgeNeighbors(simp_in[0][0].points[st_pt])
            temp_edge_list += edge_list
            if end_pts_pin[1]:
                end_pt = simp_in[-1][0].edgeids.index(edge_list[-1])
                temp_edge_list += simp_in[-1][0].EdgeNeighbors(simp_in[-1][0].points[end_pt])
            edge_list = temp_edge_list + edge_list[::-1]
        return edge_list

    #find simp that contains
    def _Find_Simp(self, pt_in, nn_id):
        simp_set, l_id_set = self.pointlist[nn_id].SimpNeighbors(nn_id)
        for i in range(len(simp_set)):
            simp = simp_set[i]
            l_id = l_id_set[i]
            edge = simp.edgeids[l_id]
            line_big = [copy.copy(self.pointpos[nn_id]), pt_in]
            simp_chain = self._Simp_Hop(pt_in, simp, line_big, l_id, next_edge = edge)
            if simp_chain is not None:
                return [simp_chain[-1][0],simp_chain[-1][2], simp_chain[-1][3], simp_chain[-1][4]] 
        return None

    #####Plotting
    # PlotPrelims - the preliminary plotting settings.  returns the newly created figure and axes.
    def _PlotPrelims(self, PP: PrintParameters):
        szx = PP.FigureSizeX
        szy = szx
        if PP.Bounds is not None:
            szy = szx*(PP.Bounds[1][1] - PP.Bounds[0][1])/(PP.Bounds[1][0] - PP.Bounds[0][0])
            szy += 1.0/PP.dpi*(int(szy*PP.dpi)%2)  #szy*dpi must be even
        fig = plt.figure(figsize=(szx,szy), dpi=PP.dpi, frameon=False)
        ax = fig.gca()
        mplstyle.use('fast')
        rcParams['savefig.pad_inches'] = 0
        ax.autoscale(tight=True)
        if PP.Bounds is not None:
            ax.set_xlim((PP.Bounds[0][0], PP.Bounds[1][0]))
            ax.set_ylim((PP.Bounds[0][1], PP.Bounds[1][1]))
        else:
            ax.set_xlim(0,self.FDsizes[0])
            ax.set_ylim(0,self.FDsizes[1])
        ax.set_aspect('equal')
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        fig.tight_layout(pad=0)
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        PP._conversion_factor = self.FDsizes[0]/bbox.width/72
        return fig, ax    
    
    # TriangulationPlotBase - This plots the underlying triangulation
    def _TriangulationPlotBase(self, ax, PP: PrintParameters):
        xpoints = [x[0] for x in self.pointpos]
        ypoints = [x[1] for x in self.pointpos]
        trianglesIn, EdgeSimp = [], []
        for x in self.simplist:
            if x.relptregion == [[0,0],[0,0],[0,0]]:
                trianglesIn.append(x.points)
            else:
                EdgeSimp.append(x)
        #plots the triangles that are completely in the FD
        if len(trianglesIn) > 0:
            ax.triplot(xpoints, ypoints, trianglesIn, c=PP.linecolor_tri, lw=PP.linewidth_tri, zorder=1)
        #plots the triangles that stradle edge(s).  This plots mutiple copies of each so that the intersection of each with the FD is included
        lines = []
        for Simp in EdgeSimp:
            for i in range(3):
                nx, ny = Simp.relptregion[i]
                nxs = nx
                if nx == 0: nxs = 1
                else: nxs = -1*np.sign(nx)
                nys = ny
                if ny == 0: nys = 1
                else: nys = -1*np.sign(ny)
                xpts = np.array([self.pointpos[Simp.points[(i+1)%3]][0],self.pointpos[Simp.points[(i+2)%3]][0]])
                ypts = np.array([self.pointpos[Simp.points[(i+1)%3]][1],self.pointpos[Simp.points[(i+2)%3]][1]])
                for j in range(0,-nx+nxs,nxs):
                    for k in range(0,-ny+nys,nys):
                        xptsn = xpts + np.array([j,j+nx])*self.FDsizes[0]
                        yptsn = ypts + np.array([k,k+ny])*self.FDsizes[1]
                        lines.append([[xptsn[i],yptsn[i]] for i in range(2)])
        lc = LineCollection(lines, linewidths=PP.linewidth_tri, colors=PP.linecolor_tri, zorder=1)
        ax.add_collection(lc)


    # PointPlotBase - plots the points, with options for the size of the point and including the point label
    def _PointPlotBase(self, ax, PP: PrintParameters):
        xpoints = [x[0] for x in self.pointpos]
        ypoints = [x[1] for x in self.pointpos]
        if PP.ptlabels:
            for k in range(len(xpoints)):
                ax.annotate(k,(xpoints[k],ypoints[k]))
        ax.scatter(xpoints, ypoints, marker='o', s=PP.markersize, c='k', zorder=2)

    
    def _TTPlotBase(self, ax,LoopIn, PP: PrintParameters):
        EdgePlotted = [False for i in range(self._totalnumedges)]  #keeps track of segments that have been plotted (so as to not plot an element twice)
        ttpatches, cweights, line_widths = [], [], []
        if not PP.Delaunay:  #regular case, works for any triangulation
            for simp in self.simplist:
                new_ttpatches, new_cweights = self._GeneralSimplexTTPlot(simp, LoopIn, EdgePlotted, PP)
                ttpatches += new_ttpatches
                if PP.color_weights:
                    cweights += new_cweights             
        else:  #looks nicer, but only works for a Delaunay triangulation
            if not PP.DelaunayAdd:
                for simp in self.simplist:
                    new_ttpatches, new_cweights = self._DelaunaySimplexTTPlot(simp, LoopIn, EdgePlotted, PP)
                    ttpatches += new_ttpatches
                    if PP.color_weights:
                        cweights += new_cweights
            else:
                PP._max_weight = max(LoopIn.weightlist)
                for simp in self.simplist:
                    new_ttpatches, new_cweights, new_l_widths = self._DelaunaySimplexTTPlot_exp(simp, LoopIn, PP)
                    ttpatches += new_ttpatches
                    line_widths += new_l_widths
                    if PP.color_weights:
                        cweights += new_cweights
                
        Pcollection = PatchCollection(ttpatches, fc="none", alpha=PP.alpha_tt, capstyle = 'butt', joinstyle = 'round', zorder=3)
        if not PP.DelaunayAdd:  Pcollection.set_linewidth(PP.linewidth_tt)
        else:  Pcollection.set_linewidth(line_widths)
            
        if not PP.color_weights: Pcollection.set_edgecolor(PP.linecolor_tt)
        else:
            if PP.log_color: Pcollection.set_array(np.log(cweights))
            else: Pcollection.set_array(cweights)
            Pcollection.set_cmap(PP.color_map)
        
        ax.add_collection(Pcollection)        


    # GeneralSimplexTTPlot - plot the segments of train tracks that are determined from a given simplex
    def _GeneralSimplexTTPlot(self, simp, LoopIn, EdgePlotted, PP):
        patches_out = []
        weights_out = []
        W = [LoopIn.weightlist[eid] for eid in simp.edgeids] #edge weights
        delta = 1e-10
        srpr = np.array(simp.relptregion)
        FDs = np.array(self.FDsizes)
        if sum(W) > delta:  # if there are any weights to plot
            vertpts = np.array([self.pointpos[pts] for pts in simp.points]) #locations of the three simplex vertices
            #local id of the extra point in each of the 3 surrounding simplices
            exlids = [simp.simplices[i].edgeids.index(simp.edgeids[i]) for i in range(3)]
            #locations of the extra point in each of the 3 surrounding simplices
            exvertpts = np.array([self.pointpos[simp.simplices[i].points[exlids[i]]] for i in range(3)])
            #convention - first point in simp.points will be treated as being in the FD
            #now let's modify the positions of all the other points
            vertpts[1,:] += FDs*srpr[2,:]
            exvertpts[0,:] += FDs*srpr[2,:]
            vertpts[2,:] -= FDs*srpr[1,:]
            exvertpts[0,:] += FDs*np.array(simp.simplices[0].relptregion[(exlids[0]+1)%3])
            exvertpts[1,:] -= FDs*np.array(simp.simplices[1].relptregion[(exlids[1]+2)%3])
            exvertpts[2,:] += FDs*np.array(simp.simplices[2].relptregion[(exlids[2]+1)%3])
            #now let's get the simplex geometric centers and edge halfwaypoints
            SimpCenter = HF.GetCenter(vertpts.tolist())
            AdjSimpCenters = [HF.GetCenter([vertpts[(1+i)%3,:], exvertpts[i,:], vertpts[(2+i)%3,:]]) for i in range(3)]
            EdgeHalf = np.array([HF.GetCenter([vertpts[(1+i)%3],vertpts[(2+i)%3]]) for i in range(3)])
            #now the points that are halfway between the edge centers and the simpcenter
            CenterEdgeHalf = np.array([HF.GetCenter([SimpCenter, EdgeHalf[i,:]]) for i in range(3)])
            #now the points that are halfway between the edge centers and the adjacent simplex centers
            AdjEdgeHalf = np.array([HF.GetCenter([AdjSimpCenters[i], EdgeHalf[i]]) for i in range(3)])
            #check that the quadratic Bezier control triangle doesn't contain a vertex.  If so, we modify the control points
            for i in range(3):
                side = 2 #default is left
                C1 = HF.Curl( AdjEdgeHalf[i,:]-EdgeHalf[i,:] , EdgeHalf[i,:]-CenterEdgeHalf[i,:] )
                if C1 > 0: side = 1 #right
                Line1 = [CenterEdgeHalf[i,:], AdjEdgeHalf[i,:]]
                Line2 = [vertpts[(i+side)%3,:], EdgeHalf[i,:]]
                t1, t2 = HF.GetIntersectionTimes(Line1, Line2)
                if t2 < 0: #need to modify
                    alpha = -t2/(1-t2)
                    CenterEdgeHalf[i,:] = (1-alpha)*CenterEdgeHalf[i,:] + alpha*EdgeHalf[i,:]
                    AdjEdgeHalf[i,:] = (1-alpha)*AdjEdgeHalf[i,:] + alpha*EdgeHalf[i,:]
            Wp = [(W[(k+1)%3]+W[(k+2)%3]-W[k])/2 for k in range(3)]  #the interior weights
            if simp.relptregion == [[0,0],[0,0],[0,0]]:  # only need to print once
                for i in range(3):
                    if not EdgePlotted[simp.edgeids[i]]:
                        if W[i] > delta:
                            patches_out.append(HF.BezierQuad(CenterEdgeHalf[i,:], EdgeHalf[i,:], AdjEdgeHalf[i,:]))
                            if PP.color_weights: weights_out.append(W[i])
                        EdgePlotted[simp.edgeids[i]] = True
                    if Wp[i] > delta:
                        patches_out.append(HF.BezierQuad(CenterEdgeHalf[(i+1)%3,:], SimpCenter, CenterEdgeHalf[(i+2)%3,:]))
                        if PP.color_weights: weights_out.append(Wp[i])
            else:  #the simplex straddles edges
                ptregionx = [0,simp.relptregion[2][0], -1*simp.relptregion[1][0]]
                ptregiony = [0,simp.relptregion[2][1], -1*simp.relptregion[1][1]]
                for xint in range(-max(ptregionx),-min(ptregionx)+1):
                    for yint in range(-max(ptregiony),-min(ptregiony)+1):
                        dr = np.array([self.FDsizes[0]*xint, self.FDsizes[1]*yint])
                        #shift all the control points and plot again
                        SC = np.array(SimpCenter) + dr
                        CEH = CenterEdgeHalf + dr
                        AEH = AdjEdgeHalf + dr
                        EH = EdgeHalf + dr
                        for i in range(3):
                            if not EdgePlotted[simp.edgeids[i]] and W[i] > delta:
                                patches_out.append(HF.BezierQuad(CEH[i,:], EH[i,:], AEH[i,:]))
                                if PP.color_weights: weights_out.append(W[i])
                            if Wp[i] > delta:
                                patches_out.append(HF.BezierQuad(CEH[(i+1)%3,:], SC, CEH[(i+2)%3,:]))
                                if PP.color_weights: weights_out.append(Wp[i])
                for i in range(3):
                    EdgePlotted[simp.edgeids[i]] = True
        return patches_out, weights_out


    #used in other function to plot the segments of train tracks that are determined from a given simplex
    #this version assumes the triangulation is Delaunay, and uses the dual Voroni Centers as control points
    def _DelaunaySimplexTTPlot(self, simp, LoopIn, EdgePlotted, PP):
        patches_out = []
        weights_out = []
        W = [LoopIn.weightlist[eid] for eid in simp.edgeids] #edge weights
        delta = 1e-10
        srpr = np.array(simp.relptregion)
        FDs = np.array(self.FDsizes)
        if sum(W) > delta:  # if there are any weights to plot
            vertpts = np.array([self.pointpos[pts] for pts in simp.points]) #locations of the three simplex vertices
            #local id of the extra point in each of the 3 surrounding simplices
            exlids = [simp.simplices[i].edgeids.index(simp.edgeids[i]) for i in range(3)]
            #locations of the extra point in each of the 3 surrounding simplices
            exvertpts = np.array([self.pointpos[simp.simplices[i].points[exlids[i]]] for i in range(3)])
            #convention - first point in simp.points will be treated as being in the FD
            #now let's modify the positions of all the other points
            vertpts[1,:] += FDs*srpr[2,:]
            exvertpts[0,:] += FDs*srpr[2,:]
            vertpts[2,:] -= FDs*srpr[1,:]
            exvertpts[0,:] += FDs*np.array(simp.simplices[0].relptregion[(exlids[0]+1)%3])
            exvertpts[1,:] -= FDs*np.array(simp.simplices[1].relptregion[(exlids[1]+2)%3])
            exvertpts[2,:] += FDs*np.array(simp.simplices[2].relptregion[(exlids[2]+1)%3])
            #now let's get the simplex Voronoi centers and halfwaypoints
            SimpVCenter = HF.GetCircumCircleCenter(vertpts.tolist())
            AdjSimpVCenters = [HF.GetCircumCircleCenter([vertpts[(1+i)%3,:], exvertpts[i,:], vertpts[(2+i)%3,:]]) for i in range(3)]
            HalfVCs = [HF.GetCenter([SimpVCenter,AdjSimpVCenters[i]]) for i in range(3)] #halfway between Voronoi centers
            #now the points that partway (frac - default = 0.5) from Center voroni to HalfVCs
            FracControlPts_In = np.array([HF.LinFuncInterp(SimpVCenter, HalfVCs[i], PP.frac) for i in range(3)])
            FracControlPts_Out = np.array([HF.LinFuncInterp(AdjSimpVCenters[i], HalfVCs[i], PP.frac) for i in range(3)])
            Wp = [(W[(k+1)%3]+W[(k+2)%3]-W[k])/2 for k in range(3)]  #the interior weights
            Vpts_copy_loc = [0,0]
            Vpts_copy_loc[0] = Vpts_copy_loc[0] or np.floor(SimpVCenter[0]/self.FDsizes[0]).astype(int)
            Vpts_copy_loc[1] = Vpts_copy_loc[1] or np.floor(SimpVCenter[1]/self.FDsizes[1]).astype(int)
            for i in range(3):
                Vpts_copy_loc[0] = Vpts_copy_loc[0] or np.floor(HalfVCs[i][0]/self.FDsizes[0]).astype(int)
                Vpts_copy_loc[1] = Vpts_copy_loc[1] or np.floor(HalfVCs[i][1]/self.FDsizes[1]).astype(int)
            
            if simp.relptregion == [[0,0],[0,0],[0,0]] and not Vpts_copy_loc == [0,0]:  
            # only need to print once (but check for Voronoi center out of FD)
                for i in range(3):
                    if not EdgePlotted[simp.edgeids[i]]:
                        if W[i] > delta:
                            patches_out.append(HF.BezierLinear(FracControlPts_In[i,:], FracControlPts_Out[i,:]))
                            if PP.color_weights: weights_out.append(W[i])
                        EdgePlotted[simp.edgeids[i]] = True
                    if Wp[i] > delta:
                        patches_out.append(HF.BezierQuad(FracControlPts_In[(i+1)%3,:], SimpVCenter, FracControlPts_In[(i+2)%3,:]))
                        if PP.color_weights: weights_out.append(Wp[i])
            else:  #the simplex straddles edges or the voronoi center is out of the FD
                xstart, xstop, ystart, ystop = 0,0,0,0
                if simp.relptregion == [[0,0],[0,0],[0,0]]:
                    xstart = min(0,-Vpts_copy_loc[0])
                    xstop = max(0,-Vpts_copy_loc[0])
                    ystart = min(0,-Vpts_copy_loc[1])
                    ystop = max(0,-Vpts_copy_loc[1])
                else:
                    ptregionx = [0,simp.relptregion[2][0], -simp.relptregion[1][0]]
                    ptregiony = [0,simp.relptregion[2][1], -simp.relptregion[1][1]]
                    xstart = -max(ptregionx)
                    xstop = -min(ptregionx)
                    ystart = -max(ptregiony)
                    ystop = -min(ptregiony)
                for xint in range(xstart,xstop+1):
                    for yint in range(ystart,ystop+1):
                        dr = np.array([self.FDsizes[0]*xint, self.FDsizes[1]*yint])
                        #shift all the control points and plot again
                        SC = np.array(SimpVCenter) + dr
                        FCPI = FracControlPts_In + dr
                        FCPO = FracControlPts_Out + dr
                        for i in range(3):
                            if W[i] > delta: #and not EdgePlotted[simp.edgeids[i]]
                                patches_out.append(HF.BezierLinear(FCPI[i,:], FCPO[i,:]))
                                if PP.color_weights: weights_out.append(W[i])
                            if Wp[i] > delta:
                                patches_out.append(HF.BezierQuad(FCPI[(i+1)%3,:], SC, FCPI[(i+2)%3,:]))
                                if PP.color_weights: weights_out.append(Wp[i])
        return patches_out, weights_out


    #used in other function to plot the segments of train tracks that are determined from a given simplex
    #this version assumes the triangulation is Delaunay, and uses the dual Voroni Centers as control points
    #this is an experimental version, where I work on ideas before incorporating them into the main plotting
    def _DelaunaySimplexTTPlot_exp(self, simp, LoopIn, PP):
        patches_out = []
        weights_out = []
        line_weights_out = []
        W = [LoopIn.weightlist[eid] for eid in simp.edgeids] #edge weights
        delta = 1e-10
        srpr = np.array(simp.relptregion)
        FDs = np.array(self.FDsizes)
        if sum(W) > delta:  # if there are any weights to plot
            vertpts = np.array([self.pointpos[pts] for pts in simp.points]) #locations of the three simplex vertices
            #local id of the extra point in each of the 3 surrounding simplices
            exlids = [simp.simplices[i].edgeids.index(simp.edgeids[i]) for i in range(3)]
            #locations of the extra point in each of the 3 surrounding simplices
            exvertpts = np.array([self.pointpos[simp.simplices[i].points[exlids[i]]] for i in range(3)])
            #convention - first point in simp.points will be treated as being in the FD
            #now let's modify the positions of all the other points
            vertpts[1,:] += FDs*srpr[2,:]
            exvertpts[0,:] += FDs*srpr[2,:]
            vertpts[2,:] -= FDs*srpr[1,:]
            exvertpts[0,:] += FDs*np.array(simp.simplices[0].relptregion[(exlids[0]+1)%3])
            exvertpts[1,:] -= FDs*np.array(simp.simplices[1].relptregion[(exlids[1]+2)%3])
            exvertpts[2,:] += FDs*np.array(simp.simplices[2].relptregion[(exlids[2]+1)%3])
            #now let's get the simplex Voronoi centers and halfwaypoints
            SimpVCenter = HF.GetCircumCircleCenter(vertpts.tolist())
            AdjSimpVCenters = [HF.GetCircumCircleCenter([vertpts[(1+i)%3,:], exvertpts[i,:], vertpts[(2+i)%3,:]]) for i in range(3)]
            HalfVCs = [HF.GetCenter([SimpVCenter,AdjSimpVCenters[i]]) for i in range(3)] #halfway between Voronoi centers
            #now the points that partway (frac - default = 0.5) from Center voroni to HalfVCs
            FracControlPts_In = np.array([HF.LinFuncInterp(SimpVCenter, HalfVCs[i], PP.frac) for i in range(3)])
            FracControlPts_Out = np.array([HF.LinFuncInterp(AdjSimpVCenters[i], HalfVCs[i], PP.frac) for i in range(3)])
            Wp = [(W[(k+1)%3]+W[(k+2)%3]-W[k])/2 for k in range(3)]  #the interior weights
            W_scaled = []
            Wp_scaled = []
            for i in range(3):
                if W[i] <= PP._max_weight*PP.tt_lw_min_frac: W_scaled.append(PP.linewidth_tt*PP.tt_lw_min_frac)
                else: W_scaled.append(PP.linewidth_tt*(W[i]/PP._max_weight))
                if Wp[i] <= PP._max_weight*PP.tt_lw_min_frac: Wp_scaled.append(PP.linewidth_tt*PP.tt_lw_min_frac)
                else: Wp_scaled.append(PP.linewidth_tt*(Wp[i]/PP._max_weight))

            #now find the modified control points
            rmp90 = np.array([[0, -1],[1, 0]])
            rmm90 = np.array([[0, 1],[-1, 0]])
            FCP_m_center = FracControlPts_In - np.array(SimpVCenter)
            FCP_m_center_mag = np.hypot(FCP_m_center[:,0],FCP_m_center[:,1])
            displace_r = np.array([(W_scaled[i] - Wp_scaled[(i+1)%3])/2*PP._conversion_factor for i in range(3)])
            displace_l = np.array([(W_scaled[i] - Wp_scaled[(i+2)%3])/2*PP._conversion_factor for i in range(3)])
            FCP_m_center_rotp = np.array([np.dot(rmp90,FCP_m_center[i,:]) for i in range(3)])
            FCP_m_center_rotm = np.array([np.dot(rmm90,FCP_m_center[i,:]) for i in range(3)])
            scaling_l = displace_l/FCP_m_center_mag
            scaling_r = displace_r/FCP_m_center_mag
            delta_vec_l = np.array([FCP_m_center_rotp[i]*scaling_l[i] for i in range(3)])
            delta_vec_r = np.array([FCP_m_center_rotm[i]*scaling_r[i] for i in range(3)])
            FCP_mod_l =  delta_vec_l+ FracControlPts_In
            FCP_mod_r = delta_vec_r + FracControlPts_In
            Center_mod_l = delta_vec_l + np.array(SimpVCenter)
            Center_mod_r = delta_vec_r + np.array(SimpVCenter)
            HalfVCs_mod_l = delta_vec_l + np.array(HalfVCs)
            HalfVCs_mod_r = delta_vec_r + np.array(HalfVCs)

            center_m = np.array([HF.GetIntersectionPoint([FCP_mod_r[(i+2)%3], Center_mod_r[(i+2)%3]], [FCP_mod_l[(i+1)%3], Center_mod_l[(i+1)%3]] ) for i in range(3)])
            control_points = np.array([[HalfVCs_mod_r[(i+2)%3],FCP_mod_r[(i+2)%3], center_m[i] , FCP_mod_l[(i+1)%3],HalfVCs_mod_l[(i+1)%3] ] for i in range(3)])
            

            Vpts_copy_loc = [0,0]
            Vpts_copy_loc[0] = Vpts_copy_loc[0] or np.floor(SimpVCenter[0]/self.FDsizes[0]).astype(int)
            Vpts_copy_loc[1] = Vpts_copy_loc[1] or np.floor(SimpVCenter[1]/self.FDsizes[1]).astype(int)
            for i in range(3):
                Vpts_copy_loc[0] = Vpts_copy_loc[0] or np.floor(HalfVCs[i][0]/self.FDsizes[0]).astype(int)
                Vpts_copy_loc[1] = Vpts_copy_loc[1] or np.floor(HalfVCs[i][1]/self.FDsizes[1]).astype(int)
            
            if simp.relptregion == [[0,0],[0,0],[0,0]] and not Vpts_copy_loc == [0,0]:  
            # only need to print once (but check for Voronoi center out of FD)
                for i in range(3):
                    if Wp[i] > delta:
                        patches_out.append(HF.BezierCustom(control_points[i,0,:], control_points[i,1,:], control_points[i,2,:], control_points[i,3,:], control_points[i,4,:]))
                        if PP.color_weights: weights_out.append(Wp[i])
                        line_weights_out.append(Wp_scaled[i])
            else:  #the simplex straddles edges or the voronoi center is out of the FD
                xstart, xstop, ystart, ystop = 0,0,0,0
                if simp.relptregion == [[0,0],[0,0],[0,0]]:
                    xstart = min(0,-Vpts_copy_loc[0])
                    xstop = max(0,-Vpts_copy_loc[0])
                    ystart = min(0,-Vpts_copy_loc[1])
                    ystop = max(0,-Vpts_copy_loc[1])
                else:
                    ptregionx = [0,simp.relptregion[2][0], -simp.relptregion[1][0]]
                    ptregiony = [0,simp.relptregion[2][1], -simp.relptregion[1][1]]
                    xstart = -max(ptregionx)
                    xstop = -min(ptregionx)
                    ystart = -max(ptregiony)
                    ystop = -min(ptregiony)
                for xint in range(xstart-1,xstop+2):
                    for yint in range(ystart-1,ystop+2):
                        dr = np.array([self.FDsizes[0]*xint, self.FDsizes[1]*yint])
                        #shift all the control points and plot again
                        for i in range(3):
                            if Wp[i] > delta:
                                patches_out.append(HF.BezierCustom(control_points[i,0,:]+dr, control_points[i,1,:]+dr, control_points[i,2,:]+dr, control_points[i,3,:]+dr, control_points[i,4,:]+dr))
                                if PP.color_weights: weights_out.append(Wp[i])
                                line_weights_out.append(Wp_scaled[i])
        return patches_out, weights_out, line_weights_out


    # TriCopy creates a copy of the current triangulation object
    # If EvolutionReset is True, atstep is set to 0 and we do not copy the weight operators
    def TriCopy(self, EvolutionReset = True):
        #first create an empty triangulation object (to be returned at the end)
        TriC = triangulation2D([], None, empty = True)
        if not EvolutionReset: TriC._atstep = self._atstep
        TriC.FDsizes = copy.deepcopy(self.FDsizes)
        TriC._ptnum = self._ptnum
        TriC.pointpos = copy.deepcopy(self.pointpos)
        TriC._pointposfuture = copy.deepcopy(self._pointposfuture)
        TriC._totalnumedges = self._totalnumedges
        TriC._Vec = self._Vec
        #create the simplist
        TriC.simplist = []
        for i in range(len(self.simplist)):
            TriC.simplist.append(simplex2D(self.simplist[i].points))
            TriC.simplist[-1].edgeids = copy.copy(self.simplist[i].edgeids)
            TriC.simplist[-1].SLindex = i
            TriC.simplist[-1].relptregion = copy.deepcopy(self.simplist[i].relptregion)
        #now create the links
        for i in range(len(self.simplist)):
            for j in range(3):
                if self.simplist[i].simplices[j] is not None:
                    TriC.simplist[i].simplices[j] = TriC.simplist[self.simplist[i].simplices[j].SLindex]
        #now fill the pointlist
        TriC.pointlist = []
        for i in range(len(self.pointlist)):
            TriC.pointlist.append(TriC.simplist[self.pointlist[i].SLindex])           
        #if we do not reset the evolution related items, we copy them
        TriC.WeightOperatorList = []
        if not EvolutionReset:
            for i in range(len(self.WeightOperatorList)):
                TriC.WeightOperatorList.append( WeightOperator( copy.copy( self.WeightOperatorList[i].es)))
                TriC.WeightOperatorList[-1].time = self.WeightOperatorList[i].time   
        return TriC