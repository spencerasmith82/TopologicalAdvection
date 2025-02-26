from abc import ABC, abstractmethod
import copy
import math
import HelperFns as HF
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter
from dataclasses import dataclass, field

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
    def __init__(self, tri, rbands = None, curves = None, Shear = False, mesh = False):
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
            curve is a list with two elements: the first is a list of point 
            positions (each point position is a list: [x,y]), and the second 
            is a boolean (is_closed) which if signals that the point position 
            list should wrap-around (True), or be considered a curve with end-
            points (False). The list of point positions define a sequence of 
            lines connected end to end.

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
            Generally, avoid loops which trasversely intersect eachother, as 
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
                # represents bands pinned to adjacent points in this triangulation.
            self.Shear = True # mesh mush be evaluated with shear coordinates
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
                # This first creates a regular loop (regular coordinates), then
                # feeds this into the triangulation object to get the shear coordinates
                tri.ShearWeightsInitialize(RegLoop, self)
                       
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
            LoopIn.weightlist[self.eids[0]] = max(WL[1]+WL[3],WL[2]+WL[4]) - WL[0]
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

#need to decide what to include in the PrintParameters 
@dataclass
class PrintParameters_Base:
    filename: str = None
    triplot: bool = True
    Delaunay: bool = False
    Bounds: list = None
    FigureSize: list = field(default_factory=lambda: [8, 8])
    dpi: int = 200
    ptlabels: bool = False
    markersize: float = 2.0
    linewidth_tri: float = 0.5
    linewidth_tt: float = 1.0
    linecolor_tri: str = 'g'
    linecolor_tt: str = 'r'
    alpha_tt: float = 1.0
    frac: float = 0.8


# triangulation2D_Base Class #################################################

#This is the triangulation class, the central class in the overall algorithm. It is initialized using a Delaunay triangulation
class triangulation2D_Base(ABC):

    #The constructor for triangulation2D.  ptlist is the list of [x,y] positions for the points at the initial time.
    #Reminder that the input points are just in the fundamental domain.  We also have the size of the fundamental domain as [0,Dx) and [0,Dy) in the x and y directions repectively.  Important that the points are in this domain.  We also pass in Dx and Dy.  There are no control points, as this will be a triangulation without boundary.
    def __init__(self, ptlist, empty = False):
        
        self.atstep = 0
        self.pointlist = None
        #self.ptnum = 0
        self.pointpos = None
        self.pointposfuture = None
        self.simplist = []
        self.totalnumedges = 0
        self.WeightOperatorList = []
        self.Vec = True
        if len(ptlist) < 10: # for small number of points, the non-vectorized version of a few functions will be faster
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

    # Evolve. Main method to evolve the state of the triangulation forward in time.  This assumes that the starting triangulation is good (no negative areas).  It takes in the new time-slice data in ptlist -- the list of [x,y] positions for the points at the next time-step.  If Maintain_Delaunay is True (False is default), then after all of the collapse events are acounted for, extra edge flips will be used to ensure that the triangulation is Delaunay

    def Evolve(self, ptlist, Maintain_Delaunay = False):
        self.LoadNewPos(ptlist)
        EventLists = self.GetEvents() 
        self.GEvolve(EventLists)  #Deals with the events in CollapseEventList and CrossingList in order
        self.UpdatePtPos()
        self.atstep += 1
        if Maintain_Delaunay:
            self.MakeDelaunay() #after the atstep increment so that the operators will have the correct time-stamp


    @abstractmethod
    def LoadNewPos(self, ptlist):
        pass

    @abstractmethod
    def GetEvents(self):
        pass

    #this returns a list of current simplices (each element is [simplex, first time for A = 0]) whose area goes through zero sometime between their current and future positions.
    def GetCollapseEvents(self):
        collapsesimplist = []
        if self.Vec:  
            AZT_bool, AZT_time = self.AreaZeroTimeMultiple()
            collapsesimplist = [[self.simplist[i],AZT_time[i]] for i in range(len(self.simplist)) if AZT_bool[i]]    
        else:
            for simp in self.simplist:
                AZT = self.AreaZeroTimeSingle(simp)
                if AZT[0]:
                    collapsesimplist.append([simp,AZT[1]])  
        collapsesimplist.sort(key=itemgetter(1), reverse=True)  # this is in decending order so that removing from the end(smallest times first) inccurs the smallest computational cost
        return collapsesimplist

    @abstractmethod
    def AreaZeroTimeMultiple(self):
        pass

    @abstractmethod
    def AreaZeroTimeSingle(self, simp):
        pass

    def UpdatePtPos(self):
        self.pointpos = self.pointposfuture

    @abstractmethod
    def GEvolve(self, EventLists):
        pass

    #Fixing a simplex and the surrounding effected simplices.  SimpIn is actually a list [simplex,area zero time].  This returns the two new simplices, so that they can be possibly added to the local event list, also the bad simplex so it can be removed (if needed from the local event list)
    def SFix(self,SimpIn,timein):
        Simp = SimpIn[0]
        colind = self.CollapsePt(Simp,SimpIn[1])  #this is the local index of the offending point during the area collapse
        Topsimp = Simp.simplices[colind]
        edge_id = Simp.edgeids[colind]
        globaltime = self.atstep + timein
        newsimps = self.EdgeFlip([Simp,Topsimp], edge_id, globaltime)  #this does most of the work in flipping the edge and cleaning up linking
        #finally, return the two new simplices, so that they can be checked to see if they need to be included in any update to the local event list. Also return the bad simplex to remove any instance from the event list.
        return [newsimps,Topsimp]
        

    #This returns the point (internal id) that passes through its opposite edge during an area collapse event known to occur at t = tcol
    def CollapsePt(self, SimpIn, tcol):
        #first get the positions of the 3 points at the time of collapse
        colpos = self.GetSimpCurrentLoc(SimpIn, tcol)
        d0 = (colpos[2][0] - colpos[0][0])*(colpos[1][0]-colpos[0][0]) + (colpos[2][1] - colpos[0][1])*(colpos[1][1] - colpos[0][1])  
        #This is the dot product of (z2-z0) and (z1-z0) ... < 0 if 0 is the middle point
        if d0 < 0:  return 0
        else:
            d1 = (colpos[2][0] - colpos[1][0])*(colpos[0][0]-colpos[1][0]) + (colpos[2][1] - colpos[1][1])*(colpos[0][1] - colpos[1][1])
            if d1 < 0:  return 1
            else:  return 2   #don't need to calculate the last dot product.  If the first two are >0, this must be <0
    
    @abstractmethod
    def GetSimpCurrentLoc(self, SimpIn, tcol):
        pass

    @abstractmethod
    def EdgeFlip(self, AdjSimps, EdgeShare, TimeIn = None):
        pass
    
        
    @abstractmethod
    def MakeDelaunay(self):
        pass
    

    # OperatorAction takes the accumulated operator list stored in WeightOperatorList and operates sucessively on the given Loop
    #the start and stop index can also be specified to break this up into stages (only used for option 1 and 2)
    #Reverse does the operator actions in reverse order (i.e. for loops in the final triangulation)
    #option 1 just changes the data in the loop object, option 2 also accumulates a weight list with the total weights after each operator has acted on the loop, and gives the global time of the operator action. Option 3 (the default) returns a weight list which has the weights at the end of each time step (the intervals between each use of the Evolve method).  This weight list does not have the time listed, as this is only know externally. this is most useful for producing a list that we can directly tie to an external list of times.  This is what we need for extracting the topological entropy (hence the default option)
    def OperatorAction(self, LoopIn, index = None, Reverse = False, option = 3):
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
                    WeightList.append([LoopIn.GetWeightTotal(), self.WeightOperatorList[i].time])
            else:
                finaltime = math.ceil(self.WeightOperatorList[i].time)
                WeightList.append([LoopIn.GetWeightTotal(), finaltime])
                for i in range(endind,startind-1,-1):
                    self.WeightOperatorList[i].Update(LoopIn, Reverse)
                    WeightList.append([LoopIn.GetWeightTotal(), self.WeightOperatorList[i].time])
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
            return WeightList
        else:
            print("Need to choose one of the options 1, 2, or 3")


    # this takes the regular edge weights (for some band) encoded in RegLoop, and uses the triangulation connectivity to create LoopIn, which represents the band in shear coordinates
    def ShearWeightsInitialize(self, RegLoop, LoopIn):
        for simp in self.simplist:
            for i in range(3):
                #LoopIn must be initialized to all zeros (this catches the second time through)
                if LoopIn.weightlist[simp.edgeids[i]] == 0:
                    #if the value for the regular loop is zero here, then the shear coordinates should be zero (already initialized as zero)
                    if not RegLoop.weightlist[simp.edgeids[i]] == 0:
                        WA = RegLoop.weightlist[simp.edgeids[(i+1)%3]]
                        WB = RegLoop.weightlist[simp.edgeids[(i+2)%3]]
                        xsimp = simp.simplices[i]
                        Lid = xsimp.LocalID(simp.points[(i+2)%3])
                        WC = RegLoop.weightlist[xsimp.edgeids[Lid]]
                        WD = RegLoop.weightlist[xsimp.edgeids[(Lid+1)%3]]
                        LoopIn.weightlist[simp.edgeids[i]] = (-WA+WB-WC+WD)//2

    
    #This initializes the edge weights that correspond to a given band (or set of bands).
    #If an EdgeWeightList is passed in, then the weights are added to the appropriate spot in this list, and are not added to the edge weights in the triangulation
    def BandWeightInitialize(self,rbands, LoopIn, wadd = 1.0):
        for i in range(len(rbands)):
            numpoints = len(rbands[i])
            AreAdjacent = []
            CurveLeft = []
            for k in range(numpoints):
                AreAdjacent.append(self.ArePointsAdjacent(rbands[i][k], rbands[i][(k+1)%numpoints]))
                triplepts = [rbands[i][(k+numpoints-1)%numpoints], rbands[i][k], rbands[i][(k+1)%numpoints]]
                CurveLeft.append(self.DoesCurveLeft(triplepts))

            for j in range(numpoints):
                Bool1 = [CurveLeft[j], AreAdjacent[j], CurveLeft[(j+1)%numpoints]]
                Bool2 = [AreAdjacent[(j+numpoints-1)%numpoints], CurveLeft[j], AreAdjacent[j]]
                triplepts = [rbands[i][(j+numpoints-1)%numpoints], rbands[i][j], rbands[i][(j+1)%numpoints]]                
                self.AddWeightsAlongLine([rbands[i][j],rbands[i][(j+1)%numpoints]], Bool1, LoopIn, wadd)
                self.AddWeightsAroundPoint(triplepts, Bool2, LoopIn, wadd)

    @abstractmethod
    def ArePointsAdjacent(self,pt1,pt2):
        pass

    
    def DoesCurveLeft(self,pttriple):
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
        return self.DoesCurveLeft(pttriple)
        

    #This takes the two points in linepoints and adds a weight of one (or non-default value) to any edges that are crossed
    #by the line.
    def AddWeightsAlongLine(self,linepoints, Boolin, LoopIn, wadd = 1.0):
        pt1, pt2 = linepoints
        if Boolin[1][0]: #this is the case of adjacent points (i.e. the line between the points is an edge)
            #only if the curvelefts' (Boolin[0], Boolin[2]) are opposite one another, do we add a weight
            if Boolin[0] is not Boolin[2]:
                pt1rtlocid = Boolin[1][1][1].LocalID(pt1)
                edgeindex = Boolin[1][1][1].edgeids[(pt1rtlocid+1)%3]
                LoopIn.weightlist[edgeindex] += wadd
        else:
            #first we need to determine the direction (which simplex) to set out from that has pt1 as a point.
            stlocid, StartSimp = self.SimpInDir([pt1,pt2])
            endlocid, EndSimp = self.SimpInDir([pt2,pt1])
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
    def SimpInDir(self,linepoints):
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
    def AddWeightsAroundPoint(self, pttriple, Boolin, LoopIn, wadd = 1):
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
            stlocid, StartSimp = self.SimpInDir([pt2,pt1])
        endlocid, EndSimp = None, None
        if Boolin[2][0]:
            if not Boolin[1]:
                EndSimp = Boolin[2][1][0]
            else:
                EndSimp = Boolin[2][1][1]
            endlocid = EndSimp.LocalID(pt2)   
        else:
            endlocid, EndSimp = self.SimpInDir([pt2,pt3])
        edgeindex = StartSimp.edgeids[(stlocid+indadd)%3]
        LoopIn.weightlist[edgeindex] += wadd
        CurrentSimp = StartSimp.simplices[(stlocid+indadd)%3]
        ptloc = CurrentSimp.LocalID(pt2)
        while not CurrentSimp is EndSimp:
            edgeindex = CurrentSimp.edgeids[(ptloc+indadd)%3]
            LoopIn.weightlist[edgeindex] += wadd
            CurrentSimp = CurrentSimp.simplices[(ptloc+indadd)%3]
            ptloc = CurrentSimp.LocalID(pt2)    


    def CurveWeightInitialize(self, curves, LoopIn, wadd = 1.0):
        for curve in curves:
            point_set, is_closed = curve
            edge_list = self.Get_Edges(point_set, is_closed)
            for edge in edge_list:
                LoopIn.weightlist[edge] += wadd

    @abstractmethod
    def Get_Edges(self, points, closed = True):
        pass

###########Plotting

    def Plot(self, LoopIn = None, PP: PrintParameters = PrintParameters()):
        fig, ax = self.PlotPrelims(PP)  # the preliminary plotting settings
        if PP.triplot:  self.TriangulationPlotBase(ax, PP)  # the underlying triangulation
        self.PointPlotBase(ax, PP)  # the points
        if LoopIn is not None:  self.TTPlotBase(ax, LoopIn, PP) #only plot the traintrack if a Loop is given
        if PP.filename is None:  plt.show()
        else:  plt.savefig(PP.filename)
        plt.close()

    @abstractmethod
    def PlotPrelims(self, PP: PrintParameters):
        pass
    
    @abstractmethod
    def TriangulationPlotBase(self, ax, PP: PrintParameters):
        pass
        
    @abstractmethod
    def PointPlotBase(self, ax, PP: PrintParameters):
        pass

    @abstractmethod
    def TTPlotBase(self, ax, LoopIn, PP: PrintParameters):
        pass





