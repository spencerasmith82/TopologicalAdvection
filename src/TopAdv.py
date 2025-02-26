from TopAdvBase import simplex2D_Base, Loop, WeightOperator, triangulation2D_Base, PrintParameters_Base
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

# simplex2D class ############################################################
class simplex2D(simplex2D_Base):
    """
    Child class simplex2D extensions/notes:

    Methods (additional notes)
    -------
    SimpNeighbors(IDin)
        In the case of a boundary simplex (with None as one/two of the simplex
        neighbors), the list is not necessarily in CCW order.
    """
    
    def __init__(self, IDlist):
        """
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
        if simplex2D_Base._count == 0:
            self.__class__.__doc__ = simplex2D_Base.__doc__ + "\n" + self.__class__.__doc__
        super().__init__(IDlist)    

    def SimpNeighbors(self, IDin):
        """Finds the simpices which share a point.

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
        lsimp = self.simplices[(stpt+1)%3]
        while (not self is lsimp) and (lsimp is not None):
            NeighborList.append(lsimp)
            lsimp_lid = lsimp.LocalID(IDin)
            LocalIDList.append(lsimp_lid)
            lsimp = lsimp.simplices[(lsimp_lid+1)%3]
        if lsimp is None:  #this deals with the boundary simplex case
            rsimp = self.simplices[(stpt+2)%3]
            while (not self is rsimp) and (rsimp is not None):
                NeighborList.append(rsimp)
                rsimp_lid = rsimp.LocalID(IDin)
                LocalIDList.append(rsimp_lid)
                rsimp = rsimp.simplices[(rsimp_lid+2)%3]
        return NeighborList, LocalIDList


    def SimpLink(self,S_other):
        """
        Links this simplex with S_other (and vice versa).
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
            locid_s2 = (S_other.points.index(self.points[(locid_s1 + 1)%3]) + 1)%3
            S_other.simplices[locid_s2] = self
# End of simplex2D class #####################################################


#need to decide what to include in the PrintParameters 
@dataclass
class PrintParameters(PrintParameters_Base):
    color_weights: bool = False
    log_color: bool = True
    color_map: str = 'inferno_r'



# triangulation2D Class ######################################################
#This is the triangulation class, the central class in the overall algorithm. It is initialized using a Delaunay triangulation
class triangulation2D(triangulation2D_Base):
    #The constructor for triangulation2D.  ptlist is the list of [x,y] positions for the points at the initial time.
    #Domain = [[x_min,y_min],[x_max,y_max]] The data point must be contained within this rectangular boundary at all times.
    def __init__(self, ptlist, Domain = None, empty = False):
        self.extranum = 24  # the number of extra boundary points to use (must be even)
        self.extrapoints = []
        if not empty: self.SetControlPoints(Domain, ptlist)
        super().__init__(ptlist, empty)

    
    def SetControlPoints(self, Domain, ptlist):
        #modify set control
        if controlptlist is not None:
            self.extranum = len(controlptlist)  #user input number of extra boundary points (could be any number)
        self.ptnum = len(ptlist) + self.extranum   #put in the number of points, since this does not change, and will be used often. 
        #calculate the extra boundary points
        self.extrapoints = []
        if controlptlist is None:
            temppoints0 = np.array(ptlist)
            ptcenter = np.average(temppoints0,axis = 0)   #the center (averaged)
            ptrange = (np.max(temppoints0, axis = 0) - np.min(temppoints0, axis = 0))/2.0   #the bounding box half widths
            rad = np.max(ptrange)   #effective bounding radius
            sfactor = 3  #the factor used to scale up the bounding points ***Using a really large factor here seems to cause the Delaunay trangulation function below to give bad output.  Will keep any bounding points not too far away, though we will need to make sure (in later additions to the code) that the points won't wander outside of these bounds at some-point in the full trajectory data.
            sfactor2 = 4
            numcirc = int(self.extranum/2)
            #now add extranum/2 points around a circle of radius sfactor*rad starting at theta = 0
            self.extrapoints = [[ptcenter[0] + sfactor*rad*math.cos(2*math.pi*i/numcirc), ptcenter[1] + sfactor*rad*math.sin(2*math.pi*i/numcirc)] for i in range(numcirc)]
            #now add another extranum/2 points around a larger circle of radius sfactor2*rad starting at 2*pi/(2*numcirc) (i.e. rotated to be angularly half-way between any pair of the first set of points)
            angoffset = 2*math.pi/(2*numcirc)
            self.extrapoints = self.extrapoints + [[ptcenter[0] + sfactor2*rad*math.cos(2*math.pi*i/numcirc + angoffset), ptcenter[1] + sfactor2*rad*math.sin(2*math.pi*i/numcirc + angoffset)] for i in range(0,numcirc)]
        else:
            self.extrapoints = controlptlist

    
    def LoadPos(self,ptlist):
        self.pointpos = ptlist + self.extrapoints  #combining the bounding simplex points with the regular points' positions (adding the extra point to the end rather than the begining is dicated by plotting concerns later)

    def SetInitialTriangulation(self):
        temppoints = np.array(self.pointpos)  #put the total point set (regular plus large bounding simplex) into numpy array form
        temptri = Delaunay(temppoints,qhull_options="QJ Pp")   #create the initial Delaunay triangulation.  The option forces the creation of simplices for degenerate points by applying a random perturbation.

        #Now we need to store the triangulation data in a local data structure 
        numsimp = temptri.simplices.shape[0]   #the number of simplices in the triangulation
        self.simplist = []

        #first create the list of simplex2D objects (not linked together yet ... need to create every object first)
        for i in range(numsimp):
            self.simplist.append(simplex2D(temptri.simplices[i].tolist()))
            self.simplist[-1].SLindex = i

        #now create the links
        for i in range(numsimp):
            linklist = temptri.neighbors[i].tolist()
            for j in range(0,len(linklist)):
                if not linklist[j] == -1:
                    self.simplist[i].simplices[j] = self.simplist[linklist[j]]    #if -1 then the simplex already points to None (true for neighbors of boundary simplices)

        self.SetPointList()
        self.SetEdgeIds()


    def SetPointList(self):
        #now create the pointlist with links to individual simplices
        #first initialize the list
        self.pointlist = []
        for i in range(self.ptnum):
            self.pointlist.append(None)
        #now go through each simplex and add that simplex to each slot in the pointlist that corresponds to an included point if the slot contains None (possibly more efficient way to do this)
        for i in range(len(self.simplist)):
            for j in range(3):
                 if self.pointlist[self.simplist[i].points[j]] is None:
                        self.pointlist[self.simplist[i].points[j]] = self.simplist[i]

    def SetEdgeIds(self):
        #Now we assign each edge an index (will do this for all options for consistency).  This goes through each simplex object, and assigns an id to each edge (and the same id to the corresponding edge in the adjacent simplex) if it has not already been assigned.  The index is just taken from an incremental counter.
        edgecounter = 0
        for i in range(0,len(self.simplist)):
            for j in range(0,3):
                tempsimp = self.simplist[i]
                if tempsimp.edgeids[j] is None:
                    tempsimp.edgeids[j] = edgecounter
                    if not tempsimp.simplices[j] is None:
                        pt = tempsimp.points[(j+1)%3]
                        Lid = (tempsimp.simplices[j].LocalID(pt)+1)%3
                        tempsimp.simplices[j].edgeids[Lid] = edgecounter
                    edgecounter += 1  
        self.totalnumedges = edgecounter 


    def LoadNewPos(self, ptlist):
        self.pointposfuture = ptlist + self.extrapoints


    def GetEvents(self):
        return self.GetCollapseEvents()
    

    #this function returns a pair [IsSoln, timeOut], where IsSoln is a boolian that is true if the first time at which the area goes through zero is between Tin and 1, and false if not.  For IsSoln == True, timeOut gives this time.
    def AreaZeroTimeSingle(self,SimpIn,Tin = 0):
        #first get the beginning and end x,y coordinate for each of the three points
        ptlist = SimpIn.points
        IP0x, IP0y = self.pointpos[ptlist[0]]
        IP1x, IP1y = self.pointpos[ptlist[1]]
        IP2x, IP2y = self.pointpos[ptlist[2]]
        FP0x, FP0y = self.pointposfuture[ptlist[0]]
        FP1x, FP1y = self.pointposfuture[ptlist[1]]
        FP2x, FP2y = self.pointposfuture[ptlist[2]]
        return  HF.AreaZeroTimeBaseSingle(IP0x, IP0y, IP1x, IP1y, IP2x, IP2y, FP0x, FP0y, FP1x, FP1y, FP2x, FP2y, Tin)

    #this goes through every simplex and looks for whether the area zero time is between Tin and 1.  
    def AreaZeroTimeMultiple(self,Tin = 0):
        nsimps = len(self.simplist)
        pts0 = np.array([self.simplist[i].points[0] for i in range(nsimps)])
        pts1 = np.array([self.simplist[i].points[1] for i in range(nsimps)])
        pts2 = np.array([self.simplist[i].points[2] for i in range(nsimps)])
        npptpos = np.array(self.pointpos)
        npptposf = np.array(self.pointposfuture)
        IP0x, IP0y = npptpos[pts0,0], npptpos[pts0,1]
        IP1x, IP1y = npptpos[pts1,0], npptpos[pts1,1]
        IP2x, IP2y = npptpos[pts2,0], npptpos[pts2,1]
        FP0x, FP0y = npptposf[pts0,0], npptposf[pts0,1]
        FP1x, FP1y = npptposf[pts1,0], npptposf[pts1,1]
        FP2x, FP2y = npptposf[pts2,0], npptposf[pts2,1]
        return HF.AreaZeroTimeBaseVec(IP0x, IP0y, IP1x, IP1y, IP2x, IP2y, FP0x, FP0y, FP1x, FP1y, FP2x, FP2y, Tin)


        #The main method for evolving the Event List (group of simplices that need fixing)
    #remember that the list is sorted in decending order, and we deal with the last element first
    def GEvolve(self,EventLists):
        delta = 1e-10
        while len(EventLists)> 0:            
            neweventssimp = []  #new simpices to check
            dellistsimp = []    #simplices to delete from the simplex event list if they exist in it
            currenttime = EventLists[-1][1]
            #deal with simplex collapse events here
            modlist = self.SFix(EventLists[-1], currenttime - delta)    #returns ... [[leftsimp,rightsimp],topsimp (old)]
            neweventssimp = modlist[0]
            delsimp = modlist[1]
            del EventLists[-1]  #get rid of the evaluated event
            #first find the time of zero area for core simplex event, and delete it if needed
            AZT = self.AreaZeroTimeSingle(delsimp, currenttime - delta)
            if AZT[0]:
                HF.BinarySearchDel(EventLists, [delsimp,AZT[1]])
            #now run through the newevents list and see if each object goes through zero area in the remaining time (if so, add to EventList with the calulated time to zero area)
            for i in range(0,len(neweventssimp)):
                AZT = self.AreaZeroTimeSingle(neweventssimp[i], currenttime - delta)
                if AZT[0]:
                    #insert in the event list at the correct spot
                    HF.BinarySearchIns(EventLists,[neweventssimp[i],AZT[1]]) 

    
    #this returns the linearly interpolated positions of each point in ptlist (usually 3, but can handle other lengths) at the time 0 < teval < 1.
    def GetSimpCurrentLoc(self, SimpIn, teval):
        ptlist = SimpIn.points
        return [self.GetCurrentLoc(pt_ind, teval) for pt_ind in ptlist]

    def GetCurrentLoc(self, pt_ind, teval):
        posi = self.pointpos[pt_ind]
        posf = self.pointposfuture[pt_ind]
        return [((posf[k]-posi[k])*teval + posi[k]) for k in range(2)]


    def EdgeFlip(self, AdjSimps, EdgeShare, TimeIn = None):
        #first get the local ids of the points not shared by these simplices
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
        rsimp = simplex2D(rslist)  #new right simplex
        lsimp = simplex2D(lslist)  #new left simplex
        #create the list of edge ids for the weight operator
        WeightIDs = [EdgeShare, Topsimp.edgeids[lptuid], Topsimp.edgeids[rptuid], Simp.edgeids[rptlid], Simp.edgeids[lptlid]]
        #create the weight operater and append to the list
        self.WeightOperatorList.append(WeightOperator(WeightIDs,TimeIn))

        #create the links these simplices have to other simplices
        rsimp.SimpLink(Topsimp.simplices[lptuid])
        lsimp.SimpLink(Topsimp.simplices[rptuid])
        rsimp.SimpLink(Simp.simplices[lptlid])
        lsimp.SimpLink(Simp.simplices[rptlid])
        rsimp.SimpLink(lsimp)
        
        #also need to reassign the weight ids
        rsimp.edgeids[0] = WeightIDs[1]  #for all of these, we know which points the local ids correspond to
        rsimp.edgeids[1] = WeightIDs[0]
        rsimp.edgeids[2] = WeightIDs[4]
        lsimp.edgeids[0] = WeightIDs[2]
        lsimp.edgeids[1] = WeightIDs[3]
        lsimp.edgeids[2] = WeightIDs[0]
        
        #replace the two bad simplices in the simplex list with the two new ones
        Simpindex = Simp.SLindex
        self.simplist[Simpindex] = lsimp
        lsimp.SLindex = Simpindex
        
        Topsimpindex = Topsimp.SLindex
        self.simplist[Topsimpindex] = rsimp
        rsimp.SLindex = Topsimpindex
                
        #look through the simplex point list to see if either of the bad simplices were there and replace if so
        if self.pointlist[bpt] is Simp:
            self.pointlist[bpt] = rsimp
        if (self.pointlist[rpt] is Simp) or (self.pointlist[rpt] is Topsimp):
            self.pointlist[rpt] = rsimp
        if self.pointlist[tpt] is Topsimp:
            self.pointlist[tpt] = lsimp
        if (self.pointlist[lpt] is Simp) or (self.pointlist[lpt] is Topsimp):
            self.pointlist[lpt] = lsimp
        
        #Next, delete all the references to simplices in both of the bad simplices
        for i in range(3):
            Simp.simplices[i] = None
            Topsimp.simplices[i] = None        
        
        return [lsimp,rsimp]    
    
      
        



    # MakeDelaunay takes the current triangulation and, through a series of edge flips, changes it into the Delaunay triangulation for this point configuration.  This function changes the underlying triangulation
    def MakeDelaunay(self,single = False):
        IsD, InteriorEdge, EdgeBSimps = None, None, None
        if not single:  #vectorized version (this one makes marginal improvements)
            IsD, InteriorEdge, EdgeBSimps = self.IsDelaunay()
        else:
            EdgeBSimps = [None for i in range(self.totalnumedges)]
            EdgeUsed = [False for i in range(self.totalnumedges)]
            IsD = [False for i in range(self.totalnumedges)]
            for simp in self.simplist:
                for j in range(3):
                    edgeid = simp.edgeids[j]
                    if not EdgeUsed[edgeid] and not simp.simplices[j] is None:
                        EdgeUsed[edgeid] = True
                        EdgeBSimps[edgeid] = [[simp,simp.simplices[j]],edgeid, True]
                        IsD[edgeid] = self.IsLocallyDelaunay([simp,simp.simplices[j]])
            InteriorEdge = EdgeUsed
            
        EdgeList = [EdgeBSimps[i] for i in range(self.totalnumedges) if IsD[i] == False and InteriorEdge[i]]
        EdgeList_Epos = [None for i in range(self.totalnumedges)]
        for i in range(len(EdgeList)):
            EdgeList_Epos[EdgeList[i][1]] = i
        #now go through the edge list and start flipping edges
        while len(EdgeList) > 0:
            EdgeSimps, edge_id, checked = EdgeList.pop()
            EdgeList_Epos[edge_id] = None
            Flip = True
            if not checked:
                Flip = not self.IsLocallyDelaunay(EdgeSimps)
            if Flip:
                LRsimps = self.EdgeFlip(EdgeSimps, edge_id ,self.atstep)
                for i in range(2): # Left and right simplices
                    loc = LRsimps[i].edgeids.index(edge_id)
                    lrsimp = LRsimps[i]
                    for j in range(2): # upper and lower simplices
                        eid = lrsimp.edgeids[(loc+1+j)%3]
                        if InteriorEdge[eid]:
                            adjsimp = lrsimp.simplices[(loc+1+j)%3]
                            ELinsert = [[lrsimp,adjsimp], eid, False]
                            if EdgeList_Epos[eid] == None:
                                EdgeList_Epos[eid] = len(EdgeList)
                                EdgeList.append(ELinsert)
                            else:
                                EdgeList[EdgeList_Epos[eid]] = ELinsert    


    
    # IsDelaunay outputs an array (length = number of edges) of booleans, which indicate if the quadrilateral with the ith edge as a diagonal is Delaunay. Also outputs an array of the pairs of simplices which bound each edge. This calls IsDelaunayBase (which is outside the current class) for a jit speed-up
    def IsDelaunay(self):
        Ax, Ay, Bx, By, Cx, Cy, Dx, Dy = [np.zeros(self.totalnumedges) for i in range(8)]
        EdgeUsed = [False for i in range(self.totalnumedges)]
        BoundingSimps = [None for i in range(self.totalnumedges)]
        for simp in self.simplist:
            for j in range(3):
                edgeid = simp.edgeids[j]
                if not EdgeUsed[edgeid] and not simp.simplices[j] is None:
                    EdgeUsed[edgeid] = True
                    Apt = simp.points[(j+2)%3]
                    Ax[edgeid], Ay[edgeid] = self.pointpos[Apt]
                    Bpt = simp.points[j]
                    Bx[edgeid], By[edgeid] = self.pointpos[Bpt]
                    Cpt = simp.points[(j+1)%3]
                    Cx[edgeid], Cy[edgeid] = self.pointpos[Cpt]
                    adjsimp = simp.simplices[j]
                    BoundingSimps[edgeid] = [[simp,adjsimp], edgeid ,True]
                    adjsimp_loc_id = adjsimp.edgeids.index(edgeid)
                    Dpt = adjsimp.points[adjsimp_loc_id]
                    Dx[edgeid], Dy[edgeid] = self.pointpos[Dpt]
        return HF.IsDelaunayBaseWMask(Ax,Ay,Bx,By,Cx,Cy,Dx,Dy,np.array(EdgeUsed)), EdgeUsed, BoundingSimps
    

    #given the two adjecent simplices, determine if the configuration is locally Delaunay.  Returns True or False
    def IsLocallyDelaunay(self,AdjSimps):
        simp1 = AdjSimps[0]
        simp2 = AdjSimps[1]
        locid = simp1.simplices.index(simp2)
        Apt = simp1.points[(locid+2)%3]
        Ax, Ay = self.pointpos[Apt]
        Bpt = simp1.points[locid]
        Bx, By = self.pointpos[Bpt]
        Cpt = simp1.points[(locid+1)%3]
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
        detvals = ADx*(BDy*CD2-CDy*BD2) - ADy*(BDx*CD2-CDx*BD2) + AD2*(BDx*CDy-CDx*BDy)
        return (detvals < 0) 


    
    def ArePointsAdjacent(self,pt1,pt2):
        AreAdjacent = False
        goodind = (len(self.pointlist) - self.extranum)-1
        IsBndryEdge = False
        if pt1 > goodind:
            if pt2 > goodind:
                IsBndryEdge = True
            else:
                temppt = pt1
                pt1 = pt2
                pt2 = pt1
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


    #using ref pt, see if point is in simp.  if not, find edge intersection and then 
    # get adj simp, find ref point and dx,dy for shift that matches edge from simp.  Then calls self recursively
    # stops when simp is found with this point interior.  returns list of pairs [simp, edge] along path
    def Simp_Hop(self, pt_in, simp, line_big, edge_prev = None, next_edge = None):
        delta = 1e-8
        #first see if pt_in is in the simp
        if self.Tri_Contains(pt_in, simp):
            return [[simp, None]]
        else:
            vertices = np.array([copy.copy(self.pointpos[p]) for p in simp.points])
            next_id = None
            if next_edge is not None:
                i = simp.edgeids.index(next_edge)
                Line_simp = [vertices[(i+1)%3],vertices[(i+2)%3]]
                if HF.IsIntersection(line_big, Line_simp):
                    next_id = i
            else: 
                if edge_prev is None:
                    for i in range(3):
                        Line_simp = [vertices[(i+1)%3],vertices[(i+2)%3]]
                        if HF.IsIntersection(line_big, Line_simp):
                            next_id = i
                            break
                else:
                    for i in range(3):
                        if not simp.edgeids[i] == edge_prev:
                            Line_simp = [vertices[(i+1)%3],vertices[(i+2)%3]]
                            if HF.IsIntersection(line_big, Line_simp):
                                next_id = i
                                break
            if next_id is None: return None
            edge = simp.edgeids[next_id]
            next_simp = simp.simplices[next_id]
            return [[simp, edge]] + self.Simp_Hop(pt_in, next_simp, line_big, edge_prev = edge)
                                 
        
    
    def Tri_Contains(self, pt, simp):
        vertices = np.array([copy.copy(self.pointpos[p]) for p in simp.points])
        trial_pt = np.array(pt)
        for i in range(3):
            c_i = HF.Curl(vertices[(i+1)%3]-vertices[i], trial_pt - vertices[i])
            if c_i < 0.0:
                return False
        return True


    def Get_Edges(self, points, closed = True):
        tree = KDTree(self.pointpos)
        _,nn = tree.query(points, k=1)
        simp_in = [self.Find_Simp(points[i], nn[i]) for i in range(len(nn))]
        edge_list = [] 
        ncl = 0
        if not closed: ncl = -1
        for i in range(len(points)+ncl):
            line_big = [points[i], points[(i+1)%len(points)]]
            simp_chain = self.Simp_Hop(points[(i+1)%len(points)], simp_in[i], line_big)
            edge_list += [simp_chain[k][1] for k in range(len(simp_chain)-1)]
        HF.Reduce_List(edge_list)
        return edge_list


    #find simp that contains
    def Find_Simp(self, pt_in, nn_id):
        simp_set, l_id_set = self.pointlist[nn_id].SimpNeighbors(nn_id)
        for i in range(len(simp_set)):
            simp = simp_set[i]
            l_id = l_id_set[i]
            edge = simp.edgeids[l_id]
            line_big = [copy.copy(self.pointpos[nn_id]), pt_in]
            simp_chain = self.Simp_Hop(pt_in, simp, line_big, next_edge = edge)
            if simp_chain is not None:
                return simp_chain[-1][0]
        return None


    #####Plotting
        # PlotPrelims - the preliminary plotting settings.  returns the newly created figure and axes.
    def PlotPrelims(self, PP: PrintParameters):
        szx, szy = PP.FigureSize
        fig = plt.figure(figsize=(szx,szy), dpi=PP.dpi, frameon=False)
        ax = fig.gca()
        rcParams['savefig.pad_inches'] = 0
        #rcParams['path.simplify_threshold'] = 1.0  #to speed up plotting ... set smaller if needing higher quality
        ax.autoscale(tight=True)
        if PP.Bounds is not None:
            ax.set_xlim((PP.Bounds[0][0], PP.Bounds[1][0]))
            ax.set_ylim((PP.Bounds[0][1], PP.Bounds[1][1]))
        ax.set_aspect('equal')
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        return fig, ax

    
    # TriangulationPlotBase - This plots the underlying triangulation
    def TriangulationPlotBase(self, ax, PP: PrintParameters):
        xpoints = [x[0] for x in self.pointpos[:len(self.pointpos)-self.extranum]]  #note that we exclude the bounding points
        ypoints = [x[1] for x in self.pointpos[:len(self.pointpos)-self.extranum]]
        triangles = [x.points for x in self.simplist if (len(set(x.points).intersection([(len(self.pointpos)-y) for y in range(1,self.extranum+1)])) == 0)]  #make sure that the list of triangles (triplets of points) do not include the excluded large triangle points
        ax.triplot(xpoints, ypoints, triangles, c=PP.linecolor_tri, lw=PP.linewidth_tri, zorder=1)

    
# PointPlotBase - plots the points (excluding the bounding points), with options for the size of the point and including the point label
    def PointPlotBase(self, ax, PP: PrintParameters):
        xpoints = [x[0] for x in self.pointpos[:len(self.pointpos)-self.extranum]]  #note that we exclude the bounding points
        ypoints = [x[1] for x in self.pointpos[:len(self.pointpos)-self.extranum]]
        if PP.ptlabels:
            for k in range(len(xpoints)):
                ax.annotate(k,(xpoints[k],ypoints[k]))
        ax.scatter(xpoints, ypoints, marker='o', s=PP.markersize, c='k', zorder=2)


    def TTPlotBase(self, ax, LoopIn, PP: PrintParameters):
        EdgePlotted = [False for i in range(self.totalnumedges)]  #keeps track of segments that have been plotted (so as to not plot an element twice)
        ttpatches = []
        if not PP.Delaunay:  #regular case, works for any triangulation
            for simp in self.simplist:
                if not None in simp.simplices:
                    ttpatches += self.GeneralSimplexTTPlot(simp, LoopIn, EdgePlotted)
        else:  #looks nicer, but only works for a Delaunay triangulation
            for simp in self.simplist:
                if not None in simp.simplices:
                    ttpatches += self.DelaunaySimplexTTPlot(simp, LoopIn, EdgePlotted, PP)
        Pcollection = PatchCollection(ttpatches, ec=PP.linecolor_tt, fc="none", lw = PP.linewidth_tt, capstyle = 'butt', joinstyle = 'round', zorder=3)
        ax.add_collection(Pcollection) 

    
    # GeneralSimplexTTPlot - plot the segments of train tracks that are determined from a given simplex
    def GeneralSimplexTTPlot(self, simp, LoopIn, EdgePlotted):
        patches_out = []
        W = [LoopIn.weightlist[eid] for eid in simp.edgeids] #edge weights
        delta = 1e-10
        if sum(W) > delta:  # if there are any weights to plot
            vertpts = np.array([self.pointpos[pts] for pts in simp.points]) #locations of the three simplex vertices
            #local id of the extra point in each of the 3 surrounding simplices
            exlids = [simp.simplices[i].edgeids.index(simp.edgeids[i]) for i in range(3)]
            #locations of the extra point in each of the 3 surrounding simplices
            exvertpts = np.array([self.pointpos[simp.simplices[i].points[exlids[i]]] for i in range(3)])
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
            for i in range(3):
                if not EdgePlotted[simp.edgeids[i]]:
                    if W[i] > delta:
                        patches_out.append(HF.BezierQuad(CenterEdgeHalf[i,:], EdgeHalf[i,:], AdjEdgeHalf[i,:]))
                    EdgePlotted[simp.edgeids[i]] = True
                if Wp[i] > delta:
                    patches_out.append(HF.BezierQuad(CenterEdgeHalf[(i+1)%3,:], SimpCenter, CenterEdgeHalf[(i+2)%3,:]))
        return patches_out


#used in other function to plot the segments of train tracks that are determined from a given simplex
    #this version assumes the triangulation is Delaunay, and uses the dual Voroni Centers as control points
    def DelaunaySimplexTTPlot(self, simp, LoopIn, EdgePlotted, PP):
        patches_out = []
        W = [LoopIn.weightlist[eid] for eid in simp.edgeids] #edge weights
        delta = 1e-10
        if sum(W) > delta:  # if there are any weights to plot
            vertpts = np.array([self.pointpos[pts] for pts in simp.points]) #locations of the three simplex vertices
            #local id of the extra point in each of the 3 surrounding simplices
            exlids = [simp.simplices[i].edgeids.index(simp.edgeids[i]) for i in range(3)]
            #locations of the extra point in each of the 3 surrounding simplices
            exvertpts = np.array([self.pointpos[simp.simplices[i].points[exlids[i]]] for i in range(3)])
            #now let's get the simplex Voronoi centers and halfwaypoints
            SimpVCenter = HF.GetCircumCircleCenter(vertpts.tolist())
            AdjSimpVCenters = [HF.GetCircumCircleCenter([vertpts[(1+i)%3,:], exvertpts[i,:], vertpts[(2+i)%3,:]]) for i in range(3)]
            HalfVCs = [HF.GetCenter([SimpVCenter,AdjSimpVCenters[i]]) for i in range(3)] #halfway between Voronoi centers
            #now the points that partway (frac - default = 0.5) from Center voroni to HalfVCs
            FracControlPts_In = np.array([HF.LinFuncInterp(SimpVCenter, HalfVCs[i], PP.frac) for i in range(3)])
            FracControlPts_Out = np.array([HF.LinFuncInterp(AdjSimpVCenters[i], HalfVCs[i], PP.frac) for i in range(3)])
            Wp = [(W[(k+1)%3]+W[(k+2)%3]-W[k])/2 for k in range(3)]  #the interior weights
            for i in range(3):
                if not EdgePlotted[simp.edgeids[i]]:
                    if W[i] > delta:
                        patches_out.append(HF.BezierLinear(FracControlPts_In[i,:], FracControlPts_Out[i,:]))
                    EdgePlotted[simp.edgeids[i]] = True
                if Wp[i] > delta:
                    patches_out.append(HF.BezierQuad(FracControlPts_In[(i+1)%3,:], SimpVCenter, FracControlPts_In[(i+2)%3,:]))
        return patches_out