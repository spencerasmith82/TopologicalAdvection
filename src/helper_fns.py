"""Module for all of the stand-alone helper functions.

None of these functions rely on any of the topological advection classes.
"""

import sys
import numpy as np
import math
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from numba import jit


def progressBar(count_value, total, suffix=''):
    """Progress bar for tracking the progress of triangulation evolution."""
    bar_length = 100
    filled_up_Length = int(round(bar_length * count_value / float(total)))
    percentage = round(100.0 * count_value / float(total), 1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percentage, '%', suffix))
    sys.stdout.flush()


def GetMinMaxXYTrajVals(Tslices):
    """Find the minimum and maximum x & y values in a trajectory set."""
    TS = np.array(Tslices)
    x_max = np.max(TS[:, :, 0])
    x_min = np.min(TS[:, :, 0])
    y_max = np.max(TS[:, :, 1])
    y_min = np.min(TS[:, :, 1])
    return [[x_min, y_min], [x_max, y_max]]


def GetBoundingDomainSlice(ptlist, frac=0.2):
    """Find the bounding domain of a time slice.

    With the option to add some padding with frac.
    """
    ptlist_temp = np.array(ptlist)
    x_max = np.max(ptlist_temp[:, 0])
    x_min = np.min(ptlist_temp[:, 0])
    y_max = np.max(ptlist_temp[:, 1])
    y_min = np.min(ptlist_temp[:, 1])
    dx = x_max - x_min
    dy = y_max - y_min
    x_pad = dx*frac
    y_pad = dy*frac
    return [[x_min - x_pad, y_min - y_pad], [x_max + x_pad, y_max + y_pad]]


def GetBoundingDomainTraj(Tslices, PeriodicBC=False, frac=None):
    """Find the bounding domain of a trajectory set."""
    BD = GetMinMaxXYTrajVals(Tslices)
    if PeriodicBC:
        BD[0] = [0, 0]
        return BD
    else:
        x_pad, y_pad = 0, 0
        Deltax = BD[1][0] - BD[0][0]
        Deltay = BD[1][1] - BD[0][1]
        if frac is None:
            npts = len(Tslices[0])
            a_ratio = Deltax/Deltay
            npts_x = int(np.sqrt(npts*a_ratio))
            npts_y = int(npts/npts_x)
            x_pad = Deltax/(npts_x-1)
            y_pad = Deltay/(npts_y-1)
        else:
            x_pad = Deltax*frac
            y_pad = Deltay*frac
        BD[0][0] -= x_pad
        BD[1][0] += x_pad
        BD[0][1] -= y_pad
        BD[1][1] += y_pad
        return BD


def TriEdgeCrossBnd(triangle, boundary):
    """Do the triangle edges crosse one of the domain boundaries."""
    BoundaryPoints = [[boundary[0][0], boundary[0][1]],
                      [boundary[1][0], boundary[0][1]],
                      [boundary[1][0], boundary[1][1]],
                      [boundary[0][0], boundary[1][1]]]
    for i in range(3):
        Line1 = [triangle[i], triangle[(i+1) % 3]]
        for j in range(len(BoundaryPoints)):
            Line2 = [BoundaryPoints[j],
                     BoundaryPoints[(j+1) % len(BoundaryPoints)]]
            if IsIntersection(Line1, Line2):
                return True
    return False


def CounterToStr(countin):
    """Return a str version of the input int (with consistent padding)."""
    if countin < 10:
        return "000"+str(countin)
    elif countin < 100:
        return "00"+str(countin)
    elif countin < 1000:
        return "0"+str(countin)
    elif countin < 10000:
        return str(countin)
    else:
        return "countertoobig"


def BinarySearch(ListIn, TimeIn):
    """Binary search on a sorted list.

    BinarySearch does a binary search on a given sorted list (each element is
    a list of length 2, were the second item is the ordering parameter).  The
    list is assumed to be in decending order (smallest last).  The item that
    is searched for is a single number  - TimeIn (time to zero area).  The
    search is over the time variable.  The index i of ListIn such that
    ListIn[i][1] > TimeIn and ListIn[i+1][1] <= TimeIn is returned.  It is
    assumed that the case of TimeIn > ListIn[0][1] and TimeIn < ListIn[-1][1]
    have already been filtered out and delt with
    """
    Lindex, Rindex = 0, len(ListIn) - 1  # starting left and right indices
    while Rindex - Lindex > 1:
        # middle index, see if our item is in the left or right interval
        Mindex = (Rindex+Lindex)//2
        if TimeIn < ListIn[Mindex][1]:
            Lindex = Mindex
        else:
            Rindex = Mindex
    return Lindex, Rindex


def BinarySearchDel(ListIn, ItemIn):
    """Binary Search Delete.

    BinarySearchDel does a binary search on a given sorted list (each element
    is a list of length 2, were the second item is the ordering parameter).
    The list is assumed to be in decending order (smallest last).  The item
    that is searched for is also a double [event,time to zero area].  The
    search is over the time variable, but the event variable is used for
    direct comparison. If a match is found, then it is deleted from the list.
    """
    delta = 1e-8  # defining window of uncertainty for the time
    Left_Time = ItemIn[1] + delta
    Right_Time = ItemIn[1] - delta
    matchindex = 0
    success = False
    if Left_Time < ListIn[0][1] and Right_Time > ListIn[-1][1]:
        k, _ = BinarySearch(ListIn, Left_Time)
        k += 1
        while Right_Time < ListIn[k][1]:
            if ListIn[k][0] is ItemIn[0]:
                success = True
                matchindex = k
                break
            k += 1
    elif Left_Time > ListIn[0][1] and Right_Time < ListIn[0][1]:
        k = 0
        while Right_Time < ListIn[k][1]:
            if ListIn[k][0] is ItemIn[0]:
                success = True
                matchindex = k
                break
            k += 1
    elif Left_Time > ListIn[-1][1] and Right_Time < ListIn[-1][1]:
        k = -1
        while Left_Time > ListIn[k][1]:
            if ListIn[k][0] is ItemIn[0]:
                success = True
                matchindex = k
                break
            k -= 1
    if success:
        del ListIn[matchindex]
    else:
        print("did not delete item from EventList, event was not found")
        print("Item In = ", ItemIn)


def BinarySearchIns(ListIn, ItemIn):
    """Binary Search Insert.

    BinarySearchIns does a binary search on a given sorted list (each element
    is a double, were the second item is the ordering parameter).  The list is
    assumed to be in decending order (smallest last).  The item that is
    searched for is also a double [event,time to zero area].  The binary
    search finds the adjacent pair of elements inbetween which the input
    item's time fits.  If such a pair is found, then the ItemIn is inserted
    into this position.  Edge Case: If there is an item (or items) with the
    same time as the input item, the the input item is inserted to the left
    (lower index) of the item(s).
    """
    if len(ListIn) == 0:  # empty list, just add the item
        ListIn.append(ItemIn)
    elif ItemIn[1] < ListIn[-1][1]:  # item has smallest time, add to end
        ListIn.append(ItemIn)
    elif ItemIn[1] >= ListIn[0][1]:  # item has largest time, add to front
        ListIn.insert(0, ItemIn)
    else:
        Lindex, Rindex = BinarySearch(ListIn, ItemIn[1])
        # found an adjacent pair of items between which we can
        # insert the input item
        if Rindex - Lindex == 1:
            ListIn.insert(Rindex, ItemIn)
        else:
            # right and left indices are concurrent. This can happen when
            # ItemIn has an identical time to one of the items in ListIn.
            # These are either the same object (in which case we don't insert),
            # or not.
            if not ItemIn[0] is ListIn[Rindex][0]:
                # there can be two different simplices with exactly the same
                # time (degenerate triangles spanning the domain), this
                # compares the two simplices and adds them if they are
                # different
                ListIn.insert(Rindex, ItemIn)


def Reduce_List(List_In):
    """Remove adjacent repeats in the list.

    Reduce_List takes a simple list of indices and removes any adjacent
    repeats (recursively, and with wrap-around boundary conditions for the
    list).  This is used to tighten up a list of edges crossed by a closed
    curve.  If the curve crosses an edge twice without any other crossings
    inbetween then the curve can slide across this edge (thus removing two
    recorded crossings)
    """
    again = False
    for i in range(len(List_In)):
        if List_In[i] == List_In[(i+1) % len(List_In)]:
            max_ind = max(i, (i+1) % len(List_In))
            min_ind = min(i, (i+1) % len(List_In))
            del List_In[max_ind]
            del List_In[min_ind]
            again = True
            break
    if again:
        Reduce_List(List_In)


def IsIntersection(Line1, Line2, timeinfo=False):
    """Do the two line segments intersect eachother.

    IsIntersection takes in two lines (each defined by two points) and outputs
    True if they intersect (between each of their point pairs). If the flag
    for time info is True, then we also output the time t1.  So Line 1 is one
    whose parameterized intersection time is returned.  The time is the
    fraction of the line from Line1[0] to Line1[1]
    """
    IsInt = False
    D1x = Line1[1][0]-Line1[0][0]  # line 1 difference in x endpoints
    D1y = Line1[1][1]-Line1[0][1]  # same for y
    D2x = Line2[1][0]-Line2[0][0]  # line 2 difference in x endpoints
    D2y = Line2[1][1]-Line2[0][1]  # same for y
    # determinant.  If == 0, then the lines are parallel,
    # and there is no intersection to find
    det = D2y*D1x - D1y*D2x
    t1, t2 = 0, 0
    if not det == 0:  # lines intersect
        bx = (Line1[0][1]*Line1[1][0] - Line1[1][1]*Line1[0][0])
        by = (Line2[0][1]*Line2[1][0] - Line2[1][1]*Line2[0][0])
        xout = (bx*D2x - by*D1x)/det
        yout = (bx*D2y - by*D1y)/det
        # find intersection time for line 1 (t1)
        if abs(D1x) > abs(D1y):
            t1 = (xout - Line1[0][0])/(D1x)
        else:
            t1 = (yout - Line1[0][1])/(D1y)
        if not (t1 > 1 or t1 < 0):
            # The intersection is between the endpoints of line 1,
            # so look for intersection time for line 2 (t2)
            if abs(D2x) > abs(D2y):
                t2 = (xout - Line2[0][0])/(D2x)
            else:
                t2 = (yout - Line2[0][1])/(D2y)
            # check if itersection is also between endpoints of line 2.
            if not (t2 > 1 or t2 < 0):
                IsInt = True
    if timeinfo:
        # return whether there is a good intersection, and the time
        return [IsInt, t1]
    else:
        return IsInt  # return just whether there is a good intersection


def AreaZeroTimeBaseSingle(aix, aiy, bix, biy, cix, ciy, afx, afy,
                           bfx, bfy, cfx, cfy, Tin=0):
    """Time of collapse for a simplex.

    AreaZeroTimeBaseSingle outputs a list of two items: the time that the
    simplex crosses zero signed area (collapsed), and a boolean indicating
    whether this time was "good" - i.e. inbetween Tin and 1.  In the case
    of two times that are good, this returns the smaller one.
    The input values are the x and y positions of the three points bounding
    the simplex (both initial and final), and Tin.
    This solves a quadratic in the time variable.
    """
    # Some intermediate variables to help cut down the number of
    # algebraic operations needed
    caix = cix-aix
    caiy = ciy-aiy
    baix = bix-aix
    baiy = biy-aiy
    cafix = cfx-afx-caix
    cafiy = cfy-afy-caiy
    bafix = bfx-afx-baix
    bafiy = bfy-afy-baiy
    # these are the three coefficients for the quadratic equation:
    # a*t**2 + b*t + c
    a = cafix*bafiy - cafiy*bafix
    b = caix*bafiy - caiy*bafix + cafix*baiy - cafiy*baix
    c = caix*baiy - caiy*baix
    # this analytically solves for the smallest real root of the
    # quadratic equation that is between Tin and 1
    if a == 0:
        if b == 0:
            return [False, None]
        else:
            t = -c/b
            if t < 1 and t > Tin:
                return [True, -c/b]
            else:
                return [False, None]
    else:
        q = b*b - 4*a*c  # discriminant of the quadratic solution
        if q >= 0:
            # for a != 0 and non-negative discriminant there are
            # two real roots (possibly degenerate).
            # the largest magnitude root to help with numerical stability
            # (no catastrophic cancellation)
            t1 = (-b-math.copysign(1, b)*math.sqrt(q))/(2*a)
            t2 = c/(a*t1)  # the other root using t1*t2 = c/a
            t1ok, t2ok = False, False
            # checking to see if either of the roots is "good", i.e.
            # between the bounds Tin and 1.
            if t1 < 1 and t1 > Tin:
                t1ok = True
            if t2 < 1 and t2 > Tin:
                t2ok = True
            # different outputs based on whether the roots are good
            if (not t1ok) and (not t2ok):
                return [False, None]  # both times are not in the interval
            elif t1ok and t2ok:
                # both times are in the interval, returning smallest time.
                return [True, min(t1, t2)]
            elif t1ok:
                return [True, t1]  # only t1 is in the interval
            else:
                return [True, t2]  # only t2 is in the interval
        else:
            return [False, None]  # case of complex roots


@jit(nopython=True)
def AreaZeroTimeBaseVec(aix, aiy, bix, biy, cix, ciy, afx, afy,
                        bfx, bfy, cfx, cfy, Tin=0):
    """Time of collapse for every simplex.

    AreaZeroTimeBaseVec outputs two np arrays (length = number of simplices):
    the time that the simplices crosses zero signed area (collapsed), and a
    boolean indicating whether this time was "good" - i.e. inbetween Tin
    and 1.  In the case of two times that are good, this returns the smaller
    one.

    The input are np arrays for the x and y positions of the three points
    bounding each initial and final simplex, and Tin.
    This solves a quadratic in the time variable
    This version is vectorized and uses jit for a speed up
    """
    # Some intermediate variables to help cut down the number of
    # algebraic operations needed
    caix = cix-aix
    caiy = ciy-aiy
    baix = bix-aix
    baiy = biy-aiy
    cafix = cfx-afx-caix
    cafiy = cfy-afy-caiy
    bafix = bfx-afx-baix
    bafiy = bfy-afy-baiy
    # these are the three coefficients for the quadratic equation:
    # a*t**2 + b*t + c
    a = cafix*bafiy - cafiy*bafix
    b = caix*bafiy - caiy*bafix + cafix*baiy - cafiy*baix
    c = caix*baiy - caiy*baix
    # replacing the if statements with masks
    maz = (a == 0)  # mask for a == 0
    maznbz = maz & ~(b == 0)  # mask for a == 0 and not b == 0
    q = b*b - 4*a*c   # discriminant of the quadratic solution
    mqgz = ~maz & (q >= 0)  # not a == 0 and non-negative discriminant
    # getting the roots
    # initialization of array of roots (two rows for two
    # possible positive roots)
    roots = -1*np.ones((2, len(a)))
    # single solution for when a == 0 and b != 0
    roots[0, maznbz] = -c[maznbz]/b[maznbz]
    # for a != 0 and non-negative discriminant there are two real roots
    # (possibly degenerate).  This first obtains the largest magnitude root
    # to help with numerical stability (no catastrophic cancellation)
    roots[0, mqgz] = (-b[mqgz]-np.sign(b[mqgz])*np.sqrt(q[mqgz]))/(2*a[mqgz])
    # this obtains the other root t1*t2 = c/a
    roots[1, mqgz] = c[mqgz]/(a[mqgz]*roots[0, mqgz])
    # masks to determine "good" roots
    # mask for first root being between our time bounds
    t1ok = (roots[0] > Tin) & (roots[0] < 1)
    # same for second root (note that roots was initialized with a
    # value out of the bounds)
    t2ok = (roots[1] > Tin) & (roots[1] < 1)
    GoodSoln = t1ok | t2ok  # at least one good solution
    BothGood = t1ok & t2ok  # both solutions are good - need to choose smallest
    # only second solution is good - case where we need to put
    # solution 2 in the place of 1
    Onlyt2Good = ~t1ok & t2ok
    # smallest solution in the case of 2 good ones
    roots[0, BothGood] = np.minimum(roots[0, BothGood], roots[1, BothGood])
    # putting solution 2 in the place of 1 when only 2 is a good solution
    roots[0, Onlyt2Good] = roots[1, Onlyt2Good]
    # outputting vector of good solutions and their times
    return GoodSoln, roots[0, :]


@jit(nopython=True)
def CopyLocations(posix, posiy, posfx, posfy, Dx, Dy):
    """Find the copy locations for the final points.

    CopyLocations outputs an array (length = number of points) of ids
    (values 0-9). An id identifies the copy of the fundamental domain that
    the final position is in. Takes in numpy vectors for the initial and final
    positions (x and y), the domain width and height convention for the copy
    ids: 0 (FD/middle center), 1 (lower left), 2 (lower center), 3 (lower
    right), 4 (middle right), 5 (upper right), 6 (upper center), 7 (upper
    left), 8 (middle left)
    """
    # initialize copy location array (default is zero, as most
    # points don't exit the FD)
    copyloc = np.zeros(len(posix))
    Deltax = posfx-posix
    Deltay = posfy-posiy
    Deltaxabs = np.abs(Deltax)
    Deltayabs = np.abs(Deltay)
    checkx = Deltaxabs > Dx/2
    checky = Deltayabs > Dy/2
    checkpoints = checkx | checky
    mask = checkpoints.copy()
    signx = Deltax[checkpoints] > 0
    signy = Deltay[checkpoints] > 0
    checkx = checkx[checkpoints]
    checky = checky[checkpoints]
    iscorner = checkx & checky
    ycross = ~checkx & checky
    xcross = checkx & ~checky
    nsignx = ~signx
    nsigny = ~signy
    iscorner_nsignx = iscorner & nsignx
    iscorner_signx = iscorner & signx
    mask[checkpoints] = iscorner_signx & signy
    copyloc[mask] = 1
    mask[checkpoints] = ycross & signy
    copyloc[mask] = 2
    mask[checkpoints] = iscorner_nsignx & signy
    copyloc[mask] = 3
    mask[checkpoints] = xcross & nsignx
    copyloc[mask] = 4
    mask[checkpoints] = iscorner_nsignx & nsigny
    copyloc[mask] = 5
    mask[checkpoints] = ycross & nsigny
    copyloc[mask] = 6
    mask[checkpoints] = iscorner_signx & nsigny
    copyloc[mask] = 7
    mask[checkpoints] = xcross & signx
    copyloc[mask] = 8
    return copyloc


@jit(nopython=True)
def IsDelaunayBase(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy):
    """Is the quadrilateral Delaunay.

    IsDelaunayBase is a vectorized (+jit) function which returns a True/False
    array corresponding to whether the input quadrilateral (points A, B, C,
    D in ccw order, and diagonal edge A-C) is Delaunay.
    """
    ADx = Ax - Dx
    ADy = Ay - Dy
    BDx = Bx - Dx
    BDy = By - Dy
    CDx = Cx - Dx
    CDy = Cy - Dy
    AD2 = np.square(ADx) + np.square(ADy)
    BD2 = np.square(BDx) + np.square(BDy)
    CD2 = np.square(CDx) + np.square(CDy)
    detvals = (ADx*(BDy*CD2 - CDy*BD2) - ADy*(BDx*CD2 - CDx*BD2)
               + AD2*(BDx*CDy - CDx*BDy))
    return (detvals < 0)


@jit(nopython=True)
def IsDelaunayBaseWMask(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, Mask):
    """Are the quadrilaterals Delaunay.

    IsDelaunayBaseWMask is a vectorized (+jit) function which returns a
    True/False array corresponding to whether the input quadrilateral
    (points A, B, C, D in ccw order, and diagonal edge A-C) is Delaunay
    """
    IsD = np.zeros(len(Ax))
    ADx = Ax[Mask] - Dx[Mask]
    ADy = Ay[Mask] - Dy[Mask]
    BDx = Bx[Mask] - Dx[Mask]
    BDy = By[Mask] - Dy[Mask]
    CDx = Cx[Mask] - Dx[Mask]
    CDy = Cy[Mask] - Dy[Mask]
    AD2 = np.square(ADx) + np.square(ADy)
    BD2 = np.square(BDx) + np.square(BDy)
    CD2 = np.square(CDx) + np.square(CDy)
    detvals = (ADx*(BDy*CD2 - CDy*BD2) - ADy*(BDx*CD2 - CDx*BD2)
               + AD2*(BDx*CDy - CDx*BDy))
    IsD[Mask] = (detvals < 0)
    return IsD


# these are some helper function used in plotting
def GetCenter(Points):
    """Get the geometric center of a set of points.

    GetCenter returns the geometric average of the input points (usually 3 pts
    in a triangle or 2 end points of a line)
    """
    X, Y = 0, 0
    for x1 in Points:
        X += x1[0]
        Y += x1[1]
    X, Y = X/len(Points), Y/len(Points)
    return [X, Y]


def GetCircumCircleCenter(PtsIn):
    """Get the circumcenter center of a set of 3 points.

    GetCircumCircleCenter returns the coordinates of the center of the
    circumcircle about the given three points
    """
    sqval = [z[0]**2+z[1]**2 for z in PtsIn]
    diffy = [PtsIn[(i+1) % 3][1]-PtsIn[(i+2) % 3][1] for i in range(3)]
    diffx = [-PtsIn[(i+1) % 3][0]+PtsIn[(i+2) % 3][0] for i in range(3)]
    D = 2*sum([PtsIn[i][0]*diffy[i] for i in range(3)])
    PtOutx = sum([sqval[i]*diffy[i] for i in range(3)])/D
    PtOuty = sum([sqval[i]*diffx[i] for i in range(3)])/D
    return [PtOutx, PtOuty]


def Curl(vec1, vec2):
    """Curl of two vectors.

    Curl returns the curl of two (2d) vectors (as lists) vec1 X vec2.
    Output is single number (z-component of resultant vec)
    """
    return vec1[0]*vec2[1] - vec1[1]*vec2[0]


def GetIntersectionPoint(Line1, Line2):
    """Get the intersection point between two line segements.

    GetIntersectionPoint returns the point of intersection between two lines
    """
    det = ((Line2[1][1]-Line2[0][1])*(Line1[1][0]-Line1[0][0])
           - (Line1[1][1]-Line1[0][1])*(Line2[1][0]-Line2[0][0]))
    if det == 0:
        return None
    else:
        bx = (Line1[0][1]*Line1[1][0] - Line1[1][1]*Line1[0][0])
        by = (Line2[0][1]*Line2[1][0] - Line2[1][1]*Line2[0][0])
        xout = (bx*(Line2[1][0]-Line2[0][0])
                - by*(Line1[1][0]-Line1[0][0]))/det
        yout = (bx*(Line2[1][1]-Line2[0][1])
                - by*(Line1[1][1]-Line1[0][1]))/det
        return [xout, yout]


def GetIntersectionTimes(Line1, Line2):
    """Parameterized 'times' for the intersection of to line segments.

    GetIntersectionTimes returns the two "time" parameters that correspond
    to the intersection of two lines. Line1 and Line2 are parameterized by
    the times t1 and t2 respectively (each is a linear interpolation from
    their initial point to final point)
    """
    a = Line1[1][0] - Line1[0][0]
    b = Line2[0][0] - Line2[1][0]
    c = Line1[1][1] - Line1[0][1]
    d = Line2[0][1] - Line2[1][1]
    det = a*d-b*c
    if det == 0:
        return None
    else:
        Bx = Line2[0][0] - Line1[0][0]
        By = Line2[0][1] - Line1[0][1]
        t1 = (d*Bx - b*By)/det
        t2 = (-c*Bx + a*By)/det
        return t1, t2


def LinFuncInterp(P1, P2, t):
    """Interpolate between two points.

    LinFuncInterp returns the point a fraction t (0-1)
    from point 1 (P1) to point 2 (P2)
    """
    return [(1-t)*P1[i]+t*P2[i] for i in range(2)]


def BezierLinear(P1, P2):
    """Linear Bezier curve.

    BezierLinear creates a linear Bezier curve (line) for plotting based
    on the 2 input points
    """
    return mpatches.PathPatch(
        mpath.Path([(P1[0], P1[1]), (P2[0], P2[1])],
                   [mpath.Path.MOVETO, mpath.Path.LINETO]))


def BezierQuad(P1, P2, P3):
    """Quadratic Bezier Curve.

    BezierQuad creates a quadratic Bezier curve for plotting based on the
    3 input points
    """
    return mpatches.PathPatch(
        mpath.Path([(P1[0], P1[1]), (P2[0], P2[1]), (P3[0], P3[1])],
                   [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3]))


def BezierCubic(P1, P2, P3, P4):
    """Cubic Bezier Curve.

    BezierCubic creates a cubic Bezier curve for plotting based on the
    4 input points
    """
    return mpatches.PathPatch(mpath.Path(
        [(P1[0], P1[1]), (P2[0], P2[1]), (P3[0], P3[1]), (P4[0], P4[1])],
        [mpath.Path.MOVETO, mpath.Path.CURVE4,
         mpath.Path.CURVE4, mpath.Path.CURVE4]))


def BezierCustom(P1, P2, P3, P4, P5):
    """Bezier Curve - Custom.

    BezierCustom creates a line-quadratic-line Bezier curve for plotting based
    on the 5 input points
    """
    return mpatches.PathPatch(
        mpath.Path([(P1[0], P1[1]), (P2[0], P2[1]), (P3[0], P3[1]),
                    (P4[0], P4[1]), (P5[0], P5[1])],
                   [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.CURVE3,
                    mpath.Path.CURVE3, mpath.Path.LINETO]))
