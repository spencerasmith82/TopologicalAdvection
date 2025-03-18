# Using the Topological Advection Algorithm

## Single Periodic Boundary Case

Let's look at how to use the topological advection algorithm.  In this notebook we will consider the case of trajectories that live on a domain with periodic boundary conditions.  Generically this means doubly periodic boundary conditions (i.e. a torus), however, here we will look at an example with a single periodic boundary (i.e. an annular domain).  This will require that we create some extra trajectories as control points.  First let's choose and example system and generate some trajectories.


```python
import numpy as np
import sys
import scipy.io as sio
sys.path.append("../src")  # accessing the source folder
import topological_advection as TA
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
```

### Example System

Let's use the Bickley Jet as an example system.  A description of this analytically defined flow can be found [in this arXiv paper](https://arxiv.org/abs/1704.05716).  We provide $\vec{V}(x,y,t)$ and use an ODE solver to get the trajectories from $\frac{d \vec{r}}{dt} = \vec{V}(x,y,t)$.  This example is periodic in the x-direction only, and we will detail how to deal  


```python
#global parameters for the Bickly Jet (units will all be in Mm,hours)
U_b = 0.06266*60*60/1000 #This is a characteristic velocity measured in Mm/hour (converted from km/s)
L_b = 1.770  #This is a characteristic length (in Mm)
r0 = 6.371   #The mean radius of Earth in Mm
k_1 = 2/r0   #the Rossby wave number k_1
k_2 = 4/r0   #the Rossby wave number k_2
k_3 = 6/r0   #the Rossby wave number k_3
#These wavenumbers are taken for a lattitude of -pi/3 (south from the equator ... -pi/2 would be south pole)
k_b = [k_1,k_2,k_3]  #the Rossby wave numbers [k_1,k_2,k_3]
c_3 = 0.461*U_b
c_2 = 0.205*U_b
c_1 = c_3 + ((np.sqrt(5)-1)/2)*(k_b[2]/k_b[1])*(c_2-c_3)
c_b = [c_1,c_2,c_3]   #The Rossby wave speeds
eps_1 = 0.075   #the 1st Rossby wave amplitude
eps_2 = 0.4     #the 2nd Rossby wave amplitude
eps_3 = 0.3     #the 3st Rossby wave amplitude
eps_b = [eps_1,eps_2,eps_3]   #The Rossby wave amplitudes
params = [U_b,L_b,k_b,c_b,eps_b]  #just repackaging the parameters to pass into the various functions

#Computational Parameters
T_i = 0  #Starting time
T_f = 11*24  #Final time in hours (converted from days)

x_l = 0   #Rectangular Domain, x-range left (in Mm)
x_r = np.pi*r0   #Rectangular Domain, x-range right (in Mm) ... this works out to be about 20 Mm
y_b = -3   #Rectangular Domain, y-range bottom (in Mm)
y_t = 3   #Rectangular Domain, y-range top (in Mm)

T_num = 500 #The number of time-steps to take (will use equal time-step evolution for advected particles)
#dt = (T_f-T_i)/T_num   #The time-step to use
times = np.linspace(T_i, T_f, T_num)  #the set of times used in the ode solver

#The velocity vector function
def VelFunc(t, z, p):
    x, y = z
    U, L, k, c, eps = p
    vx0 = U*np.cosh(y/L)**(-2)
    vx1_0 = sum([eps[i]*np.exp(1j*k[i]*(x-c[i]*t)) for i in range(1,3)])
    vx1 = 2*U*np.cosh(y/L)**(-3)*np.sinh(y/L)*vx1_0.real
    vy1_0 = sum([eps[i]*1j*k[i]*np.exp(1j*k[i]*(x-c[i]*t)) for i in range(1,3)])
    vy1 = U*L*np.cosh(y/L)**(-2)*vy1_0.real
    return [vx0 + vx1, vy1]

# Initialize the particles with random positions in the domain
Ntot =  500  # the total number of initial points to seed
InitCond = np.array([np.random.uniform(x_l, x_r, Ntot), np.random.uniform(y_b, y_t, Ntot)]).T

# the trajectories
Traj = np.array([solve_ivp(fun=VelFunc, t_span=(T_i,T_f), y0=IC,
                    method='Radau', t_eval=times, args=(params,)).y.T for IC in InitCond])

# modular arithmatic to keep the trajectories in the given x range.
Traj[:,:,0] = Traj[:,:,0] % x_r
```

Now we have trajectories, we are almost ready to illustrate features of the topological advection algorithm.  First, thought, we need to make sure that the trajectories are formatted correctly.

### How to Structure Trajectory Data

Due to using the doubly periodic option for a domain that is periodic in one direction and bounded in the other direction, we will need to do a little more work than usual to get the trajectory data in an acceptable form.

First we will find the actual max and min y values, and use these to shift the points vertically (the code needs a rectangular domain that has (0, 0) as the lower left corner).  Then we will add in two rows of stationary control points at the top and bottom of the domain. Finally, we will transpose the data. Currently Traj has the signature Traj[trajectory id][time][x/y], but we need to have it as time slices (time as the first index): Tslice[time][trajectory id][x/y].


```python
y_max = np.max(Traj[:,:,1])
y_min = np.min(Traj[:,:,1])
Deltay = y_max - y_min
Deltax = x_r
#shift the trajectories up
Traj[:,:,1] = Traj[:,:,1] - y_min
#find padding
a_ratio = Deltax/Deltay
npts_x = int(np.sqrt(Ntot*a_ratio))
npts_y = int(Ntot/npts_x)
dx = Deltax/(npts_x-1)
dy = Deltay/(npts_y-1)
# shift for the y-padding (1.5*dx on both sides)
Traj[:,:,1] = Traj[:,:,1] + 1.5*dy
# define domain
domain = [[0, 0], [x_r, Deltay + 3*dy]]

# define boundary control points and generate stationary trajectories to add to the trajectory set.
x_vals = np.linspace(0, x_r, npts_x, endpoint=False)
top_bnd = np.array([x_vals + dx/4, np.array([domain[1][1]- dy/2 for i in range(x_vals.shape[0])])]).T
btm_bnd = np.array([x_vals + dx*3/4, np.array([dy/2 for i in range(x_vals.shape[0])])]).T
# these are the boundary points
bnd_pts = np.concatenate((top_bnd, btm_bnd))
# these are the boundary trajectories (repeating the boundary points for each time)
bnd_traj = np.tile(bnd_pts, (T_num,1,1)).transpose(1,0,2)
# update the trajectory set to include the boundary trajectories
Traj_plus = np.concatenate((Traj, bnd_traj))
# converting a set of trajectories to a set of time-slices
Tslices = Traj_plus.transpose(1,0,2)  # just swapping the time and particle id number
```

### Topological Advection Object Initialization

First we need to initialize a topological_advection object with the time-slices and time array. We must also pass in the bounding domain. If we don't provide a boundary, it will be automatically calculated from min/max x/y trajectory values (which in this case would not give you good bounds).

Also, importantly, we are setting the PeriodicBC flag to True, indicating that we will use the module for periodic boundary conditions (and not bounded trajectories).


```python
TopAdvec = TA.TopologicalAdvection(Tslices, times, Domain=domain, PeriodicBC=True)
```

### Triangulation Plotting

Before doing any work with this object, let's first plot the triangulation to see what we have. 


```python
# Don't have a loop to plot yet, so we set PlotLoop = False
# We want to plot the triangulation at the intial time-slice, so
# set Initial = True
TopAdvec.Plot(PlotLoop = False, Initial = True)
```


    
![png](output_9_0.png)
    


### Plotting Parameters

The first step to plotting is to set the PlotParamters attribute of our TopAdvec object (there are default values, but you might want to change them). If you want to know the current set of values for the plotting parameters, use the PrintPlotParameters method:


```python
TopAdvec.PrintPlotParameters()
```

    filename: None
    triplot: True
    Delaunay: True
    DelaunayAdd: False
    Bounds: [[0, 0], [20.015086796020572, np.float64(9.885601188008248)]]
    FigureSizeX: 8
    dpi: 200
    ptlabels: False
    markersize: 2.0
    linewidth_tri: 0.5
    linecolor_tri: g
    color_weights: False
    log_color: True
    color_map: inferno_r
    linewidth_tt: 1.0
    linecolor_tt: r
    alpha_tt: 1.0
    frac: 0.9
    tt_lw_min_frac: 0.05
    _conversion_factor: 0.03474841457642461
    _max_weight: None


### Modifying Plotting Parameters

To modify the plotting parameters, we call the set SetPlotParameters method.  Let's set the point maker size to be larger (markersize = 20), change the triangluation line color (linecolor_tri = 'm'), and line width (linewidth_tri = 4), and add the point ids as labels (ptlabels = True).


```python
TopAdvec.SetPlotParameters(markersize = 20, linecolor_tri = 'm',
                           linewidth_tri = 4, ptlabels = True)
TopAdvec.Plot(PlotLoop = False, Initial = True)
```


    
![png](output_13_0.png)
    


To restore the default plotting parameters, use the ResetPlotParametersDefault method:


```python
TopAdvec.ResetPlotParametersDefault()
```

### Finding Topological Entropy

Before getting to creating and plotting loops, let's take a look at one of the main things this algorithm was build for: finding the topological entropy of a flow (really a lower bound on this given the trajectory set).  To find the topological entropy simply use the GetTopologicalEntropy method:


```python
# the TE, fitting error, and weight list are returned
# they are also set as attributes of TopAdvec, i.e. 
# TopAdvec.TopologicalEntropy = [TE, TE_err] and 
# TopAdvec.TotalWeightOverTime = WeightsM
TE, TE_err, Weights = TopAdvec.GetTopologicalEntropy()
print("The Topological Entropy is ", TE, " +/- ", TE_err)
# Note that a progress bar is displayed to give you a sense of how long the
# Triangulation evolution will take.
```

    The Topological Entropy is  0.0134495906309593  +/-  8.001019829109524e-05===========================] 99.8% ...


Note that the topological entropy is given in units of inverse time (which is set by your units for the time array).

The TE is calculated by fitting the log of the total weights (total weight is a measure of loop length) vs. time to a straight line (exponential stretching, so slope of this line is TE).  We can plot the weight data to see how good the fit is:


```python
plt.plot(TopAdvec.Times, TopAdvec.TotalWeightOverTime)
plt.xlabel("Time")
plt.ylabel("Loop Length")
plt.title("Exponential Increase in Loop Length over Time")
plt.show()
```


    
![png](output_19_0.png)
    



```python
# Plotting with similogy to see the linear fit
W_0 = TopAdvec.TotalWeightOverTime[0]
T_0 = TopAdvec.Times[0]
TE = TopAdvec.TopologicalEntropy[0]
Weight_Fit = [W_0*np.exp(TE*(t-T_0)) for t in TopAdvec.Times]
plt.semilogy(TopAdvec.Times, TopAdvec.TotalWeightOverTime, c='b', label='weight data')
plt.semilogy(TopAdvec.Times, Weight_Fit, c='k', label='linear fit')
plt.xlabel("Time")
plt.ylabel("Loop Length")
plt.legend()
plt.title("Topological Entropy as Slope of log Weights")
ax = plt.gca()
plt.text(0.3, 0.7, 'TE (slope) = '+str(TE), horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes,fontsize = 10)
plt.show()
```


    
![png](output_20_0.png)
    


Note that there is an initial transitory period before the stretching starts.  If we want to have more control over the segment of the weight list that is used for the topological entropy calculation, we have a few options:


```python
# We can directly specify the starting and ending indices for the fit: 
w_ind=[50, -1]
# We could, alternatively, set the fraction of the time length to start the fitting at:
# start_fraction=0.2

TE, TE_err, Weights = TopAdvec.GetTopologicalEntropy(ss_indices=w_ind)
print("The Topological Entropy is ", TE, " +/- ", TE_err)
```

    The Topological Entropy is  0.012829642286598018  +/-  7.948378742622482e-05



```python
# Plotting with similogy to see the linear fit
W_0 = TopAdvec.TotalWeightOverTime[w_ind[0]]
T_0 = TopAdvec.Times[w_ind[0]]
TE = TopAdvec.TopologicalEntropy[0]
Weight_Fit = [W_0*np.exp(TE*(t-T_0)) for t in TopAdvec.Times[w_ind[0]:w_ind[1]]]
plt.semilogy(TopAdvec.Times, TopAdvec.TotalWeightOverTime, c='b', label='weight data')
plt.semilogy(TopAdvec.Times[w_ind[0]:w_ind[1]], Weight_Fit, c='k', label='linear fit')
plt.xlabel("Time")
plt.ylabel("Loop Length")
plt.legend()
plt.title("Topological Entropy as Slope of log Weights")
ax = plt.gca()
plt.text(0.3, 0.7, 'TE (slope) = '+str(TE), horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes,fontsize = 10)
plt.show()
```


    
![png](output_23_0.png)
    


As you can see, the topological entropy estimate is probably not great.  This example could benefit from more points and a larger final time.

### Final Triangulation Plotting

The triangulation has been evolved forward as part of the topological entropy calculation, so we could plot the final triangulation:


```python
# note that we have set Initial = False to plot the final triangluation
TopAdvec.Plot(PlotLoop = False, Initial = False)
```


    
![png](output_26_0.png)
    


### Evolving the Triangulation

The above triangulation has a lot of long, skinny triangles in it, as the GetTopologicalEntropy method evolves the triangulation forward using collapse events only.  We certainly could use this to plot loops, but it looks much nicer to have a Delaunay triangulation.  One can re-evolve the triangulation with the Delaunay = True flag set.


```python
# Evolve the triangulation again, maintaining a Delaunay triangulation
TopAdvec.EvolveTri(Delaunay = True)
# plot the final triangulation again
TopAdvec.Plot(PlotLoop = False, Initial = False)
```

    [====================================================================================================] 99.8% ...


    
![png](output_28_1.png)
    


### Geometric Curves

Now let's see about initializing loops.  We do this using the CurveSet object (accessed as an attribute of our TopAdvec object).  This has methods for adding geometric curves to a curve set that will then be used to initialize a topological loop.  We'll look at an example (and a few more later):


```python
# Don't need to clear the curve right now, but if there were any curves
# previously added to the curve set, we would need to clear them:
TopAdvec.ClearCurves()
# Add a vertical line to the curve set
TopAdvec.CurveSet.AddVerticalLine(x_val = 1.0)
```

### Topological Loop Initialization and Plotting

Now we initialize a topological loop with these accumulated geometric curves.  Note that while the curves are allowed to overlap (won't break anything), the way the crossings are re-connected is determined by the triangulation locally at the crossing point (which we don't control), so you should just make sure your curves do not overlap to begin with.


```python
# First initialize the loop with the curves
TopAdvec.ResetPlotParametersDefault()
TopAdvec.InitializeLoopWithCurves()
# now plot the topological loops
TopAdvec.Plot(Initial = True)  # note that PlotLoop = True is the default
```


    
![png](output_32_0.png)
    


A few things to note about loop plotting:  First, loops are topological objects, and what is plotted is neccesarily a particular geometric representation of the loop (based on using the Voronoi diagram dual to the Delaunay triangulation as control points).  This representation is topologically equivalent to the initializing geometric curves (can be deformed into one-another without going through the trajectory points - i.e. isotopic).  Second, geometric curves that are 'open' will have their ends wrap around the nearest point so that their loop is closed.  Third, what is being plotted is actually a train-track (segments of a loop that go though a given triangle are bundled together into one segment, which splits and merges with other segments; all while being tangent at splitting point).  As the loop becomes more convoluted, the train-track provides a way to visualize the aspects of the topological loop data.  What is lost in this representation is how many pieces of the loop have been bundled together (i.e. the weights/intersection coordinates).  In a moment we will see a few way to encode this information in the plot.

### Evolving Loops Forward

Now that the triangulation has been evolved forward to the final time (and therefore has accumulated an ordered list of weight operators), we can evolve any loop forward very efficiently (evolving a loop forward is a few orders of magnitude faster than evolving the whole triangulation forward).  To evolve the current loop forward just use the EvolveLoop method:


```python
# Evolve the current loop forwards (there are actually two copies of the loop in the TopAdvec.LoodData object,
# and only one is evolved forwards - this is to allow plotting a given loop at both the initial and final times).
TopAdvec.EvolveLoop()
# Now plot the loop in its final state
TopAdvec.Plot(PlotLoop = True, Initial = False)
```


    
![png](output_35_0.png)
    


### Plotting Options for Highlighting Weights

The above train-tracks plot of the loop after being advected forwards fills up much of the domain (because this is a mixing flow).  However, we don't know which of the weights are large and which are small.  There are two ways of highlighting weights when plotting: by color and by line width.  Lets' first consider color:


```python
# Set the plot parameter color_weights to True.
TopAdvec.SetPlotParameters(color_weights = True)
TopAdvec.Plot(PlotLoop = True, Initial = False)
```


    
![png](output_37_0.png)
    


The darker train-track segments indicate larger weights, and the lighter are smaller weights.  The default setting for the color_map is 'inferno_r', which you can change if you have a favorite (try to use only perceptually uniform sequential color maps).  The default for how we assign a weight a color is given by log_color = True (the linear color scale is mapped onto the log of the weights, highlighting the full spectrum of weights).  For the next plot, let's set log_color = False, and let's turn off the background triangulation, which is getting crowded. We can also set the domain so that the control points are out of of view.


```python
TopAdvec.SetPlotParameters(log_color = False, triplot = False,
                           Bounds = [[0, 0.05*domain[1][1]],
                                     [domain[1][0],0.95*domain[1][1]]])
TopAdvec.Plot(PlotLoop = True, Initial = False)
```


    
![png](output_39_0.png)
    


With the colors being linear in the weights, the plot highlights the very largest weights most.  Next, let's see how to encode the weight data in the width of the train-track segments.  We do this with the DelaunayAdd = True plot parameter.  We will also turn the color weights off.  The DelaunayAdd option plots the train-track segments with a line width that scales linearly with weight (max line width is set with linewidth_tt).  The individual segments then perceptually add as they merge at junctions.  A minimum linewidth (under which all lines are plotted with this width) is set using tt_lw_min_frac (default = 0.05 means a maximum linewidth of 10 gives a minimum linewidth of 0.5).


```python
TopAdvec.SetPlotParameters(DelaunayAdd = True, linewidth_tt = 5.0, color_weights = False)
TopAdvec.Plot(PlotLoop = True, Initial = False)
```


    
![png](output_41_0.png)
    


The distribution of weights is now strikingly apparent.  If the original curve represented a material curve of dye in the fluid, this plot represents information on the relative dye concentration in different regions after being advected forward in time.

Instead of creating a movie of the mixing behavior, let's try to find a few coherent sets (bounded by curves which don't appreciably stretch out over time).


```python
TopAdvec.ClearCurves()
# We add an ellipse to the curve set
TopAdvec.CurveSet.AddEllipse(center=[0.78*domain[1][0], 0.65*domain[1][1]],
                             a=0.9, b=0.6,
                             phi=0, NumPoints=200)
# Add a custom curve to the curve set
x_pts = np.linspace(0.45*domain[1][0], 0.55*domain[1][0], 100)
y_pts = 0.3*np.square(x_pts-0.5*domain[1][0])+ 0.45*domain[1][1]
points = np.array([x_pts,y_pts]).T
TopAdvec.CurveSet.AddOpenCurve(points)
```


```python
TopAdvec.ResetPlotParametersDefault()
TopAdvec.InitializeLoopWithCurves()
# now plot the topological loops
TopAdvec.Plot(Initial = True)
```


    
![png](output_44_0.png)
    



```python
TopAdvec.EvolveLoop()
# Now plot the loop in its final state
TopAdvec.Plot(PlotLoop = True, Initial = False)
```


    
![png](output_45_0.png)
    


## Making Videos

We can also make a movie of the evolution of this loop using the MovieFigures method.  This creates figures of the loop/triangulation at each timestep and outputs them to a folder.  You can then use your favorite video editing software (like ffmpeg) to stitch these together into a movie.  For this option, it is best if the steps between timeslices are uniform.  Again the first step is to get your plotting attributes set the way you want them (these will be consistent across all the movie frames).


```python
# Let's reduce the dpi (dot per inch) and FigureSizeX (width of the figure in inches - the height is automatically 
# set to maintain the aspect ratio), so that the image files will be smaller.
TopAdvec.SetPlotParameters(FigureSizeX = 6, dpi =  150, linewidth_tt = 1.0,
                          triplot = False,
                           Bounds = [[0, 0.05*domain[1][1]],
                                     [domain[1][0],0.95*domain[1][1]]])

TopAdvec.Plot(PlotLoop = True, Initial = False)
```


    
![png](output_47_0.png)
    



```python
# Now create the movie figures.  Will use the default settings for the folder name, file names,
# and file type:
TopAdvec.InitializeLoopWithCurves()
TopAdvec.MovieFigures(PlotLoop=True, Delaunay=True, ImageFolder="MovieImages/",
                      ImageName="EvolvingLoop", filetype=".png")
```

    [====================================================================================================] 99.8% ...

Now to create the movie from the images, you can run ffmpeg with the following setup in your terminal:

ffmpeg -r 25 -i "MovieImages/EvolvingLoop%04d.png" -vcodec libx264 -crf 28 -pix_fmt yuv420p Videos/AdvectingLoop3.mp4


https://github.com/user-attachments/assets/01c97b53-7e41-4355-a3cf-f4b412db2bfe

