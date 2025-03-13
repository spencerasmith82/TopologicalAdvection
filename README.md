# Topological Advection
Represent curves on a 2D surface up to topological equivalence and evolve (advect) these curves forward in time due to the movement of points in a flow.

## Fluid Dynamics Motivation
Imagine that you are interested in investigating some Lagrangian properties of a 2D flow - perhaps quantifying its mixing efficiency, or finding coherent barriers to transport - but your knowledge of the flow is through a finite, possibly small, set of particle trajectories.  To quantify mixing or find coherent sets, you must be able to numerically advect a material curve forward (to see the exponential stretching of its length or lack of stretching repectively).  With a small set of particle trajectories you might not be able to reconstruct the flow's velocity field with high enough fidelity to do this.  Even if you could, naively representing a material curve with a set of numerically advected points will fail if the fluid is mixing, as the number of points necessary will grow exponentially.  The Topological Advection algorithm gets around these issues by trading a geometric description of a material curve for a topological description.  This trade-off results in a very fast algorithm, which unlocks new possibilities in quantifying mixing and detecting coherent structures.  While fluid flows are the main motivation, trajectories from any 2D flow can be analyzed (e.g. biological, granular, experimental or model generated, mathematically motivated, etc.).

## Brief Algorithm Synopsis
Curves are encoded topologically as ‘loops’ with a triangulation of the points acting as a basis for the loops (using intersection coordinates). As the points move, the triangulation is updated (using edge flips), and operators which act on loops are accumulated.  Any loop represended in the initial triangluation basis can then be evolved, or pushed forward in time, to the advected version of this loop in the final triangulation basis. 

## Documentation
For details on the modules, classes, methods, etc. that make up the topological advection algorithm, see the [Coding Documentation](https://spencerasmith82.github.io/TopologicalAdvection/).  A good starting place for most use cases is the top-level class topological_advection.py.
The code is completely in python (with python notebooks for examples), and uses vectorization (and numba jit) where appropriate for a speed boost.

## Examples
In [examples](examples/), you will find two python notebooks. One deals with trajectories that remain within a bounded region of the plane, while the other deals with trajectories that live on the torus (doubly periodic domain).  In both cases the notebooks cover the main ways to interact with the code: Creating triangulations, evolving triangulations forward in time, initializing topological curves ('loops'), evolving loops forward in time, **calculating the topological entropy of a flow**, plotting the triangulation and loops, and creating images for a movie.

## Cite / Background Reading
A paper to cite will be coming shortly.  For now, use: Spencer A. Smith, **Topological Advection**, 2025. as a placeholder.


## Appetizer
In the following video, the advected particle trajectories are from a simulation of 2D forced turbulence.  The topological advection algorithm is then used to evolve forward an initial material line.  The material curve is that is plotted is a geometric representation of the underlying topological loop data.

https://github.com/user-attachments/assets/c61ff3c1-f57b-4941-9b59-f3ceca115a58

