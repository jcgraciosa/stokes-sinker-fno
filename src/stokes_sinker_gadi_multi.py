# %% [markdown]
# # Linear stokes sinker
# 
# The stokes sinker is a benchmark to test sinking velocities determined by the stokes solver.
# 
# Two materials are used, one for the background and one for the sinking ball.
# 
# Stokes velocity is calculated and compared with the velocity from the UW model to benchmark the stokes solver
# 

# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
from mpi4py import MPI
import os
import random

import zarr

# %%
# import pyvista as pv
# import vtk

# %%
#### visualisation within script
render = True

#### save output
save_output = False

# %%
### number of steps for the model

# %%
### stokes tolerance
tol = 1e-5

# %%
res = 80

# %%
nsteps = 100

x_min = -1.
x_max = 1.
z_min = 0
z_max = 1

# parameters:
# viscosity contrast - vs - not-random
# density sphere - ds - not-random
# sphere radius - sr - not-random - 0.05, 0.1, 0.15, 0.2, 0.25
# sphere start - stXxYx - drawn randomly

# %%
# Set constants for the viscosity of each material
# viscosities to try - 100, 1000, 10000, 100000, 1000000
# default value is 1000
viscBG     =  1.0
viscSphere = 100. # free param

# %%
# set density of the different materials
# densities to try for the sphere - 100, 1000, 10000, 100000, 1000000
densityBG     =  0.0
densitySphere = 100. # free param

# sphere radius to try - 0.05, 0.1, 0.15, 0.2, 0.25
sphereRadius = 0.05 # free param

sphereCentre = (random.uniform(x_min + sphereRadius + 0.05, x_max - sphereRadius - 0.05), 
                random.uniform(z_min + sphereRadius + 0.05, z_max - sphereRadius - 0.05))


# %%
# Set size and position of dense sphere.
# sphereRadius = 0.1
# sphereCentre = (0., 0.7)


#sphereCentre = (0.0, 0.7)

# %%
# define some names for our index
materialLightIndex = 0
materialHeavyIndex = 1
outputPath = f"./stokes_train/sinker_res{res}_vs{viscSphere}_ds{densitySphere}_sr{sphereRadius}_stX{sphereCentre[0]:.3f}Y{sphereCentre[1]:.3f}"

outfile = outputPath + "/stokes"

if uw.mpi.rank==0:      
    ### create folder if not run before
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

# %%
# location of tracer at bottom of sinker
x_pos = sphereCentre[0]
y_pos = sphereCentre[1] - sphereRadius

# mesh = uw.meshing.StructuredQuadBox(minCoords=(x_min, z_min), maxCoords=(x_max, z_max),  elementRes=(res,res))
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(x_min, z_min), maxCoords=(x_max, z_max), cellSize=1.0 / res, regular=False
)

# %% [markdown]
# ####  Create Stokes object and the required mesh variables (velocity and pressure)

# %%
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
bfz = uw.discretisation.MeshVariable("BF", mesh, 1, degree=2) # for the body force
visc = uw.discretisation.MeshVariable("VISC", mesh, 1, degree=2)

# %%
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)

# %%
### free slip BC

stokes.add_dirichlet_bc( (0.,0.), 'Left',   (0) ) # left/right: function, boundaries, components
stokes.add_dirichlet_bc( (0.,0.), 'Right',  (0) )

stokes.add_dirichlet_bc( (0.,0.), 'Top',    (1) )
stokes.add_dirichlet_bc( (0.,0.), 'Bottom', (1) )# top/bottom: function, boundaries, components 


# %%
swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.IndexSwarmVariable("M", swarm, indices=2)
swarm.populate(fill_param=4)

# %%
sphere = np.array(
    [[sphereCentre[0], sphereCentre[1], sphereRadius, 1]]
    )

# %% [markdown]
# Update the material variable to include the background and sphere

# %%
with swarm.access(material):
    material.data[...] = materialLightIndex

    for i in range(sphere.shape[0]):
        cx, cy, r, m = sphere[i, :]
        inside = (swarm.data[:, 0] - cx) ** 2 + (swarm.data[:, 1] - cy) ** 2 < r**2
        material.data[inside] = m

# %%
### add tracer for sinker velocity
tracer_coords = np.zeros(shape=(1, 2))
tracer_coords[:, 0], tracer_coords[:, 1] = x_pos, y_pos

# %%
tracer = uw.swarm.Swarm(mesh=mesh)

# %%
tracer.add_particles_with_coordinates(tracer_coords)

# %%

density_fn = material.createMask([densityBG, densitySphere])

# %%
# density_fn

# %%
### assign material viscosity

viscosity_fn = material.createMask([viscBG, viscSphere])


# %%
stokes.constitutive_model.Parameters.viscosity = viscosity_fn
stokes.bodyforce = sympy.Matrix([0, -1 * density_fn])
stokes.saddle_preconditioner = 1.0 / stokes.constitutive_model.Parameters.viscosity

# projection object to calculate the body force on the mesh 
bfz_calc = uw.systems.Projection(mesh, bfz)
bfz_calc.uw_function = -1*density_fn
bfz_calc.smoothing = 1.0e-5
bfz_calc.petsc_options.delValue("ksp_monitor")

# projection object to calculate the gradient along Z
visc_calc = uw.systems.Projection(mesh, visc)
visc_calc.uw_function = viscosity_fn
visc_calc.smoothing = 1.0e-5
visc_calc.petsc_options.delValue("ksp_monitor")

# %%
with mesh.access():
    print(bfz.data.min())
    print(visc.data.max())

# %%
stokes.tolerance = tol

# %%
step = 0
time = 0.0

# %%
tSinker = np.zeros(nsteps)*np.nan
ySinker = np.zeros(nsteps)*np.nan

# %%
if uw.mpi.size == 1:
    stokes.petsc_options['pc_type'] = 'lu'

# %%
stokes.petsc_options.view()


# %% [markdown]
# #### Stokes solver loop

# %%
while step < nsteps:
    ### Get the position of the sinking ball
    #with tracer.access():
    #    ymin = tracer.data[:,1].min()

    #if ymin <= 1e-5: # break if tracer is close to bottom
    #    break

    #ySinker[step] = ymin
    tSinker[step] = time
    
    ### print some stuff
    #if uw.mpi.rank == 0:
    #    print(f"Step: {str(step).rjust(3)}")
    #    print(f"Step: {str(step).rjust(3)}, time: {time:6.2f}, tracer:  {ymin:6.2f}")

    ### solve stokes
    stokes.solve(zero_init_guess=True)
    
    ### estimate dt
    dt = stokes.estimate_dt()

    ### advect the swarm
    swarm.advection(stokes.u.sym, dt, corrector=False)


    ### advect tracer
    tracer.advection(stokes.u.sym, dt, corrector=False)

    bfz_calc.solve()
    visc_calc.solve()

    mesh.write_timestep_xdmf(filename = outfile, meshVars=[v, p, bfz, visc], index=step)

    step += 1
    time += dt

