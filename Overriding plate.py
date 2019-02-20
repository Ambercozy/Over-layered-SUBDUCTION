Python 3.7.1 (v3.7.1:260ec2c36a, Oct 20 2018, 14:05:16) [MSC v.1915 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import underworld as uw
import math
from underworld import function as fn
import glucifer
import numpy as np
import os

outputPath = os.path.join(os.path.abspath("."),"output/")

if uw.rank()==0:
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
uw.barrier()

xRes = 192
yRes = 48
boxLength = 4.0
boxHeight = 1.0

mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"),
                                 elementRes  = (xRes, yRes),
                                 minCoord    = (0.,0.),
                                 maxCoord    = (boxLength, boxHeight),
                                 periodic    =  [True, False] )

velocityField  = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=2 )
pressureField  = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )

swarm = uw.swarm.Swarm( mesh=mesh )
materialVariable   = swarm.add_variable( dataType="int", count=1 )
swarmLayout = uw.swarm.layouts.GlobalSpaceFillerLayout( swarm=swarm, particlesPerCell=20 )
swarm.populate_using_layout( layout=swarmLayout )

# intialise the 'materialVariable' data to represent two different materials.
upperMantleIndex   = 0
lowerMantleIndex   = 1
upperSlabIndex     = 2
lowerSlabIndex     = 3
coreSlabIndex      = 4
upperOverslabIndex = 5
lowerOverslabIndex = 6
coreOverslabIndex  = 7
weakZoneIndex      = 8

# Initial material layout has a flat lying slab at 15/ degree perturation with over-riding plate
lowerMantleY       = 0.4
slabLowerShape     = np.array( [ (1.5,0.925), (3.55,0.925), (3.50,0.900), (1.5,0.900), (1.32,0.825), (1.32,0.850) ] )
slabCoreShape      = np.array( [ (1.5,0.975), (3.65,0.975), (3.55,0.925), (1.5,0.925), (1.32,0.850), (1.32,0.900) ] )
slabUpperShape     = np.array( [ (1.5,1.000), (3.70,1.000), (3.65,0.975), (1.5,0.975), (1.32,0.900), (1.32,0.925) ] )
overslabLowerShape = np.array([(0,0.955), (1.32,0.955), (1.32,0.9), (0,0.9)])
overslabCoreShape  = np.array([(0,0.992), (1.32,0.992), (1.32,0.955), (0,0.955)])
overslabUpperShape = np.array([(0,1.000), (1.32,1.000), (1.32,0.975), (0,0.975)])
weakZoneShape      = np.array( [ (1.32, 0.925), (1.32, 1.000), (1.5, 1.000)])

slabLower = fn.shape.Polygon( slabLowerShape )
slabUpper = fn.shape.Polygon( slabUpperShape )
slabCore  = fn.shape.Polygon( slabCoreShape  )
overslabLower = fn.shape.Polygon( overslabLowerShape )
overslabUpper = fn.shape.Polygon( overslabUpperShape )
overslabCore  = fn.shape.Polygon( overslabCoreShape  )
weakZone      = fn.shape.Polygon( weakZoneShape )

# initialise everything to be upper mantle material
materialVariable.data[:] = upperMantleIndex

# change material index if the particle is not upper mantle

for index in range ( len(swarm.particleCoordinates.data) ):
    coord = swarm.particleCoordinates.data[index][:]
    if coord[1] < lowerMantleY:
        materialVariable.data[index]     = lowerMantleIndex
    if slabCore.evaluate(tuple(coord)):
             materialVariable.data[index] =coreSlabIndex
    if slabUpper.evaluate(tuple(coord)):
             materialVariable.data[index] = upperSlabIndex
    if slabLower.evaluate(tuple(coord)):
            materialVariable.data[index]  = lowerSlabIndex
    if overslabLower.evaluate(tuple(coord)):
            materialVariable.data[index]  =  lowerOverslabIndex
    if overslabUpper.evaluate(tuple(coord)):
            materialVariable.data[index]  =  upperOverslabIndex
    if overslabCore.evaluate(tuple(coord)):
            materialVariable.data[index]  =  coreOverslabIndex
    elif weakZone.evaluate(tuple(coord)):
            materialVariable.data[index]  =  weakZoneIndex
			
store = glucifer.Store('output/over-ride plate')
figParticle = glucifer.Figure( store, figsize=(960,300), name="Particles")
figParticle.append( glucifer.objects.Points(swarm, materialVariable, pointSize=2, colours='white green gray yellow purple orange red blue'))
figParticle.show()

upperMantleViscosity =    1.0
lowerMantleViscosity =  100.0
slabViscosity        =  500.0
coreViscosity        =  500.0
weakZoneViscosity    =  100.0

# The yielding of the upper slab is dependent on the strain rate.
strainRate_2ndInvariant = fn.tensor.second_invariant(
                           fn.tensor.symmetric(
                           velocityField.fn_gradient))
cohesion = 0.06
vonMises = 0.5 * cohesion/ (strainRate_2ndInvariant+1.0e-18)

# The upper slab viscosity is the minimum of the 'slabViscosity' or the 'vonMises'
slabYieldvisc = fn.exception.SafeMaths( fn.misc.min(vonMises, slabViscosity) )

# Viscosity function for the materials
viscosityMap = { upperMantleIndex   : upperMantleViscosity,
                 lowerMantleIndex   : lowerMantleViscosity,
                 weakZoneIndex      : weakZoneViscosity,
                 upperSlabIndex     : slabYieldvisc,
                 lowerSlabIndex     : slabYieldvisc,
                 coreSlabIndex      : coreViscosity,
                 upperOverslabIndex : slabYieldvisc,
                 lowerOverslabIndex : slabYieldvisc,
                 coreOverslabIndex  : coreViscosity}
viscosityMapFn = fn.branching.map( fn_key = materialVariable, mapping = viscosityMap)

mantleDensity = 0.0
slabDensity   = 1.0

densityMap = { upperMantleIndex   : mantleDensity,
               lowerMantleIndex   : mantleDensity,
               upperSlabIndex     : slabDensity,
               lowerSlabIndex     : slabDensity,
               coreSlabIndex      : slabDensity,
               upperOverslabIndex : slabDensity,
               coreOverslabIndex  : slabDensity,
               lowerOverslabIndex : slabDensity,
               weakZoneIndex      : slabDensity}
densityFn  = fn.branching.map( fn_key = materialVariable, mapping = densityMap )

# Define our vertical unit vector using a phython tuple
z_hat = ( 0.0, 1.0 )

# now create a buoyancy force vector
buoyancyFn = -1.0 * densityFn * z_hat

# Set initial conditions
velocityField.data[:] = [0.,0.]
pressureField.data[:] = 0.

# send boundary condition information to underworld
iWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
bottomWall = mesh.specialSets["MinJ_VertexSet"]

periodicBC = uw.conditions.DirichletCondition( variable         = velocityField,
                                               indexSetsPerDof  = ( bottomWall, jWalls) )

# initial linear slab viscosity setup
stokes = uw.systems.Stokes(    velocityField = velocityField,
                               pressureField = pressureField,
                               voronoi_swarm = swarm,
                               conditions    = periodicBC,
                               fn_viscosity  = viscosityMapFn,
                               fn_bodyforce  = buoyancyFn )
# create solver and solve
solver = uw.systems.Solver(stokes)

# use "lu" direct solve if running in serial
if(uw.nProcs()==1):
    solver.set_inner_method("lu")
	
advector = uw.systems.SwarmAdvector( swarm=swarm, velocityField=velocityField, order=2 )

# the root mean square Velocity
velSquared = uw.utils.Integral( fn.math.dot(velocityField,velocityField), mesh )
area = uw.utils.Integral( 1.,mesh )
Vrms = math.sqrt( velSquared.evaluate()[0]/area.evaluate()[0] )

#Plot of Velocity Magnitude
figVelocityMag = glucifer.Figure(store, figsize=(960,300))
figVelocityMag.append( glucifer.objects.Surface(mesh, fn.math.sqrt(fn.math.dot(velocityField,velocityField))) )

#Plot of Strain Rate, 2nd Invariant
figStrainRate = glucifer.Figure(store, figsize=(960,300))
figStrainRate.append( glucifer.objects.Surface(mesh, strainRate_2ndInvariant, logScale=True) )

#Plot of particles viscosity*
figViscosity = glucifer.Figure(store, figsize=(960,300))
figViscosity.append( glucifer.objects.Points(swarm, viscosityMapFn, pointSize=2) )

#Plot of particles stress invariant
figStress = glucifer.Figure( store, figsize=(960,300) )
figStress.append( glucifer.objects.Points(swarm, 2.0*viscosityMapFn*strainRate_2ndInvariant, pointSize=2, logScale=True) )

time = 0. # Initial time
step = 0  # Initial timestep
maxSteps = 49 # Maximum timesteps
steps_output = 10  # output every 10 timesteps

# define an update function 
def update():
    # Retrieve the maximum possible timestep for the advection system.
    dt = advector.get_max_dt()
    # Advect using this timestep size
    advector.integrate(dt)
    return time+dt, step+1
while step < maxSteps:
    # Solve non linear Stokes system
    solver.solve(nonLinearIterate=True)
    # output figure to file at intervals = steps_output
    if step % steps_output == 0 or step == maxSteps-1:
        #Important to set the timestep for the store object here or will overwrite previous step
        store.step = step
        figParticle.save(     outputPath + "particle"    + str(step).zfill(4) )
        figVelocityMag.save(  outputPath + "velocityMag" + str(step).zfill(4) )
        figStrainRate.save(   outputPath + "strainRate"  + str(step).zfill(4) )
        figViscosity.save(    outputPath + "viscosity"   + str(step).zfill(4) )
        figStress.save(       outputPath + "stress"      + str(step).zfill(4) )
        
        Vrms = math.sqrt( velSquared.evaluate()[0]/area.evaluate()[0] )
        print 'step = {0:6d}; time = [1:.3e]; Vrms = {2:.3e}'.format(step,time,Vrms)

    # update
    time,step = update()

import glucifer
saved = glucifer.Viewer('output/over-ride plate')

figs = saved.figures
steps= saved.steps
print("Saved database '%s'" % (saved.filename))
print(" - %d figures : %s"  % (len(figs), str(figs.keys())))
print(" - %d timesteps (final = %d) : %s" % (len(steps), steps[-1], steps))

#Re-visualise the final timestep
saved.step = steps[-1]
for name in saved.figures:
    saved.figure(name)
    saved["quality"] = 2
    saved["title"] = "Timestep ##"
    saved.show()
	
