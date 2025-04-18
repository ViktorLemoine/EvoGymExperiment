from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot
import os
import numpy as np

path = os.path.join('world_data', 'simple_walker_env.json')

world = EvoWorld.from_json(path)

#visualize the EvoWorld environment
world.pretty_print()

robot_structure, robot_connections = sample_robot((4, 4))

world.add_from_array(
	name='robot',
	structure=robot_structure,
	x=3,
	y=1,
	connections=robot_connections
)

sim = EvoSim(world)
sim.reset()

world.pretty_print()

viewer = EvoViewer(sim)
viewer.track_objects('robot')

for i in range(500):
	sim.set_action(
		'robot',
		np.random.uniform(
			low = 0.6,
			high = 1.6,
			size=(sim.get_dim_action_space('robot'),)
		)
	)
	sim.step()
	viewer.render('screen')