from evogym.world import EvoWorld, WorldObject
from evogym.sim import EvoSim
from evogym.viewer import EvoViewer
from evogym.utils import *
from gymnasium.envs.registration import register

register(
    id = 'Walker-v0',
    entry_point = 'envs.walk:WalkingFlat',
    max_episode_steps = 500
)