import numpy as np
from gym import spaces
from gym.core import ObservationWrapper


class FullyObsWrapper(ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)
        
        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            10,
            0,
            env.agent_dir
        ])

        return {
            'id': obs['id'],
            'mission': obs['mission'],
            'image': full_grid,
            'image_full': full_grid,
            'image_partial': obs['image']
        }
