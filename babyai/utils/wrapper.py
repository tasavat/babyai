import numpy as np
from gym import spaces
from gym.core import ObservationWrapper


class FullyObsWrapper(ObservationWrapper):
    """
    Fully onne-hot encoded observable gridworld
    """

    def __init__(self, env):
        super().__init__(env)
        
        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 21),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        
        # full grid
        full_obs = env.grid.encode()
        full_obs[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            10,
            0,
            env.agent_dir
        ])
        
        # one-hot encoding
        full_obs_oh = np.zeros((full_obs.shape[0], full_obs.shape[1], 21))
        channel_start_index = {0: 0, 1:11, 2:17}
        for x in range(full_obs.shape[0]):
            for y in range(full_obs.shape[1]):
                for ch in range(full_obs.shape[2]):
                    value = full_obs[x][y][ch]
                    full_obs_oh[x][y][channel_start_index[ch] + value] = 1

        return {
            'id': obs['id'],
            'mission': obs['mission'],
            'image': full_obs_oh
        }
