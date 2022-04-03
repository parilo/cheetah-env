import mujoco_py
import os

from gym import spaces
import numpy as np
from gym.utils import seeding


class CheetahEnv:

    def __init__(self, seed=1):
        self._np_random, _ = seeding.np_random(seed)

        xml_path = os.path.join(
            os.path.dirname(__file__),
            'assets',
            'cheetah.xml'
        )
        model = mujoco_py.load_model_from_path(xml_path)
        self._sim = mujoco_py.MjSim(model)
        self._viewer = None

        self._obs_space = spaces.Box(low=-1.0, high=2.0, shape=(17 + 18,), dtype=np.float32)
        self._action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    def _get_obs(self):
        return {
            "qpos": np.copy(self._sim.data.qpos[2:]),
            "qvel": np.copy(self._sim.data.qvel),
        }

    def reset(self):
        self._sim.reset()
        return self._get_obs()

    def step(self, action):
        self._sim.data.ctrl[:] = action
        self._sim.step()
        return self._get_obs(), None, False, {}

    def render(self):
        self._get_viewer().render()

    def _get_viewer(self):
        if self._viewer is None:
            self._viewer = mujoco_py.MjViewer(self._sim)
        return self._viewer
