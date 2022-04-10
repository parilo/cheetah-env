import mujoco_py
import os

from gym import spaces
import numpy as np
from gym.utils import seeding


def quad_from_two_vecs(vec1: np.ndarray, vec2: np.ndarray):
    xyz: np.ndarray = np.cross(vec1, vec2)
    w = np.dot(vec1, vec2) + np.linalg.norm(vec1) * np.linalg.norm(vec2)
    quat = np.array([w] + xyz.tolist())
    return quat / np.linalg.norm(quat)


def set_velocity_lin_pointer(
        sim: mujoco_py.MjSim,
        name: str,
        origin: np.ndarray,
        vel: np.ndarray,
):
    id_body = sim.model.body_name2id(name)
    id_geom = sim.model.geom_name2id(name)
    sim.model.body_pos[id_body] = origin
    vel_norm = np.linalg.norm(vel)
    vel_dir = vel / vel_norm
    sim.model.body_quat[id_body] = quad_from_two_vecs(np.array([0, 0, 1]), vel_dir)
    sim.model.geom_size[id_geom][1] = vel_norm / 10


def set_velocity_rot_pointer(
        sim: mujoco_py.MjSim,
        name: str,
        origin: np.ndarray,
        vel_z: float,
):
    id_body = sim.model.body_name2id(name)
    id_geom = sim.model.geom_name2id(name)
    sim.model.body_pos[id_body] = origin
    sim.model.geom_size[id_geom][1] = vel_z / 10


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
        self._target_vel_lin = np.array([0, 0, 0])
        self._target_vel_rot_z = 0

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
        self._update_velocity_lin_pointer()
        self._update_velocity_rot_pointer()
        self._update_target_velocity_lin_pointer()
        self._update_target_velocity_rot_pointer()
        return self._get_obs(), None, False, {}

    def render(self):
        self._get_viewer().render()

    def _get_torso_qpos(self):
        return self._sim.data.qpos[:7]

    def _get_torso_qvel(self):
        return self._sim.data.qvel[:6]

    def _get_torso_qpos_qvel(self):
        return self._get_torso_qpos(), self._get_torso_qvel()

    def _update_velocity_lin_pointer(self):
        set_velocity_lin_pointer(
            self._sim,
            'current_vel_lin',
            self._get_torso_qpos()[:3] + np.array([0, 0, 0.4]),
            self._get_torso_qvel()[:3],
        )

    def _update_velocity_rot_pointer(self):
        set_velocity_rot_pointer(
            self._sim,
            'current_vel_rot',
            self._get_torso_qpos()[:3] + np.array([0, 0, 0.4]),
            self._get_torso_qvel()[5],
        )

    def set_target_velocity_lin_pointer(self, vel: np.ndarray):
        self._target_vel_lin = vel

    def _update_target_velocity_lin_pointer(self):
        set_velocity_lin_pointer(
            self._sim,
            'target_vel_lin',
            self._get_torso_qpos()[:3] + np.array([0, 0, 0.4]),
            self._target_vel_lin,
        )

    def set_target_velocity_rot_pointer(self, vel_z: float):
        self._target_vel_rot_z = vel_z

    def _update_target_velocity_rot_pointer(self):
        set_velocity_rot_pointer(
            self._sim,
            'target_vel_rot',
            self._get_torso_qpos()[:3] + np.array([0, 0, 0.4]),
            self._target_vel_rot_z,
        )

    def _get_viewer(self):
        if self._viewer is None:
            self._viewer = mujoco_py.MjViewer(self._sim)
        return self._viewer
