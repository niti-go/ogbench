import os

import mujoco
import numpy as np
from gymnasium.spaces import Box

from ogbench.locomaze.point import PointEnv
from ogbench.locomaze.zermelo_flow import FlowField


class ZermeloPointEnv(PointEnv):
    """Point mass environment with a background fluid flow field (Zermelo navigation).

    The flow field adds a displacement to the agent's position at each step,
    simulating navigation through a fluid with a static velocity field.
    """

    def __init__(self, flow_field_path=None, include_flow_in_obs=True, **kwargs):
        self._flow_field_path = flow_field_path
        self._include_flow_in_obs = include_flow_in_obs
        self._flow_field = FlowField(flow_field_path)
        super().__init__(**kwargs)

        if self._include_flow_in_obs:
            # Extend observation space by 2 for local flow vector
            base_shape = self.observation_space.shape[0]
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(base_shape + 2,), dtype=np.float64
            )

    def step(self, action):
        prev_qpos = self.data.qpos.copy()
        prev_qvel = self.data.qvel.copy()

        action = 0.2 * action

        # Agent's intended displacement
        self.data.qpos[:] = self.data.qpos + action

        # Flow displacement: dt * flow_velocity
        dt = self.frame_skip * self.model.opt.timestep
        x, y = self.data.qpos[0], self.data.qpos[1]
        flow_vx, flow_vy = self._flow_field.get_flow(x, y)
        self.data.qpos[0] += dt * flow_vx
        self.data.qpos[1] += dt * flow_vy

        self.data.qvel[:] = np.array([0.0, 0.0])

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        observation = self.get_ob()

        if self.render_mode == 'human':
            self.render()

        return (
            observation,
            0.0,
            False,
            False,
            {
                'xy': self.get_xy(),
                'prev_qpos': prev_qpos,
                'prev_qvel': prev_qvel,
                'qpos': qpos,
                'qvel': qvel,
            },
        )

    def get_ob(self):
        base_ob = self.data.qpos.flat.copy()
        if self._include_flow_in_obs:
            x, y = self.data.qpos[0], self.data.qpos[1]
            flow_vx, flow_vy = self._flow_field.get_flow(x, y)
            return np.concatenate([base_ob, [flow_vx, flow_vy]])
        return base_ob
