import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides
from copy import deepcopy


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class HalfCheetahEnvOracle(MujocoEnv, Serializable):

    FILE = 'half_cheetah.xml'

    def __init__(self, *args, **kwargs):
        self._goal_vel = None
        super(HalfCheetahEnvOracle, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def sample_goals(self, num_goals):
        return np.random.uniform(0.0, 2.0, (num_goals, ))

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        # print("debug52", reset_args)
        if type(reset_args) is dict:
            goal_vel = reset_args['goal']
            noise = reset_args['noise']
            # if self.action_noise != noise:
            #     print("debug action noise changing")
            #     self.action_noise = noise
        else:
            goal_vel = reset_args
        if goal_vel is not None:
            self._goal_vel = goal_vel
        else:  # keep the goal the same between resets
            #self._goal_vel = np.random.uniform(0.1, 0.8)
            self._goal_vel = np.random.uniform(0.0, 2.0)
        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        return obs

    def get_current_obs(self):
        obs = np.concatenate([
            self.model.data.qpos.flatten()[1:],
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])
        return np.r_[obs, np.array([self._goal_vel])]

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        run_cost = 1.*np.abs(self.get_body_comvel("torso")[0] - self._goal_vel)
        cost = ctrl_cost + run_cost
        reward = -cost
        done = False
        return Step(next_obs, reward, done)

    @overrides
    def log_diagnostics(self, paths, prefix=''):
        #path["observations"][-1][-3] - path["observations"][0][-3]
        progs = [
            path["observations"][-1][-4] - path["observations"][0][-4]
            for path in paths
        ]
        logger.record_tabular(prefix+'AverageForwardProgress', np.mean(progs))
        logger.record_tabular(prefix+'MaxForwardProgress', np.max(progs))
        logger.record_tabular(prefix+'MinForwardProgress', np.min(progs))
        logger.record_tabular(prefix+'StdForwardProgress', np.std(progs))

    def clip_goal_from_obs(self, paths):
        paths_copy = deepcopy(paths)
        for path in paths_copy:
            clipped_obs = path['observations'][:, :-1]
            path['observations'] = clipped_obs
        return paths_copy