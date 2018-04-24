import numpy as np
from rllab.envs.mujoco import mujoco_env
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from rllab.envs.base import Step

from copy import deepcopy

class Reacher7DofMultitaskEnvOracle(
    mujoco_env.MujocoEnv, Serializable
):
    def __init__(self, distance_metric_order=None, *args, **kwargs):
        self.goal = None
        if 'noise' in kwargs:
            noise = kwargs['noise']
        else:
            noise = 0.0
        self.__class__.FILE = 'r7dof_versions/reacher_7dof.xml'
        super().__init__(action_noise=noise)
        Serializable.__init__(self, *args, **kwargs)

        # Serializable.quick_init(self, locals())
        # mujoco_env.MujocoEnv.__init__(
        #     self,
        #     file_path='r7dof_versions/reacher_7dof.xml',   # You probably need to change this
        #     action_noise=noise,
        #     #frame_skip = 5
        # )

    def viewer_setup(self):
        if self.viewer is None:
            self.start_viewer()
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 3.5
        self.viewer.azimuth = -30

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.model.data.qpos.flat[-7:-4]
        ])

    def step(self, action):
        self.frame_skip = 5
        distance = np.linalg.norm(
            self.get_body_com("tips_arm") - self.get_body_com("goal")
        )
        reward = - distance
        self.forward_dynamics(action)
        # self.do_simulation(action, self.frame_skip)
        next_obs = self.get_current_obs()
        done = False
        return Step(next_obs, reward, done) #, dict(distance=distance)

    def sample_goals(self, num_goals):
        goals_list = []
        for _ in range(num_goals):
            newgoal = np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)
            goals_list.append(newgoal)
        return np.array(goals_list)


    @overrides
    def reset(self, reset_args=None, **kwargs):
        qpos = np.copy(self.init_qpos)
        qvel = np.copy(self.init_qvel) + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )

        if type(reset_args) is dict:
            goal_pos = reset_args['goal']
            noise = reset_args['noise']
            if self.action_noise != noise:
                print("debug action noise changing")
                self.action_noise = noise
        else:
            goal_pos = reset_args

        if goal_pos is not None:
            self.goal = goal_pos
        else:  # change goal between resets
            self.goal = np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3]).reshape(3,1)

        qpos[-7:-4] = self.goal
        qvel[-7:] = 0
        setattr(self.model.data, 'qpos', qpos)
        setattr(self.model.data, 'qvel', qvel)
        self.model.data.qvel = qvel
        self.model._compute_subtree()
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()


    def clip_goal_from_obs(self, paths):
        paths_copy = deepcopy(paths)
        for path in paths_copy:
            clipped_obs = path['observations'][:, :-3]
            path['observations'] = clipped_obs
        return paths_copy


    # def reset_model(self):
    #     qpos = np.copy(self.init_qpos)
    #     qvel = np.copy(self.init_qvel) + self.np_random.uniform(
    #         low=-0.005, high=0.005, size=self.model.nv
    #     )
    #     print(np.shape(qpos[-7:-4]))
    #     qpos[-7:-4] = self._desired_xyz
    #     qvel[-7:] = 0
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    # def _get_obs(self):
    #     return np.concatenate([
    #         self.model.data.qpos.flat[:7],
    #         self.model.data.qvel.flat[:7],
    #         self.get_body_com("tips_arm"),
    #     ])

    # def _step(self, a):
    #     distance = np.linalg.norm(
    #         self.get_body_com("tips_arm") - self.get_body_com("goal")
    #     )
    #     reward = - distance
    #     self.do_simulation(a, self.frame_skip)
    #     ob = self._get_obs()
    #     done = False
    #     return ob, reward, done, dict(distance=distance)

    # def log_diagnostics(self, paths):
    #     pass
