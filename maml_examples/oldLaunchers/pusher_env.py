import numpy as np
from gym import utils
from rllab.envs.mujoco import mujoco_env
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab import spaces

from rllab.envs.base import Step
import joblib
import os.path
# import mujoco_py
# from mujoco_py.mjlib import mjlib
from PIL import Image
import rllab.misc.logger as logger


BIG = 1e6


class PusherEnv(utils.EzPickle, Serializable):
    def __init__(self, xml_file=None, distractors=True, debug=False, *args, **kwargs):
        logger.log("initializing environment pusher")
        utils.EzPickle.__init__(self)
        logger.log("using xml_file", xml_file)
        if xml_file is None:
            xml_file = 'pusher.xml'
        self.__class__.FILE = xml_file
        self.include_distractors = distractors
        self.debug = debug
        logger.log("Debugging environment set to %s" %self.debug)
        self.test_dir = "/home/rosen/FaReLI_data/pushing/test2_paired_push_demos_noimg_DO NOT USE/"
        self.train_dir = "/home/rosen/FaReLI_data/pushing/paired_push_demos_noimg_noise/"
        self.xml_dir = "/home/rosen/FaReLI_data/pushing/push_textures/sim_push_xmls/"
        # self.test_dir = "/root/code/rllab/saved_expert_traj/PUSHER-DEMOS/test2_paired_push_demos_noimg/"
        # self.train_dir = "/root/code/rllab/saved_expert_traj/PUSHER-DEMOS/paired_push_demos_noimg/"
        # self.xml_dir = "/root/code/rllab/vendor/mujoco_models/push_textures/sim_push_xmls/"
        self.goal_num = None
        self.test = False
        self.target_on_left=None
        self.reset()
        self.onehot = False
        self.onehot_dim = 5
        self.onehot_position = 0  # if dim is 5, options are 0 through 4 for onehot, and -1 for vector of zeroes
        # self.reset_xml_on_reset=False
        # self.xml_file=xml_file
        # self.observation_space = self.mujoco.observation_space
        self.action_space = self.mujoco.action_space
        # self.goal_num, self.test = self.sample_goals(num_goals=1, test=False)[0]
        # self.mujoco = mujoco_env.MujocoEnv(file_path=xml_file)
        # super().__init__()
        Serializable.__init__(self, *args, **kwargs)
        # mujoco_env.MujocoEnv.__init__(self, file_path=xml_file)

    def viewer_setup(self):
        print("debug, starting viewer")
        if self.mujoco.viewer is None:
            self.mujoco.start_viewer()
        self.mujoco.viewer.cam.trackbodyid = -1
        self.mujoco.viewer.cam.distance = 4.0

    def get_current_obs(self):
        return self._get_obs()

    def step(self, action):
        self.mujoco.frame_skip = 5
        ob, reward, done, env_info = self._step(a=action)
        return Step(ob, reward, done, **env_info)

    def sample_goals(self, num_goals, test=False):
        out = []
        for _ in range(num_goals):
            if not test:
                while True:
                    i = int(np.random.choice(1000,1))
                    if os.path.isfile(self.train_dir+str(i)+".pkl"):
                        out.append((i,False))
                        break
            else:
                while True:
                    i = int(np.random.choice(1000,1))
                    if os.path.isfile(self.test_dir+str(i)+".pkl"):
                        out.append((i,True))
                        break
        return np.array(out)

    @overrides
    def reset(self, reset_args=None, **kwargs):
        if reset_args is None:
            logger.log("Debug, warning, reset_args for env is None")
        goal = reset_args
        if goal is not None:
            assert len(goal)==2, "wrong size goal"
            goal_num, test = goal
            if (goal_num != self.goal_num) or (test != self.test):
                if self.mujoco.viewer is not None:
                    self.mujoco.stop_viewer()
                self.mujoco.terminate()
                self.goal_num, self.test = goal
                demo_path = (self.train_dir + str(self.goal_num) + ".pkl") if not self.test else (
                self.test_dir + str(self.goal_num) + ".pkl")
                demo_data = joblib.load(demo_path)
                xml_file = demo_data["xml"]
                xml_file = xml_file.replace("/root/code/rllab/vendor/mujoco_models/", self.xml_dir)
                # print("debug,xml_file", xml_file)
                if int(xml_file[-5])%2==0 and not self.debug:
                    print("inverted_order", xml_file, self.goal_num, self.test)
                    self.shuffle_order=[1,0]
                else:
                    print("normal_order",xml_file, self.goal_num, self.test)
                    self.shuffle_order=[0,1]
                self.mujoco = mujoco_env.MujocoEnv(file_path=xml_file)
            # else:
                # print("continuing with xml", self.mujoco.FILE, self.goal_num, self.test)
        elif self.goal_num is None:  #if we already have a goal_num, we don't sample a new one, just reset the model
            self.goal_num, self.test = self.sample_goals(num_goals=1,test=False)[0]
            demo_path = (self.train_dir+str(self.goal_num)+".pkl") if not self.test else (self.test_dir+str(self.goal_num)+".pkl")
            demo_data = joblib.load(demo_path)
            xml_file = demo_data["xml"]
            xml_file = xml_file.replace("/root/code/rllab/vendor/mujoco_models/",self.xml_dir)

            if int(xml_file[-5]) % 2 == 0 and not self.debug:
                print("inverted_order, first time initializing env", xml_file)
                self.shuffle_order = [1, 0]  # TODO: flip back, this is set to [0,1] just for debugging purposes
            else:
                print("normal_order, first time initializing env", xml_file)
                self.shuffle_order = [0, 1]
            self.mujoco = mujoco_env.MujocoEnv(file_path=xml_file)
            # self.viewer_setup()
        self.reset_model()
        return self.get_current_obs()

    def _step(self, a):
        # normalize actions
        if self.mujoco.action_space is not None:
            lb, ub = self.mujoco.action_space.low, self.mujoco.action_space.high
            a = lb + (a + 1.) * 0.5 * (ub - lb)
            a = np.clip(a, lb, ub)

        vec_1 = self.mujoco.get_body_com("object") - self.mujoco.get_body_com("tips_arm")
        vec_2 = self.mujoco.get_body_com("object") - self.mujoco.get_body_com("goal")

        # reward_near = - np.linalg.norm(vec_1)
        # reward_dist = - np.linalg.norm(vec_2)
        reward_near = - np.square(vec_1).sum()
        reward_dist = - np.square(vec_2).sum()
        reward_ctrl = - np.square(a).sum()
        reward = 1.0*reward_dist + 0.1 * reward_ctrl + 10.0*0.5 * reward_near

        self.mujoco.do_simulation(a, n_frames=self.mujoco.frame_skip)
        # extra added to copy rllab forward_dynamics.
        self.mujoco.model.forward()

        if self.target_on_left:
            success_left = 1 if np.sum(vec_2**2) <0.017 else 0
            success_right = -1
        else:
            success_left = -1
            success_right = 1 if np.sum(vec_2**2) <0.017 else 0

        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl,
            success_left=success_left,
            success_right=success_right,
            target_on_left=self.target_on_left,
        )

    def reset_model(self):
        qpos = np.copy(self.mujoco.init_qpos)

        self.goal_pos = np.asarray([0., 0.])
        while True:
            self.obj_pos = np.concatenate([
                    np.random.uniform(low=-0.3, high=0, size=1),
                    np.random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.obj_pos - self.goal_pos) > 0.17:
                break

        if self.include_distractors:
            if self.obj_pos[1] < 0:
                y_range = [0.0, 0.2]
                self.target_on_left=True
                # print("target on left")
            else:
                y_range = [-0.2, 0.0]
                self.target_on_left=False
                # print("target on right")

            while True:
                self.distractor_pos = np.concatenate([
                        np.random.uniform(low=-0.3, high=0, size=1),
                        np.random.uniform(low=y_range[0], high=y_range[1], size=1)])
                if np.linalg.norm(self.distractor_pos - self.goal_pos) > 0.17 and np.linalg.norm(self.obj_pos - self.distractor_pos) > 0.1:
                    break
            qpos[-6:-4] = self.distractor_pos.reshape(2,1)
        qpos[-4:-2] = self.obj_pos.reshape(2,1)
        qpos[-2:] = self.goal_pos.reshape(2,1)
        qvel = self.mujoco.init_qvel + np.random.uniform(low=-0.005,
                high=0.005, size=(self.mujoco.model.nv))

        #qvel[-4:] = 0
        #self.set_state(qpos, qvel)
        #return self._get_obs()

        setattr(self.mujoco.model.data, 'qpos', qpos)
        setattr(self.mujoco.model.data, 'qvel', qvel)
        self.mujoco.model.data.qvel = qvel
        self.mujoco.model._compute_subtree()
        self.mujoco.model.forward()
        self.current_com = self.mujoco.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self._get_obs()

    def get_current_image_obs(self):
        image = self.mujoco.viewer.get_image()
        pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
        pil_image = pil_image.resize((125,125), Image.ANTIALIAS)
        image = np.flipud(np.array(pil_image))
        return image, np.concatenate([
            self.mujoco.model.data.qpos.flat[:7],
            self.mujoco.model.data.qvel.flat[:7],
            self.mujoco.get_body_com('tips_arm'),
            self.mujoco.get_body_com('goal'),
            ])


    def _get_obs(self):
        if self.debug:
            return np.concatenate([
                self.mujoco.model.data.qpos.flat[:7],
                self.mujoco.model.data.qvel.flat[:7],
                self.mujoco.get_body_com("tips_arm"),
                self.mujoco.get_body_com("distractor"),
                self.mujoco.get_body_com("object"),
                self.mujoco.get_body_com("goal"),
            ])
        if self.include_distractors:
            if self.shuffle_order[0] == 0:
                return np.concatenate([
                    self.mujoco.model.data.qpos.flat[:7],
                    self.mujoco.model.data.qvel.flat[:7],
                    self.mujoco.get_body_com("tips_arm"),
                    self.mujoco.get_body_com("distractor"),
                    self.mujoco.get_body_com("object"),
                    self.mujoco.get_body_com("goal"),
                ])
            else:
                return np.concatenate([
                    self.mujoco.model.data.qpos.flat[:7],
                    self.mujoco.model.data.qvel.flat[:7],
                    self.mujoco.get_body_com("tips_arm"),
                    self.mujoco.get_body_com("object"),
                    self.mujoco.get_body_com("distractor"),
                    self.mujoco.get_body_com("goal"),
                ])
        else:
            assert False, "not supported"
            # if not self.onehot:
            #     # logger.log("debug3")
            #     return np.concatenate([
            #         self.mujoco.model.data.qpos.flat[:7],
            #         self.mujoco.model.data.qvel.flat[:7],
            #         self.mujoco.get_body_com("tips_arm"),
            #         self.mujoco.get_body_com("object"),
            #         self.mujoco.get_body_com("goal"),
            #     ])
            # else:
            #     # logger.log("debug4")
            #     extra = np.zeros(self.onehot_dim)
            #     if self.onehot_position == -1:
            #         pass  # we keep the vector zeroed out
            #     elif self.onehot_position in range(self.onehot_dim):
            #         extra[self.onehot_position] = 1.0
            #     else:
            #         assert False, "invalid value of self.onehot_position"
            #     return np.concatenate([
            #         self.mujoco.model.data.qpos.flat[:7],
            #         self.mujoco.model.data.qvel.flat[:7],
            #         self.mujoco.get_body_com("tips_arm"),
            #         self.mujoco.get_body_com("object"),
            #         self.mujoco.get_body_com("goal"),
            #         extra
            #     ])

    def render(self):
        self.mujoco.render()

    @property
    @overrides
    def observation_space(self):
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    def get_viewer(self):
        return self.mujoco.get_viewer()


    def log_diagnostics(self, paths, prefix=''):
        """
        Log extra information per iteration based on the collected paths
        """
        return self.mujoco.log_diagnostics(paths=paths,prefix=prefix)


def shuffle_demo(demoX):
    assert np.shape(demoX)[1] == 26, np.shape(demoX)
    slice1 = demoX[:,:17]  # qpos, qvel, tips arm
    slice2 = demoX[:,17:20] #distractor
    slice3 = demoX[:,20:23] #object
    slice4 = demoX[:,23:26] #goal
    return np.concatenate((slice1,slice3,slice2, slice4),-1)