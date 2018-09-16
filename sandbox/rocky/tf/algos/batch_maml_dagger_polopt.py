from rllab.algos.base import RLAlgorithm
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from rllab.sampler.stateful_pool import singleton_pool
from copy import deepcopy
import matplotlib
matplotlib.use('Pdf')
import itertools


import matplotlib.pyplot as plt
import os.path as osp
import rllab.misc.logger as logger
import rllab.plotter as plotter
import tensorflow as tf
import time
import numpy as np
import random as rd
import joblib
from rllab.misc.tensor_utils import split_tensor_dict_list, stack_tensor_dict_list
# from maml_examples.reacher_env import fingertip
from rllab.sampler.utils import rollout, joblib_dump_safe
from maml_examples.maml_experiment_vars import TESTING_ITRS, PLOT_ITRS, VIDEO_ITRS, BASELINE_TRAINING_ITRS
from maml_examples import pusher_env

class BatchMAMLPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods, with maml.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            experts_dir,
            experts_loader,
            metalearn_baseline=False,
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size=100,
            max_path_length=500,
            meta_batch_size=100,
            num_grad_updates=1,
            num_grad_updates_for_testing=1,
            discount=0.99,
            gae_lambda=1,
            beta_steps=1,
            beta_curve=None,
            plot=False,
            pause_for_plot=False,
            make_video=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            use_maml=True,
            use_maml_il=False,
            test_on_training_goals=False,
            limit_demos_num=None,
            test_goals_mult=1,
            load_policy=None,
            pre_std_modifier=1.0,
            post_std_modifier_train=1.0,
            post_std_modifier_test=1.0,
            goals_pool_to_load=None,
            goals_pickle_to=None,
            extra_input=None,
            extra_input_dim=0,
            seed=1,
            debug_pusher=False,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.  #
        :param max_path_length: Maximum length of a single rollout.
        :param meta_batch_size: Number of tasks sampled per meta-update
        :param num_grad_updates: Number of fast gradient updates
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.seed=seed
        self.env = env
        self.policy = policy
        self.load_policy = load_policy
        self.baseline = baseline
        self.metalearn_baseline = metalearn_baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        # batch_size is the number of trajectories for one fast grad update.
        # self.batch_size is the number of total transitions to collect.
        self.batch_size = batch_size * max_path_length * meta_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.beta_steps = beta_steps
        self.beta_curve = beta_curve if beta_curve is not None else [self.beta_steps]
        self.old_il_loss = None
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.make_video = make_video
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.meta_batch_size = meta_batch_size  # number of tasks
        self.num_grad_updates = num_grad_updates  # number of gradient steps during training
        self.num_grad_updates_for_testing = num_grad_updates_for_testing  # number of gradient steps during training
        self.use_maml_il = use_maml_il
        self.test_on_training_goals= test_on_training_goals
        self.testing_itrs = TESTING_ITRS
        if self.metalearn_baseline:
            self.testing_itrs.insert(0,0)
        logger.log("test_on_training_goals %s" % self.test_on_training_goals)
        self.limit_demos_num = limit_demos_num
        self.test_goals_mult = test_goals_mult
        self.pre_std_modifier = pre_std_modifier
        self.post_std_modifier_train = post_std_modifier_train
        self.post_std_modifier_test = post_std_modifier_test
        #   self.action_limiter_multiplier = action_limiter_multiplier
        self.extra_input = extra_input
        self.extra_input_dim = extra_input_dim
        self.debug_pusher=debug_pusher
        self.experts_dir = experts_dir
        self.experts_loader = experts_loader

        # setup goals
        print("debug1", tf.__version__)
        self._setup_goals(goals_pool_to_load, goals_pickle_to=None)

        # setup sampler
        if sampler_cls is None:
            if singleton_pool.n_parallel > 1:
                sampler_cls = BatchSampler
                print("Using Batch Sampler")
            else:
                sampler_cls = VectorizedSampler
                print("Using Vectorized Sampler")
        if sampler_args is None:
            sampler_args = dict()
        if 'n_envs' not in sampler_args.keys():
            sampler_args['n_envs'] = self.meta_batch_size
        self.sampler = sampler_cls(self, **sampler_args)

        # setup experts
        self.experts = []
        for expert_num_path in glob.glob('{}/*'.format(self.experts_dir)):
            model_num_paths = glob.glob('{}/checkpoints/model*.pkl'.format(expert_num_path)
            get_model_num = int(model_num_path.split('/')[-1].split('model')[-1].split('.pkl')[0])
            lastest_model_num = max(map(get_model_num, model_num_paths))
            self.experts.append(self.expert_loader(
                '{}/checkpoints/model{}.pkl'.format(expert_num_path, latest_model_num)))

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr, reset_args=None, log_prefix='', preupdate=False):
        # This obtains samples using self.policy, and calling policy.get_actions(obses)
        # return_dict specifies how the samples should be returned (dict separates samples
        # by task)
        paths = self.sampler.obtain_samples(itr=itr, reset_args=reset_args, return_dict=True, 
            log_prefix=log_prefix, extra_input=self.extra_input, 
            extra_input_dim=(self.extra_input_dim if self.extra_input is not None else 0), preupdate=preupdate)
        assert type(paths) == dict
        return paths

    def query_expert(self, itr, paths, log_prefix=''):
        for task_num in paths.keys():
            for path in paths[task_num]:
                path['expert_actions'], _, _, _ = self.experts[task_num].detstep(path['observations'])
        return paths

    def process_samples(self, itr, paths, prefix='', log=True, fast_process=False, testitr=False, metalearn_baseline=False):
        return self.sampler.process_samples(itr, paths, prefix=prefix, log=log, fast_process=fast_process, testitr=testitr, metalearn_baseline=metalearn_baseline)

    def train(self):
        # TODO - make this a util
        flatten_list = lambda l: [item for sublist in l for item in sublist]
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        # with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        with tf.Session(config=config) as sess:
            tf.set_random_seed(1)
            # Code for loading a previous policy. Somewhat hacky because needs to be in sess.
            if self.load_policy is not None:
                self.policy = joblib.load(self.load_policy)['policy']
            self.init_opt()
            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = []
            # sess.run(tf.global_variables_initializer())
            for var in tf.global_variables():
                # note - this is hacky, may be better way to do this in newer TF.
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninit_vars.append(var)
            sess.run(tf.variables_initializer(uninit_vars))
            self.start_worker()
            start_time = time.time()
            self.metaitr=0
            # epochs
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                np.random.seed(self.seed+itr)
                tf.set_random_seed(self.seed+itr)
                rd.seed(self.seed+itr)
                with logger.prefix('itr #%d | ' % itr):
                    all_paths_for_plotting = []
                    all_postupdate_paths = []
                    self.beta_steps = min(self.beta_steps, self.beta_curve[min(itr,len(self.beta_curve)-1)])
                    beta_steps_range = range(self.beta_steps) if itr not in self.testing_itrs else range(self.test_goals_mult)
                    beta0_step0_paths = None
                    num_inner_updates = self.num_grad_updates_for_testing if itr in self.testing_itrs else self.num_grad_updates
                    # outer loop
                    for beta_step in beta_steps_range:
                        all_samples_data_for_betastep = []
                        print("debug, pre-update std modifier")
                        self.policy.std_modifier = self.pre_std_modifier
                        self.policy.switch_to_init_dist()  # Switch to pre-update policy
                        if itr in self.testing_itrs:
                            env = self.env
                            while 'sample_goals' not in dir(env):
                                env = env.wrapped_env
                            # Note: FinnMamlEnv always returns list of all goals
                            self.goals_to_use_dict[itr] = env.sample_goals(self.meta_batch_size) 
                        # inner loop
                        for step in range(num_inner_updates+1):
                            # gather samples for all inner & outer loop updates
                            logger.log('** Betastep %s ** Step %s **' % (str(beta_step), str(step)))
                            logger.log("Obtaining samples...")
                            if itr in self.testing_itrs:
                                if step < num_inner_updates:
                                    print('debug12.0.0, test-time sampling step=', step) #, goals_to_use)
                                    paths = self.obtain_samples(itr=itr, reset_args=self.goals_to_use_dict[itr],
                                                                    log_prefix=str(beta_step) + "_" + str(step), preupdate=True)
                                    paths = store_agent_infos(paths)  # agent_infos_orig is populated here
                                elif step == num_inner_updates:
                                    print('debug12.0.1, test-time sampling step=', step) #, goals_to_use)
                                    paths = self.obtain_samples(itr=itr, reset_args=self.goals_to_use_dict[itr],
                                                                    log_prefix=str(beta_step) + "_" + str(step), preupdate=False)
                                    all_postupdate_paths.extend(paths.values())
                            else:
                                if beta_step==0 and step < num_inner_updates:
                                    print("debug12.1, sample on-policy") #, self.goals_to_use_dict[itr])
                                    paths = self.obtain_samples(itr=itr, reset_args=self.goals_to_use_dict[itr], 
                                        log_prefix=str(beta_step)+"_"+str(step), preupdate=True)
                                    if beta_step == 0 and step == 0:
                                        paths = store_agent_infos(paths)  # agent_infos_orig is populated here
                                        beta0_step0_paths = deepcopy(paths)
                                elif step == num_inner_updates:
                                    print("debug12.2, query expert for on-policy sampled traj")
                                    paths = self.obtain_samples(itr=itr, reset_args=self.goals_to_use_dict[itr],
                                                                    log_prefix=str(beta_step) + "_" + str(step), preupdate=False)
                                    paths = self.query_expert(itr=itr, paths=deepcopy(paths), log_prefix=str(beta_step)+"_"+str(step))

                            # process samples
                            all_paths_for_plotting.append(paths)
                            logger.log("Processing samples...")
                            samples_data = {}
                            for tasknum in paths.keys():  # the keys are the tasks
                                # don't log because this will spam the console with every task.
                                if self.use_maml_il and step == num_inner_updates:
                                    fast_process = True
                                else:
                                    fast_process = False
                                if itr in self.testing_itrs:
                                    testitr = True
                                else:
                                    testitr = False
                                samples_data[tasknum] = self.process_samples(itr, paths[tasknum], log=False, 
                                    fast_process=fast_process, testitr=testitr, metalearn_baseline=self.metalearn_baseline)

                            all_samples_data_for_betastep.append(samples_data)

                            # logging
                            self.process_samples(itr, flatten_list(paths.values()), prefix=str(step), log=True, 
                                fast_process=True, testitr=testitr, metalearn_baseline=self.metalearn_baseline)
                            if itr not in self.testing_itrs:
                                self.log_diagnostics(flatten_list(paths.values()), prefix=str(step))

                            if step == num_inner_updates:
                                logger.record_tabular("AverageReturnLastTest", self.sampler.memory["AverageReturnLastTest"],front=True)  #TODO: add functionality for multiple grad steps
                                logger.record_tabular("TestItr", ("1" if testitr else "0"),front=True)
                                logger.record_tabular("MetaItr", self.metaitr,front=True)
                           
                            if step == num_inner_updates-1:
                                if itr not in self.testing_itrs:
                                    print("debug, post update train std modifier")
                                    self.policy.std_modifier = self.post_std_modifier_train*self.policy.std_modifier
                                else:
                                    print("debug, post update test std modifier")
                                    self.policy.std_modifier = self.post_std_modifier_test*self.policy.std_modifier
                                if (itr in self.testing_itrs or not self.use_maml_il or step<num_inner_updates-1) and step < num_inner_updates:
                                    # do not update on last grad step, and do not update on second to last step when training MAMLIL
                                    logger.log("Computing policy updates...")
                                    self.policy.compute_updated_dists(samples=samples_data)

                        # optimize policy
                        logger.log("Optimizing policy...")
                        start_loss = self.optimize_policy(itr, all_samples_data_for_betastep)

                    if itr not in self.testing_itrs:              
                        self.metaitr += 1

                    # save
                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr, all_samples_data_for_betastep[-1])  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = all_samples_data_for_betastep[-1]["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular('ItrTime', time.time() - itr_start_time)
                    logger.dump_tabular(with_prefix=False)

                    # plot
                    self.plot(itr)

        self.shutdown_worker()

    def log_diagnostics(self, paths, prefix):
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)

    def plot(self, itr):
        # wtf is this shit
        # The rest is some example plotting code.
        # Plotting code is useful for visualizing trajectories across a few different tasks.
        if True and itr in PLOT_ITRS and self.env.observation_space.shape[0] == 2: # point-mass
            logger.log("Saving visualization of paths")
            for ind in range(min(5, self.meta_batch_size)):
                plt.clf()
                plt.plot(self.goals_to_use_dict[itr][ind][0], self.goals_to_use_dict[itr][ind][1], 'k*', markersize=10)
                plt.hold(True)

                preupdate_paths = all_paths_for_plotting[0]
                postupdate_paths = all_paths_for_plotting[-1]

                pre_points = preupdate_paths[ind][0]['observations']
                post_points = postupdate_paths[ind][0]['observations']
                plt.plot(pre_points[:,0], pre_points[:,1], '-r', linewidth=2)
                plt.plot(post_points[:,0], post_points[:,1], '-b', linewidth=1)

                pre_points = preupdate_paths[ind][1]['observations']
                post_points = postupdate_paths[ind][1]['observations']
                plt.plot(pre_points[:,0], pre_points[:,1], '--r', linewidth=2)
                plt.plot(post_points[:,0], post_points[:,1], '--b', linewidth=1)

                pre_points = preupdate_paths[ind][2]['observations']
                post_points = postupdate_paths[ind][2]['observations']
                plt.plot(pre_points[:,0], pre_points[:,1], '-.r', linewidth=2)
                plt.plot(post_points[:,0], post_points[:,1], '-.b', linewidth=1)

                plt.plot(0,0, 'k.', markersize=5)
                plt.xlim([-0.8, 0.8])
                plt.ylim([-0.8, 0.8])
                plt.legend(['goal', 'preupdate path', 'postupdate path'])
                plt.savefig(osp.join(logger.get_snapshot_dir(), 'prepost_path' + str(ind) + '_' + str(itr) + '.png'))
                print(osp.join(logger.get_snapshot_dir(), 'prepost_path' + str(ind) + '_' + str(itr) + '.png'))
        elif True and itr in PLOT_ITRS and self.env.observation_space.shape[0] == 8:  # 2D reacher
            logger.log("Saving visualization of paths")

            # def fingertip(env):
            #     while 'get_body_com' not in dir(env):
            #         env = env.wrapped_env
            #     return env.get_body_com('fingertip')

            for ind in range(min(5, self.meta_batch_size)):
                plt.clf()
                print("debug13,",itr,ind)
                a = self.goals_to_use_dict[itr][ind]
                plt.plot(self.goals_to_use_dict[itr][ind][0], self.goals_to_use_dict[itr][ind][1], 'k*', markersize=10)
                plt.hold(True)

                preupdate_paths = all_paths_for_plotting[0]
                postupdate_paths = all_paths_for_plotting[-1]

                pre_points = np.array([obs[6:8] for obs in preupdate_paths[ind][0]['observations']])
                post_points = np.array([obs[6:8] for obs in postupdate_paths[ind][0]['observations']])
                plt.plot(pre_points[:,0], pre_points[:,1], '-r', linewidth=2)
                plt.plot(post_points[:,0], post_points[:,1], '-b', linewidth=1)

                pre_points = np.array([obs[6:8] for obs in preupdate_paths[ind][1]['observations']])
                post_points = np.array([obs[6:8] for obs in postupdate_paths[ind][1]['observations']])
                plt.plot(pre_points[:,0], pre_points[:,1], '--r', linewidth=2)
                plt.plot(post_points[:,0], post_points[:,1], '--b', linewidth=1)

                pre_points = np.array([obs[6:8] for obs in preupdate_paths[ind][2]['observations']])
                post_points = np.array([obs[6:8] for obs in postupdate_paths[ind][2]['observations']])
                plt.plot(pre_points[:,0], pre_points[:,1], '-.r', linewidth=2)
                plt.plot(post_points[:,0], post_points[:,1], '-.b', linewidth=1)

                plt.plot(0,0, 'k.', markersize=5)
                plt.xlim([-0.25, 0.25])
                plt.ylim([-0.25, 0.25])
                plt.legend(['goal', 'preupdate path', 'postupdate path'])
                plt.savefig(osp.join(logger.get_snapshot_dir(), 'prepost_path' + str(ind) + '_' + str(itr) + '.png'))
                print(osp.join(logger.get_snapshot_dir(), 'prepost_path' + str(ind) + '_' + str(itr) + '.png'))

                if self.make_video and itr in VIDEO_ITRS:
                    logger.log("Saving videos...")
                    self.env.reset(reset_args=self.goals_to_use_dict[itr][ind])
                    video_filename = osp.join(logger.get_snapshot_dir(), 'post_path_%s_%s.gif' % (ind, itr))
                    rollout(env=self.env, agent=self.policy, max_path_length=self.max_path_length,
                            animated=True, speedup=2, save_video=True, video_filename=video_filename,
                            reset_arg=self.goals_to_use_dict[itr][ind],
                            use_maml=True, maml_task_index=ind,
                            maml_num_tasks=self.meta_batch_size)
        elif self.make_video and itr in VIDEO_ITRS:
            for ind in range(min(2, self.meta_batch_size)):
                logger.log("Saving videos...")
                self.env.reset(reset_args=self.goals_to_use_dict[itr][ind])
                video_filename = osp.join(logger.get_snapshot_dir(), 'post_path_%s_%s.gif' % (ind, itr))
                rollout(env=self.env, agent=self.policy, max_path_length=self.max_path_length,
                        animated=True, speedup=2, save_video=True, video_filename=video_filename,
                        reset_arg=self.goals_to_use_dict[itr][ind],
                        use_maml=True, maml_task_index=ind,
                        maml_num_tasks=self.meta_batch_size, extra_input_dim=self.extra_input_dim)
            self.policy.switch_to_init_dist()
            for ind in range(min(2, self.meta_batch_size)):
                logger.log("Saving videos...")
                self.env.reset(reset_args=self.goals_to_use_dict[itr][ind])
                video_filename = osp.join(logger.get_snapshot_dir(), 'pre_path_%s_%s.gif' % (ind, itr))
                rollout(env=self.env, agent=self.policy, max_path_length=self.max_path_length,
                        animated=True, speedup=2, save_video=True, video_filename=video_filename,
                        reset_arg=self.goals_to_use_dict[itr][ind],
                        use_maml=False,
                        extra_input_dim = self.extra_input_dim,
                        # maml_task_index=ind,
                        # maml_num_tasks=self.meta_batch_size
                        )
        elif False and itr in PLOT_ITRS:  # swimmer or cheetah
            logger.log("Saving visualization of paths")
            for ind in range(min(5, self.meta_batch_size)):
                plt.clf()
                goal_vel = self.goals_to_use_dict[itr][ind]
                plt.title('Swimmer paths, goal vel='+str(goal_vel))
                plt.hold(True)

                prepathobs = all_paths_for_plotting[0][ind][0]['observations']
                postpathobs = all_paths_for_plotting[-1][ind][0]['observations']
                plt.plot(prepathobs[:,0], prepathobs[:,1], '-r', linewidth=2)
                plt.plot(postpathobs[:,0], postpathobs[:,1], '--b', linewidth=1)
                plt.plot(prepathobs[-1,0], prepathobs[-1,1], 'r*', markersize=10)
                plt.plot(postpathobs[-1,0], postpathobs[-1,1], 'b*', markersize=10)
                plt.xlim([-1.0, 5.0])
                plt.ylim([-1.0, 1.0])

                plt.legend(['preupdate path', 'postupdate path'], loc=2)
                plt.savefig(osp.join(logger.get_snapshot_dir(), 'swim1d_prepost_itr' + str(itr) + '_id' + str(ind) + '.pdf'))

    def _setup_goals(self, goals_pool_to_load, goals_pickle_to):
        if goals_pool_to_load is not None:
            # load goals
            logger.log("Loading goals pool from %s ..." % goals_pool_to_load)
            self.goals_pool = joblib.load(goals_pool_to_load)['goals_pool']
            self.goals_idxs_for_itr_dict = joblib.load(goals_pool_to_load)['idxs_dict']
        else:
            # build goals pool and idxs_dict
            goals_pool_size = (self.n_itr-self.start_itr)*self.meta_batch_size
            logger.log("Sampling a pool of tasks/goals for this meta-batch...")
            env = self.env
            while 'sample_goals' not in dir(env):
               env = env.wrapped_env
            self.goals_pool = env.sample_goals(goals_pool_size)
            self.goals_idxs_for_itr_dict = {}
            for itr in range(self.start_itr, self.n_itr):
                self.goals_idxs_for_itr_dict[itr] = rd.sample(range(goals_pool_size), self.meta_batch_size)

            # save goals pool
            if goals_pickle_to is not None:
                # logger.log("Saving goals to %s..." % goals_pickle_to)
                # joblib_dump_safe(self.goals_to_use_dict, goals_pickle_to)
                logger.log("Saving goals pool to %s..." % goals_pickle_to)
                joblib_dump_safe(dict(goals_pool=self.goals_pool, idxs_dict=self.goals_idxs_for_itr_dict), goals_pickle_to)

        # inspect goals pool
        env = self.env
        while 'sample_goals' not in dir(env):
            env = env.wrapped_env
        reset_dimensions = env.sample_goals(1).shape[1:]
        dimensions = np.shape(self.goals_pool[self.goals_idxs_for_itr_dict[0][0]])
        assert reset_dimensions == dimensions, "loaded dimensions are %s, do not match with environment's %s" % (
            dimensions, reset_dimensions)

        # inspect goals_idxs_for_itr_dict
        assert set(range(self.start_itr, self.n_itr)).issubset(set(self.goals_idxs_for_itr_dict.keys())), \
            "Not all meta-iteration numbers have idx_dict in %s" % goals_pool_to_load
        for itr in range(self.start_itr, self.n_itr):
            num_goals = len(self.goals_idxs_for_itr_dict[itr])
            assert num_goals >= self.meta_batch_size, "iteration %s contained %s goals when at least %s are needed" % (itr, num_goals, self.meta_batch_size)
            self.goals_idxs_for_itr_dict[itr] = self.goals_idxs_for_itr_dict[itr][:self.meta_batch_size]

        # build goals_to_use_dict
        self.goals_to_use_dict = {}
        for itr in range(self.start_itr, self.n_itr):
            if itr not in self.testing_itrs or self.test_on_training_goals:
                self.goals_to_use_dict[itr] = np.array([self.goals_pool[idx] for idx in self.goals_idxs_for_itr_dict[itr]])

def store_agent_infos(paths):
    tasknums = paths.keys()
    for t in tasknums:
        for path in paths[t]:
            path['agent_infos_orig'] = deepcopy(path['agent_infos'])
    return paths

