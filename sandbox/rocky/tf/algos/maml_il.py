import tensorflow as tf
import numpy as np
import rllab.misc.logger as logger
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.batch_maml_polopt import BatchMAMLPolopt
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.optimizers.quad_dist_expert_optimizer import QuadDistExpertOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from maml_examples.maml_experiment_vars import TESTING_ITRS, BASELINE_TRAINING_ITRS
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors
from rllab.misc.ext import sliced_fun
from collections import OrderedDict


class MAMLIL(BatchMAMLPolopt):

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            use_maml=True,
            use_vision=False,
            use_corr_term=True,
            beta_steps=1,
            adam_steps=1,
            adam_curve=None,
            l2loss_std_mult=1.0,
            importance_sampling_modifier=tf.identity,
            metalearn_baseline=False,
            penalty=0.05,
            constrain_against_central = True,
            constrain_together = False,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict(min_penalty=1e-8)
            optimizer = QuadDistExpertOptimizer("main_optimizer", adam_steps=adam_steps, use_momentum_optimizer=False)  #  **optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.adam_curve = adam_curve if adam_curve is not None else [adam_steps]
        self.use_maml = use_maml
        self.use_vision = use_vision
        self.use_corr_term=use_corr_term
        self.kl_constrain_step = -1
        self.l2loss_std_multiplier = l2loss_std_mult
        self.ism = importance_sampling_modifier
        self.old_start_il_loss = None
        self.metalearn_baseline = metalearn_baseline
        if "extra_input" in kwargs.keys():
            self.extra_input = kwargs["extra_input"]
        else:
            self.extra_input = ""
        if "extra_input_dim" in kwargs.keys():
            self.extra_input_dim = kwargs["extra_input_dim"]
        else:
            self.extra_input_dim = 0

        self.penalty = penalty
        self.constrain_together = constrain_together
        self.constrain_against_central = constrain_against_central

        super(MAMLIL, self).__init__(optimizer=optimizer, beta_steps=beta_steps, use_maml_il=True, metalearn_baseline=metalearn_baseline, **kwargs)


    def make_vars(self, stepnum='0'):
        # lists over the meta_batch_size
        # We should only need the last stepnum for meta-optimization.
        obs_vars, action_vars, adv_vars, rewards_vars, returns_vars, path_lengths_vars, expert_action_vars = [], [], [], [], [], [], []
        for i in range(self.meta_batch_size):
            obs_vars.append(self.env.observation_space.new_tensor_variable(
                'obs' + stepnum + '_' + str(i),
                extra_dims=1,
                add_to_flat_dim=(0 if self.extra_input is None else self.extra_input_dim),
            ))
            action_vars.append(self.env.action_space.new_tensor_variable(
                'action' + stepnum + '_' + str(i),
                extra_dims=1,
            ))
            adv_vars.append(tensor_utils.new_tensor(
                    'advantage' + stepnum + '_' + str(i),
                    ndim=1, dtype=tf.float32,
                ))
            if self.metalearn_baseline:
                rewards_vars.append(tensor_utils.new_tensor(
                    'rewards' + stepnum + '_' + str(i),
                    ndim=1, dtype=tf.float32,
                ))
                returns_vars.append(tensor_utils.new_tensor(
                    'returns' + stepnum + '_' + str(i),
                    ndim=1, dtype=tf.float32,
                ))
                # path_lengths_vars.append(tensor_utils.new_tensor(
                #     'path_lengths' + stepnum + '_' + str(i),
                #     ndim=1, dtype=tf.float32,
                # ))
            expert_action_vars.append(self.env.action_space.new_tensor_variable(
                name='expert_actions' + stepnum + '_' + str(i),
                extra_dims=1,
            ))
        if not self.metalearn_baseline:
            return obs_vars, action_vars, adv_vars, expert_action_vars
        else:
            return obs_vars, action_vars, adv_vars, rewards_vars, returns_vars, expert_action_vars # path_lengths_vars before expert action


    @overrides
    def init_opt(self):
        assert not int(self.policy.recurrent)  # not supported
        assert self.use_maml  # only maml supported

        dist = self.policy.distribution


        old_dist_info_vars, old_dist_info_vars_list = [], []
        for i in range(self.meta_batch_size):
            old_dist_info_vars.append({
                k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s_%s' % (i, k))
                for k, shape in dist.dist_info_specs
                })
            old_dist_info_vars_list += [old_dist_info_vars[i][k] for k in dist.dist_info_keys]

        theta0_dist_info_vars, theta0_dist_info_vars_list = [], []
        for i in range(self.meta_batch_size):
            theta0_dist_info_vars.append({
                k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='theta0_%s_%s' % (i, k))
                for k, shape in dist.dist_info_specs
                })
            theta0_dist_info_vars_list += [theta0_dist_info_vars[i][k] for k in dist.dist_info_keys]

        theta_l_dist_info_vars, theta_l_dist_info_vars_list = [], []  #theta_l is the current beta step's pre-inner grad update params
        for i in range(self.meta_batch_size):
            theta_l_dist_info_vars.append({
                k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='theta_l_%s_%s' % (i, k))
                for k, shape in dist.dist_info_specs
                })
            theta_l_dist_info_vars_list += [theta_l_dist_info_vars[i][k] for k in dist.dist_info_keys]


        state_info_vars, state_info_vars_list = {}, []  # TODO: is this needed?

        all_surr_objs, all_surr_objs_slow, input_vars_list, inner_input_vars_list = [], [], [], []
        new_params = []
        old_logli_sym = []
        old_lr = []
        old_adv = []
        old_action_vars = []
        old_obs_vars = []
        input_vars_list += tuple(theta0_dist_info_vars_list) + tuple(theta_l_dist_info_vars_list)
        inner_input_vars_list += tuple(theta0_dist_info_vars_list) + tuple(theta_l_dist_info_vars_list)

        for grad_step in range(self.num_grad_updates):  # we are doing this for all but the last step
            if not self.metalearn_baseline:
                obs_vars, action_vars, adv_vars, expert_action_vars = self.make_vars(str(grad_step))
            else:
                obs_vars, action_vars, adv_vars, rewards_vars, returns_vars, expert_action_vars = self.make_vars(str(grad_step))  # path_lengths_vars before expert actions

            inner_surr_objs, inner_surr_objs_simple, inner_surr_objs_sym = [], [], []  # surrogate objectives
            # inner_surr_objs_slow = []  # surrogate objectives

            new_params = []
            kls = []
            old_logli_sym.append([])
            old_lr.append([])
            old_adv.append([])
            old_action_vars.append([])
            old_obs_vars.append([])

            al_const = tf.constant(np.arange(self.max_path_length).reshape(-1, 1) / 100.0)
            al = tf.tile(al_const, tf.stack([tf.cast(tf.shape(obs_vars[0])[0] / self.max_path_length, tf.int32), 1]))
            al = tf.cast(al, dtype=tf.float32)

            self.central_policy_dist_infos = []
            for i in range(self.meta_batch_size):  # for training task T_i
                adv = adv_vars[i]
                if self.metalearn_baseline:


                    # al = tf.concat([al_const]*int(self.batch_size/self.max_path_length/self.meta_batch_size),0)
                    enh_obs_i = tf.concat([obs_vars[i]*self.baseline.obs_mask,
                                           (obs_vars[i] ** 2)*self.baseline.obs_mask,
                                           al, al ** 2, al ** 3], axis=1)
                    # enh_obs_i = tf.concat([al, al ** 2, al ** 3], axis=1)

                    if 'surr_obj' not in dir(self.baseline):
                        assert i == 0
                        self.baseline.set_init_surr_obj(input_list=[enh_obs_i]+ [returns_vars[i]], surr_obj_tensor=None)  # we define the surr obj in the baseline
                    adv_sym = self.baseline.build_adv_sym(enh_obs_vars=enh_obs_i,
                                                      rewards_vars=rewards_vars[i],
                                                      returns_vars=returns_vars[i],
                                                      # baseline_pred_loss=baseline_pred_loss_i,
                                                      # path_lengths_vars=path_lengths_vars[i],
                                                      all_params=self.baseline.all_params)

                dist_info_sym_i, params = self.policy.dist_info_sym(obs_vars[i], state_info_vars, all_params=self.policy.all_params)
                self.central_policy_dist_infos.append([dist_info_sym_i , obs_vars[i]])

                if self.kl_constrain_step == 0:
                    kl = dist.kl_sym(old_dist_info_vars[i], dist_info_sym_i)
                    kls.append(kl)
                new_params.append(params)
                logli_i = dist.log_likelihood_sym(action_vars[i], dist_info_sym_i)
            
                keys = self.policy.all_params.keys()
                theta_circle = OrderedDict({key: tf.stop_gradient(self.policy.all_params[key]) for key in keys})
                dist_info_sym_i_circle, _ = self.policy.dist_info_sym(obs_vars[i], state_info_vars, all_params=theta_circle)
                lr_per_step_fast = dist.likelihood_ratio_sym(action_vars[i], theta0_dist_info_vars[i], dist_info_sym_i_circle)
                lr_per_step_fast = self.ism(lr_per_step_fast)

                old_logli_sym[-1].append(logli_i)
                old_lr[-1].append(lr_per_step_fast)
                if not self.metalearn_baseline:
                    old_adv[-1].append(adv)
                else:
                    old_adv[-1].append(adv_sym)
                old_action_vars[-1].append(action_vars[i])
                old_obs_vars[-1].append(obs_vars[i])
                # formulate a minimization problem
                # The gradient of the surrogate objective is the policy gradient
                # inner_surr_objs.append(-tf.reduce_mean(tf.multiply(tf.multiply(logli_i, lr_by_path), adv)))
                # inner_surr_objs.append(-tf.reduce_mean(tf.multiply(tf.multiply(logli_i, 1.0), adv)))
                inner_surr_objs.append(-tf.reduce_mean(tf.multiply(tf.multiply(logli_i, lr_per_step_fast), adv)))
                inner_surr_objs_simple.append(-tf.reduce_mean(tf.multiply(logli_i, adv)))
                # inner_surr_objs_slow.append(-tf.reduce_mean(tf.multiply(tf.multiply(logli_i, lr_per_step_slow), adv)))
                # inner_surr_objs.append(-tf.reduce_mean(tf.multiply(logli_i, adv)))
                if self.metalearn_baseline:
                    inner_surr_objs_sym.append(-tf.reduce_mean(tf.multiply(tf.multiply(logli_i, lr_per_step_fast), adv_sym)))
            inner_input_vars_list += obs_vars + action_vars + adv_vars
            if not self.metalearn_baseline:
                input_vars_list += obs_vars + action_vars + adv_vars
            else:
                input_vars_list += obs_vars + action_vars + rewards_vars + returns_vars  # + path_lengths_vars
            # For computing the fast update for sampling
            # At this point, inner_input_vars_list is theta0 + theta_l + obs + action + adv
            self.policy.set_init_surr_obj(inner_input_vars_list, inner_surr_objs_simple)

            input_vars_list += expert_action_vars # TODO: is this pre-update expert action vars? Should we kill this?
            if not self.metalearn_baseline:
                all_surr_objs.append(inner_surr_objs)
                # all_surr_objs_slow.append(inner_surr_objs_slow)
            else:
                all_surr_objs.append(inner_surr_objs_sym)

        # LAST INNER GRAD STEP
        if not self.metalearn_baseline:
            obs_vars, action_vars, _, expert_action_vars = self.make_vars('test')  # adv_vars was here instead of _
        else:
            obs_vars, action_vars, _, _, _, expert_action_vars = self.make_vars('test')
        outer_surr_objs = []
        # outer_surr_objs_slow = []
        # old_outer_surr_objs = []
        updated_params = []
        for i in range(self.meta_batch_size):  # here we cycle through the last grad update but for validation tasks (i is the index of a task)
            # old_dist_info_sym_i, _ = self.policy.dist_info_sym(obs_vars[i], state_info_vars,all_params=self.policy.all_params)
            # import pprint
            # pp = pprint.PrettyPrinter()
            # pp.pprint(("debug, new_params[i]",new_params[i]))
            dist_info_sym_i, updated_params_i = self.policy.updated_dist_info_sym(task_id=i,surr_obj=all_surr_objs[-1][i],new_obs_var=obs_vars[i], params_dict=new_params[i])
            # pp.pprint(("debug, updated_params_i",updated_params_i))
            # dist_info_sym_i_slow, _ = self.policy.updated_dist_info_sym(task_id=i,surr_obj=all_surr_objs_slow[-1][i],new_obs_var=obs_vars[i], params_dict=new_params[i])
            if self.kl_constrain_step == -1:  # if we only care about the kl of the last step, the last item in kls will be the overall
                kl = dist.kl_sym(old_dist_info_vars[i], dist_info_sym_i)
                kls.append(kl)  # we either get kl from here or from kl_constrain_step =0

            updated_params.append(updated_params_i)
            # # here we define the loss for meta-gradient
            a_star = expert_action_vars[i]
            s = dist_info_sym_i["log_std"]
            m = dist_info_sym_i["mean"]
            outer_surr_obj = tf.reduce_mean(m**2 - 2*m*a_star+a_star**2+self.l2loss_std_multiplier*(tf.square(tf.exp(s))))
            outer_surr_objs.append(outer_surr_obj)

        outer_surr_obj = tf.reduce_mean(tf.stack(outer_surr_objs, 0))  # mean over all the different tasks
        # outer_surr_obj_slow = tf.reduce_mean(tf.stack(outer_surr_objs_slow, 0))  # mean over all the different tasks
        input_vars_list += obs_vars + action_vars + expert_action_vars + old_dist_info_vars_list  # +adv_vars # TODO: kill action_vars from this list, and if we're not doing kl, kill old_dist_info_vars_list too
        mean_kl = tf.cast(tf.reduce_mean(tf.concat(kls, 0)),tf.float32)

        # CORRECTION TERM ATTEMPT 2
        if self.use_corr_term:
            term1_list = []
            keys = self.policy.all_params.keys()
            theta_triangle = OrderedDict({key: self.policy.all_params[key] * 1.0 for key in keys})
            theta_box = OrderedDict({key: self.policy.all_params[key] * 1.0 for key in keys})
            for i in range(self.meta_batch_size):

                def grads_dotprod(A, B):
                    return tf.reduce_sum([tf.reduce_sum(a * b) for a, b in zip(A, B)])

                print("debug, constructing corr term for task", i)

                term0_i = tf.gradients(outer_surr_objs[i],[updated_params[i][key] for key in keys])
                dist_info_sym_i_triangle, _ = self.policy.dist_info_sym(old_obs_vars[0][i], state_info_vars,all_params=theta_triangle)
                dist_info_sym_i_box, _ = self.policy.dist_info_sym(old_obs_vars[0][i], state_info_vars,all_params=theta_box)
                logli_i_triangle = dist.log_likelihood_sym(old_action_vars[0][i], dist_info_sym_i_triangle)
                logli_i_box = dist.log_likelihood_sym(old_action_vars[0][i], dist_info_sym_i_box)
                L = tf.reduce_mean(logli_i_triangle * old_adv[0][i] * old_lr[0][i] * logli_i_box)
                term1_i = grads_dotprod(term0_i, tf.gradients(L, [theta_triangle[key] for key in keys]))
                term1_list.append(term1_i)

            corr_term = OrderedDict(zip([self.policy.all_params[key] for key in keys],tf.gradients(tf.reduce_mean(term1_list) * self.policy.step_size, [theta_box[key] for key in keys])))  #TODO: need to test it with the step size
        else:
            corr_term = None

        if self.metalearn_baseline:
            target=[self.policy.all_params[key] for key in self.policy.all_params.keys()] + [self.baseline.all_params['meta_constant']]
            # target = [self.policy.all_params[key] for key in self.policy.all_params.keys()] + [self.baseline.all_params[key] for key in self.baseline.all_params.keys()]
            # target=[self.policy.all_params[key] for key in self.policy.all_params.keys()]
        else:
            if not self.use_vision:
                target = [self.policy.all_params[key] for key in self.policy.all_params.keys()]
            else:
                target = [self.policy.get_params_internal()]

        self.optimizer.update_opt(
            loss=outer_surr_obj,
            # dummy_loss = outer_surr_obj_slow,
            target=target,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_vars_list,
            constraint_name="mean_kl",
            correction_term=corr_term
        )

        return dict()
#######################################

    @overrides
    def init_experts_opt(self):

        ###############################
        #
        # Variable Definitions
        #
        ###############################

        all_task_dist_info_vars = []
        all_obs_vars = []

        for i, policy in enumerate(self.local_policies):

            task_obs_var = self.env_partitions[i].observation_space.new_tensor_variable('obs%d' % i, extra_dims=1)
            task_dist_info_vars = []

            for j, other_policy in enumerate(self.local_policies):

                state_info_vars = dict()  # Not handling recurrent policies
                dist_info_vars = other_policy.dist_info_sym(task_obs_var, state_info_vars)
                task_dist_info_vars.append(dist_info_vars)

            all_obs_vars.append(task_obs_var)
            all_task_dist_info_vars.append(task_dist_info_vars)

        obs_var = self.env.observation_space.new_tensor_variable('obs', extra_dims=1)
        action_var = self.env.action_space.new_tensor_variable('action', extra_dims=1)
        advantage_var = tensor_utils.new_tensor('advantage', ndim=1, dtype=tf.float32)

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s' % k)
            for k, shape in self.policy.distribution.dist_info_specs
        }

        old_dist_info_vars_list = [old_dist_info_vars[k] for k in self.policy.distribution.dist_info_keys]

        central_obs_vars = [elem[1] for elem in self.central_policy_dist_infos]

        input_list = [obs_var, action_var, advantage_var] + old_dist_info_vars_list + all_obs_vars + central_obs_vars

        ###############################
        #
        # Local Policy Optimization
        #
        ###############################

        self.optimizers = []
        self.metrics = []

        for n, policy in enumerate(self.local_policies):

            state_info_vars = dict()
            dist_info_vars = policy.dist_info_sym(obs_var, state_info_vars)
            dist = policy.distribution

            kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
            lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
            surr_loss = - tf.reduce_mean(lr * advantage_var)

            if self.constrain_together:
                additional_loss = Metrics.kl_on_others(n, dist, all_task_dist_info_vars)

            elif self.constrain_against_central:
                additional_loss = Metrics.kl_on_central(dist, dist_info_vars, self.central_policy_dist_infos[n][0])
            
            else:
                additional_loss = tf.constant(0.0)

            local_loss = surr_loss + self.penalty * additional_loss

            kl_metric = tensor_utils.compile_function(inputs=input_list, outputs=additional_loss, log_name="KLPenalty%d" % n)
            self.metrics.append(kl_metric)

            mean_kl_constraint = tf.reduce_mean(kl)

            optimizer = PenaltyLbfgsOptimizer(name='expertOptimizer_'+str(n))
            optimizer.update_opt(
                loss=local_loss,
                target=policy,
                leq_constraint=(mean_kl_constraint, self.step_size),
                inputs=input_list,
                constraint_name="mean_kl_%d" % n,
            )
            self.optimizers.append(optimizer)

       
        return dict()


    def optimize_expert_policies(self, itr, all_samples_data):

        dist_info_keys = self.policy.distribution.dist_info_keys
        for n, optimizer in enumerate(self.optimizers):

            obs_act_adv_values = tuple(ext.extract(all_samples_data[n], "observations", "actions", "advantages"))
            dist_info_list = tuple([all_samples_data[n]["agent_infos"][k] for k in dist_info_keys])
            all_task_obs_values = tuple([samples_data["observations"] for samples_data in all_samples_data])

            all_input_values = obs_act_adv_values + dist_info_list + all_task_obs_values + all_task_obs_values
            optimizer.optimize(all_input_values)

            kl_penalty = sliced_fun(self.metrics[n], 1)(all_input_values)
            #logger.record_tabular('KLPenalty%d' % n, kl_penalty)


    @overrides
    def optimize_policy(self, itr, all_samples_data):
        assert len(all_samples_data) >= self.num_grad_updates + 1  # we collected the rollouts to compute the grads and then the test!
        assert self.use_maml

        input_vals_list = []

        # Code to account for off-policy sampling when more than 1 beta steps
        theta0_dist_info_list = []
        for i in range(self.meta_batch_size):
            if 'agent_infos_orig' not in all_samples_data[0][i].keys():
                assert False, "agent_infos_orig is missing--this should have been handled in batch_maml_polopt"
            else:
                agent_infos_orig = all_samples_data[0][i]['agent_infos_orig']
            theta0_dist_info_list += [agent_infos_orig[k] for k in self.policy.distribution.dist_info_keys]
        input_vals_list += tuple(theta0_dist_info_list)

        theta_l_dist_info_list = []
        for i in range(self.meta_batch_size):
            agent_infos = all_samples_data[0][i]['agent_infos']
            theta_l_dist_info_list += [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        input_vals_list += tuple(theta_l_dist_info_list)

        for step in range(self.num_grad_updates):
            obs_list, action_list, adv_list, rewards_list, returns_list, path_lengths_list, expert_action_list = [], [], [], [], [], [], []
            for i in range(self.meta_batch_size):  # for each task
                if not self.metalearn_baseline:
                    inputs = ext.extract(
                        all_samples_data[step][i],
                        "observations", "actions", "advantages", "expert_actions",
                    )
                    obs_list.append(inputs[0])
                    action_list.append(inputs[1])
                    adv_list.append(inputs[2])
                    expert_action_list.append(inputs[3])
                else:
                    inputs = ext.extract(
                        all_samples_data[step][i],
                        "observations", "actions", "rewards", "returns", "expert_actions", "paths"
                    )
                    obs_list.append(inputs[0])
                    action_list.append(inputs[1])
                    rewards_list.append(inputs[2])
                    returns_list.append(inputs[3])
                    expert_action_list.append(inputs[4])
                    # path_lengths_list.append([len(p['rewards']) for p in inputs[5]])
            if not self.metalearn_baseline:
                input_vals_list += obs_list + action_list + adv_list + expert_action_list
            else:
                input_vals_list += obs_list + action_list + rewards_list + returns_list + expert_action_list  #+ path_lengths_list before expert action list


        for step in [self.num_grad_updates]:  # last step
            obs_list, action_list, expert_action_list = [], [], []  # last step's adv_list not currently used in maml_il
            for i in range(self.meta_batch_size):  # for each task
                inputs = ext.extract(
                    all_samples_data[step][i],
                    "observations", "actions", "expert_actions",
                )
                obs_list.append(inputs[0])
                action_list.append(inputs[1])
                expert_action_list.append(inputs[2])

            input_vals_list += obs_list + action_list + expert_action_list

        # Code to compute the kl distance, kind of pointless on non-testing iterations as agent_infos are zeroed out on expert traj samples
        dist_info_list = []
        for i in range(self.meta_batch_size):
            # agent_infos = {x:all_samples_data[self.kl_constrain_step][i]['agent_infos'][x] for x in ['mean','log_std']}  ##kl_constrain_step default is -1, meaning post all alpha grad updates
            agent_infos = all_samples_data[self.kl_constrain_step][i]['agent_infos']  ##kl_constrain_step default is -1, meaning post all alpha grad updates
            dist_info_list += [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        input_vals_list += tuple(dist_info_list)  # This populates old_dist_info_vars_list

      #  logger.log("Computing KL before")
      #  mean_kl_before = self.optimizer.constraint_val(input_vals_list)  # TODO: need to make sure the input list has the correct form. Maybe start naming the input lists based on what they're needed for

        logger.log("Computing loss before")
       # loss_before = self.optimizer.loss(input_vals_list)
        if itr not in TESTING_ITRS:
            steps = self.adam_curve[min(itr,len(self.adam_curve)-1)]
            logger.log("Optimizing using %s Adam steps on itr %s" % (steps, itr))
            start_loss = self.optimizer.optimize(input_vals_list, steps=steps)
            # self.optimizer.optimize(input_vals_list)
            return start_loss

        else:
            logger.log("Not Optimizing")
            #logger.record_tabular("ILLoss",float('nan'))
            return None
    
    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        debug_params = self.policy.get_params_internal()

        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )



class Metrics:
    @staticmethod
    def symmetric_kl(dist, info_vars_1, info_vars_2):
     
        side1 = tf.reduce_mean(dist.kl_sym(info_vars_2, info_vars_1))
        side2 = tf.reduce_mean(dist.kl_sym(info_vars_1, info_vars_2))
        return (side1 + side2) / 2

    @staticmethod
    def kl_on_others(n, dist, dist_info_vars):
        # \sum_{j=1} E_{\sim S_j}[D_{kl}(\pi_j || \pi_i)]
        if len(dist_info_vars) < 2:
            return 0

        kl_with_others = 0
        for i in range(len(dist_info_vars)):
            if i != n:
                kl_with_others += Metrics.symmetric_kl(dist, dist_info_vars[i][i], dist_info_vars[i][n])

        return kl_with_others / (len(dist_info_vars) - 1)

    @staticmethod
    def kl_on_central(dist, distInfoVarsExpert, distInfoVarsCentral):
        # \sum_{j=1} E_{\sim S_j}[D_{kl}(\pi_j || \pi_i)]
       
        return Metrics.symmetric_kl(dist, distInfoVarsExpert , distInfoVarsCentral)




