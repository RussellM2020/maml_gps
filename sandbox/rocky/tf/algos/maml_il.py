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
from maml_examples.maml_experiment_vars import TESTING_ITRS
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors

class MAMLIL(BatchMAMLPolopt):

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            use_maml=True,
            beta_steps=1,
            adam_steps=1,
            l2loss_std_mult=1.0,
            importance_sampling_modifier=tf.identity,
            metalearn_baseline=False,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict(min_penalty=1e-8)
            optimizer = QuadDistExpertOptimizer("name1", adam_steps=adam_steps)  #  **optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.use_maml = use_maml
        self.kl_constrain_step = -1
        self.l2loss_std_multiplier = l2loss_std_mult
        self.ism = importance_sampling_modifier
        self.metalearn_baseline = metalearn_baseline
        super(MAMLIL, self).__init__(optimizer=optimizer, beta_steps=beta_steps, use_maml_il=True, metalearn_baseline=metalearn_baseline, **kwargs)


    def make_vars(self, stepnum='0'):
        # lists over the meta_batch_size
        # We should only need the last stepnum for meta-optimization.
        obs_vars, action_vars, adv_vars, rewards_vars, returns_vars, path_lengths_vars, expert_action_vars = [], [], [], [], [], [], []
        for i in range(self.meta_batch_size):
            obs_vars.append(self.env.observation_space.new_tensor_variable(
                'obs' + stepnum + '_' + str(i),
                extra_dims=1,
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

        all_surr_objs, input_vars_list, inner_input_vars_list = [], [], []
        new_params = []
        old_logli_sym = []
        old_adv = []
        # old_action_vars = []
        # old_dist_info_sym = []
        input_vars_list += tuple(theta0_dist_info_vars_list) + tuple(theta_l_dist_info_vars_list)
        inner_input_vars_list += tuple(theta0_dist_info_vars_list) + tuple(theta_l_dist_info_vars_list)

        for grad_step in range(self.num_grad_updates):  # we are doing this for all but the last step
            if not self.metalearn_baseline:
                obs_vars, action_vars, adv_vars, expert_action_vars = self.make_vars(str(grad_step))
            else:
                obs_vars, action_vars, adv_vars, rewards_vars, returns_vars, expert_action_vars = self.make_vars(str(grad_step))  # path_lengths_vars before expert actions

            inner_surr_objs, inner_surr_objs_sym = [], []  # surrogate objectives

            new_params = []
            kls = []
            old_logli_sym.append([])
            old_adv.append([])
            # old_action_vars.append(action_vars)

            for i in range(self.meta_batch_size):  # for training task T_i
                adv = adv_vars[i]
                if self.metalearn_baseline:
                    predicted_returns_sym, _ = self.baseline.predict_sym(obs_vars=obs_vars[i], all_params=self.baseline.all_params)
                    predicted_returns_means_sym = tf.reshape(predicted_returns_sym['mean'], [-1])

                    predicted_returns_log_std_sym = tf.reshape(predicted_returns_sym['log_std'], [-1])
                    # print("debug63", predicted_returns_means_sym - returns_vars[i])
                    # print("debug62", predicted_returns_means_sym, returns_vars[i])
                    baseline_pred_loss_i = tf.nn.l2_loss(predicted_returns_means_sym -returns_vars[i]) - 0.0 * tf.reduce_sum(predicted_returns_log_std_sym)
                    if 'surr_obj' not in dir(self.baseline):
                        self.baseline.set_init_surr_obj(input_list=[obs_vars[0]] + [returns_vars[0]], surr_obj_tensor=baseline_pred_loss_i)
                    adv_sym = self.baseline.build_adv_sym(obs_vars=obs_vars[i],
                                                      rewards_vars=rewards_vars[i],
                                                      returns_vars=returns_vars[i],
                                                      baseline_pred_loss=baseline_pred_loss_i,
                                                      # path_lengths_vars=path_lengths_vars[i],
                                                      all_params=self.baseline.all_params)

                dist_info_sym_i, params = self.policy.dist_info_sym(obs_vars[i], state_info_vars, all_params=self.policy.all_params)
                if self.kl_constrain_step == 0:
                    kl = dist.kl_sym(old_dist_info_vars[i], dist_info_sym_i)
                    kls.append(kl)
                new_params.append(params)
                logli_i = dist.log_likelihood_sym(action_vars[i], dist_info_sym_i)
                lr = dist.likelihood_ratio_sym(action_vars[i], theta0_dist_info_vars[i], theta_l_dist_info_vars[i])
                old_logli_sym[-1].append(logli_i)
                old_adv[-1].append(adv)
                # lr1 = dist.likelihood_ratio_sym(action_vars[i], theta0_dist_info_vars[i], dist_info_vars_i)
                # lr = tf.clip_by_value(lr,0.5,2.0)
                lr = self.ism(lr)
                # formulate a minimization problem
                # The gradient of the surrogate objective is the policy gradient
                inner_surr_objs.append(-tf.reduce_mean(tf.multiply(tf.multiply(logli_i, lr,"debug2"), adv, "debug3")))
                if self.metalearn_baseline:
                    inner_surr_objs_sym.append(-tf.reduce_mean(tf.multiply(tf.multiply(logli_i, lr,"debug4"), adv_sym, "debug5")))
                # inner_surr_objs.append(-tf.reduce_mean(lr * adv_vars[i]))
            inner_input_vars_list += obs_vars + action_vars + adv_vars
            if not self.metalearn_baseline:
                input_vars_list += obs_vars + action_vars + adv_vars
            else:
                input_vars_list += obs_vars + action_vars + rewards_vars + returns_vars  # + path_lengths_vars
            # For computing the fast update for sampling
            # At this point, inner_input_vars_list is theta0 + theta_l + obs + action + adv
            self.policy.set_init_surr_obj(inner_input_vars_list, inner_surr_objs)

            input_vars_list += expert_action_vars # TODO: is this pre-update expert action vars? Should we kill this?
            if not self.metalearn_baseline:
                all_surr_objs.append(inner_surr_objs)
            else:
                all_surr_objs.append(inner_surr_objs_sym)

        # LAST INNER GRAD STEP
        if not self.metalearn_baseline:
            obs_vars, action_vars, _, expert_action_vars = self.make_vars('test')  # adv_vars was here instead of _
        else:
            obs_vars, action_vars, _, _, _, expert_action_vars = self.make_vars('test')
        outer_surr_objs = []
        corr_terms =[]
        for i in range(self.meta_batch_size):  # here we cycle through the last grad update but for validation tasks (i is the index of a task)
            # old_dist_info_sym_i, _ = self.policy.dist_info_sym(obs_vars[i], state_info_vars,all_params=self.policy.all_params)
            dist_info_sym_i, updated_params_i = self.policy.updated_dist_info_sym(task_id=i,surr_obj=all_surr_objs[-1][i],new_obs_var=obs_vars[i], params_dict=new_params[i])
            if self.kl_constrain_step == -1:  # if we only care about the kl of the last step, the last item in kls will be the overall
                kl = dist.kl_sym(old_dist_info_vars[i], dist_info_sym_i)
                kls.append(kl)  # we either get kl from here or from kl_constrain_step =0

            # here we define the loss for meta-gradient
            a_star = expert_action_vars[i]
            s = dist_info_sym_i["log_std"]
            m = dist_info_sym_i["mean"]
            print("debug32", m) # shape ?, 7
            outer_surr_obj = tf.reduce_mean(self.l2loss_std_multiplier*(tf.square(tf.exp(s)))+tf.square(m)-2*tf.multiply(m,a_star))
            # outer_surr_obj = tf.nn.l2_loss(m-e+0.0*s)
            outer_surr_objs.append(outer_surr_obj)
            # term0 = [tf.gradients(dist_info_vars_i["mean"][:,d], [new_params[i][key] for key in new_params[i].keys()]) for d in range(self.policy.action_dim)] # probably want to break this up into 7 gradients
            # term0 = tf.gradients(tf.nn.l2_loss(m-e), [new_params[i][key] for key in new_params[i].keys()])
            print("debug41", new_params[i])
            print("debug42", updated_params_i)
            # term0 = tf.gradients(tf.nn.l2_loss(m-e+0.0*s), [updated_params_i[key] for key in updated_params_i.keys()])
            term0 = tf.gradients(outer_surr_obj, [updated_params_i[key] for key in updated_params_i.keys()])
            # print("debug36", term0)
            # print("debug51", old_logli_sym[0][i])
            paths_range = range(int(self.batch_size/self.max_path_length/self.meta_batch_size))
            print("debug45", len(paths_range))

            temp1 =tf.reduce_sum(tf.reshape(old_logli_sym[0][i],[self.max_path_length,-1]),0)
            term1 = [tf.gradients(temp1[p],[self.policy.all_params[key] for key in self.policy.all_params.keys()]) for p in paths_range]

            temp2 =tf.reduce_mean(tf.reshape(old_logli_sym[0][i]*old_adv[0][i],[self.max_path_length,-1]),0)
            term2 = [tf.gradients(temp2[p],[self.policy.all_params[key] for key in self.policy.all_params.keys()]) for p in paths_range]

            # print("debug60", temp1)
            # print("debug61", temp2)

            # grad_wrt_theta = lambda x: tf.gradients(x, [self.policy.all_params[key] for key in self.policy.all_params.keys()])
            # temp1_1 = tf.map_fn(grad_wrt_theta, temp1)  # 20 x (17,100; 100; etc)
            # temp2_1 = tf.map_fn(grad_wrt_theta, temp1)  # 20 x (17,100; 100; etc)
            # flat_params =tf.concat([tf.reshape(self.policy.all_params[key],[-1]) for key in self.policy.all_params.keys()],0)
            # temp1_1 = [tf.gradients(temp1[j],flat_params) for j in range(int(self.batch_size/self.max_path_length/self.meta_batch_size))]
            # temp2_1 = [tf.gradients(temp2[j],flat_params) for j in range(int(self.batch_size/self.max_path_length/self.meta_batch_size))]

            # print("debug62", temp1_1[19])
            # print("debug63", temp2_1[19])

            # temp3 = tf.reduce_mean( [tf.matmul(tf.reshape(t1_1,[-1,1]),tf.reshape(t2_1,[1,-1])) for t1_1,t2_1 in zip(temp1_1,temp2_1)],0)

            # print("debug62", temp1_1)
            # print("debug63", temp2_1)
            # print("debug64", temp3)

            # term1 = tf.gradients(0.5*tf.reduce_mean(old_logli_sym[0][i]), [self.policy.all_params[key] for key in self.policy.all_params.keys()])
            # term2 = tf.gradients(inner_surr_objs[i], [self.policy.all_params[key] for key in self.policy.all_params.keys()])
            # term2 = tf.reduce_sum((m-a_star)*tf.convert_to_tensor([tf.reduce_sum([tf.reduce_sum(a*b) for a,b in zip(term0_d,term1)]) for term0_d in term0]))
            # term01 = tf.reduce_sum([tf.reduce_sum(a*b) for a,b in zip(term0,term1)])

            # corr_term_i = [self.policy.step_size*term01*t for t in term2]

            def grads_dotprod(A,B):
                # multiplies gradients because tf is too stupid
                return tf.reduce_sum([tf.reduce_sum(a*b) for a,b in zip(A,B)])

            def mult_grad_by_number(num, grad_list):
                return [num * grad for grad in grad_list]
            corr_term_i_v2_per_path_list = [mult_grad_by_number(self.policy.step_size*grads_dotprod(term0,term2[p]), term1[p]) for p in paths_range]

            # corr_term_i_v2 = tf.reduce_mean(tf.stack(corr_term_i_v2_per_path_list),0)
            corr_term_i_v2 = [tf.reduce_mean([c[y] for c in corr_term_i_v2_per_path_list],0) for y in range(len(corr_term_i_v2_per_path_list[0]))]

            # corr_terms.append(corr_term_i_v1)
            corr_terms.append(corr_term_i_v2)
            print("debug36", term0)
            # print("debug37", term1)
            # print("debug38", term01)

            print("debug52", corr_term_i_v2)


        outer_surr_obj = tf.reduce_mean(tf.stack(outer_surr_objs, 0))  # mean over all the different tasks
        corr_term = [tf.reduce_mean([c[y] for c in corr_terms],0) for y in range(len(corr_terms[0]))]
        print("debug39", corr_term)

        input_vars_list += obs_vars + action_vars + expert_action_vars + old_dist_info_vars_list  # +adv_vars # TODO: kill action_vars from this list, and if we're not doing kl, kill old_dist_info_vars_list too

        mean_kl = tf.reduce_mean(tf.concat(kls, 0))

        self.optimizer.update_opt(
            loss=outer_surr_obj,
            target=None,
            # target=(self.policy, self.baseline),
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_vars_list,
            constraint_name="mean_kl",
            correction_term=corr_term
        )

        return dict()


#######################################
    @overrides
    def optimize_policy(self, itr, all_samples_data):
        assert len(all_samples_data) == self.num_grad_updates + 1  # we collected the rollouts to compute the grads and then the test!
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
                input_vals_list += obs_list + action_list + rewards_list + returns_list  + expert_action_list  #+ path_lengths_list before expert action list


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
            agent_infos = all_samples_data[self.kl_constrain_step][i]['agent_infos']  ##kl_constrain_step default is -1, meaning post all alpha grad updates
            dist_info_list += [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        input_vals_list += tuple(dist_info_list)  # This populates old_dist_info_vars_list

      #  logger.log("Computing KL before")
      #  mean_kl_before = self.optimizer.constraint_val(input_vals_list)  # TODO: need to make sure the input list has the correct form. Maybe start naming the input lists based on what they're needed for

        logger.log("Computing loss before")
       # loss_before = self.optimizer.loss(input_vals_list)
        if itr not in TESTING_ITRS:
            logger.log("Optimizing")
            self.optimizer.optimize(input_vals_list)
        else:
            logger.log("Not Optimizing")
        logger.log("Computing loss after")
      #  loss_after = self.optimizer.loss(input_vals_list)
      #  logger.log("Computing KL after")
       # mean_kl = self.optimizer.constraint_val(input_vals_list)
       # logger.record_tabular('MeanKLBefore', mean_kl_before)
       # logger.record_tabular('MeanKL', mean_kl)
      #  logger.record_tabular('LossBefore', loss_before)
      #  logger.record_tabular('LossAfter', loss_after)
      #  logger.record_tabular('dLoss', loss_before - loss_after)
        # getting rid of the above because of issues with tabular
        return dict()


    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )






