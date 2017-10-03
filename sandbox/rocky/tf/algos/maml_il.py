
import tensorflow as tf
import rllab.misc.logger as logger
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.batch_maml_polopt import BatchMAMLPolopt
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.optimizers.quad_dist_expert_optimizer import QuadDistExpertOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


class MAMLIL(BatchMAMLPolopt):

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            use_maml=True,
            beta_steps=1,
            l2loss_std_mult=10.0,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict(min_penalty=1e-8)
            optimizer = QuadDistExpertOptimizer("name1", beta_steps=beta_steps)  #  **optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.use_maml = use_maml
        self.kl_constrain_step = -1
        self.l2loss_std_multiplier = l2loss_std_mult
        super(MAMLIL, self).__init__(optimizer=optimizer, **kwargs)


    def make_vars(self, stepnum='0'):
        # lists over the meta_batch_size
        # We should only need the last stepnum.
        obs_vars, action_vars, adv_vars, expert_action_vars = [], [], [], []
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
            expert_action_vars.append(self.env.action_space.new_tensor_variable(
                name='expert_actions' + stepnum + '_' + str(i),
                extra_dims=1,
            ))

        return obs_vars, action_vars, adv_vars, expert_action_vars

    @overrides
    def init_opt(self):
        assert not int(self.policy.recurrent)  # not supported
        assert self.use_maml  # only maml supported

        dist = self.policy.distribution

        old_dist_info_vars, old_dist_info_vars_list = [], []  # TODO: I think this should be in the make_vars function
        for i in range(self.meta_batch_size):
            old_dist_info_vars.append({
                k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s_%s' % (i, k))
                for k, shape in dist.dist_info_specs
                })
            old_dist_info_vars_list += [old_dist_info_vars[i][k] for k in dist.dist_info_keys]

        state_info_vars, state_info_vars_list = {}, []  # TODO: is this needed?

        all_surr_objs, input_vars_list = [], []  # TODO: we should probably use different variable names for the inside and outside objective
        new_params = []
        dist_info_vars_list = []
        for grad_step in [0]:  # range(self.num_grad_updates): we are only going to do this for grad_step=0
            obs_vars, action_vars, adv_vars, expert_action_vars = self.make_vars(str(grad_step))
            surr_objs = []  # surrogate objectives

            new_params = []
            kls = []

            for i in range(self.meta_batch_size):  # for training task T_i

                dist_info_vars_i, params = self.policy.dist_info_sym(obs_vars[i], state_info_vars, all_params=self.policy.all_params)
                if self.kl_constrain_step == 0:
                    kl = dist.kl_sym(old_dist_info_vars[i], dist_info_vars_i)
                    kls.append(kl)
                new_params.append(params)
                logli = dist.log_likelihood_sym(action_vars[i], dist_info_vars_i)
                # formulate a minimization problem
                # The gradient of the surrogate objective is the policy gradient
                surr_objs.append(-tf.reduce_mean(logli * adv_vars[i]))

            input_vars_list += obs_vars + action_vars + adv_vars + expert_action_vars
            # For computing the fast update for sampling
            self.policy.set_init_surr_obj(input_vars_list, surr_objs)

            all_surr_objs.append(surr_objs)

        # last inner grad step
        obs_vars, action_vars, adv_vars, expert_action_vars = self.make_vars('test')
        surr_objs = []
        for i in range(self.meta_batch_size):  # here we cycle through the last grad update but for validation tasks (i is the index of a task)
            dist_info_vars_i, _ = self.policy.updated_dist_info_sym(i, all_surr_objs[-1][i], obs_vars[i], params_dict=new_params[i])
            #print("debug2", dist_info_vars_i)
            if self.kl_constrain_step == -1:  # if we only care about the kl of the last step, the last item in kls will be the overall
                kl = dist.kl_sym(old_dist_info_vars[i], dist_info_vars_i)
                kls.append(kl)  # we either get kl from here or from kl_constrain_step =0

            # here we define the loss for meta-gradient
            e = expert_action_vars[i]
            s = dist_info_vars_i["log_std"]
            m = dist_info_vars_i["mean"]
            surr_objs.append(tf.reduce_mean(self.l2loss_std_multiplier*tf.exp(s)**2+m**2-2*m*e))
            #surr_objs.append(tf.reduce_mean((m-e)*(m-e)))
            #surr_objs.append(tf.nn.l2_loss(m-e))

        surr_obj = tf.reduce_mean(tf.stack(surr_objs, 0))  # mean over all the different tasks
        input_vars_list += obs_vars + action_vars + adv_vars + expert_action_vars + old_dist_info_vars_list   # TODO: do we need the input_list values over anything that's not the last grad step?

        mean_kl = tf.reduce_mean(tf.concat(kls, 0))

        self.optimizer.update_opt(
            loss=surr_obj,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_vars_list,
            constraint_name="mean_kl"
        )
        return dict()


#######################################
    @overrides
    def optimize_policy(self, itr, all_samples_data):
        assert len(all_samples_data) == self.num_grad_updates + 1  # we collected the rollouts to compute the grads and then the test!
        assert self.use_maml

        input_vals_list = []
        for step in range(len(all_samples_data)):
            obs_list, action_list, adv_list, expert_action_list = [], [], [], []
            for i in range(self.meta_batch_size):  # for each task
                inputs = ext.extract(
                    all_samples_data[step][i],
                    "observations", "actions", "advantages", "expert_actions",
                )
                obs_list.append(inputs[0])
                action_list.append(inputs[1])
                adv_list.append(inputs[2])
                expert_action_list.append(inputs[3])
                # if i == 0:  # diagnostic printout
                #     print("diagnostic1 step", step)
                #
                #     print(obs_list, "\n\n", action_list, '\n\n', expert_action_list, '\n\n')

            input_vals_list += obs_list + action_list + adv_list + expert_action_list # [ [obs_0], [act_0], [adv_0]. [act*_0], [obs_1], ... ]

        # Code to compute the kl distance
        dist_info_list = []
        for i in range(self.meta_batch_size):
            agent_infos = all_samples_data[self.kl_constrain_step][i]['agent_infos']  ##kl_constrain_step default is -1, meaning post all alpha grad updates
            dist_info_list += [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
            # if i == 0:
            #     print('diagnostic2 \n', dist_info_list)
            #     print('diagnostic3 \n', agent_infos["mean"])
            #     print('diag4 \n', agent_infos["log_std"])
        input_vals_list += tuple(dist_info_list)  # TODO: doesn't this populate old_dist_info_vars_list?
        logger.log("Computing KL before")
        mean_kl_before = self.optimizer.constraint_val(input_vals_list)  # TODO: need to make sure the input list has the correct form. Maybe start naming the input lists based on what they're needed for

        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(input_vals_list)
        if itr % 2 == 0:
            logger.log("Optimizing")
            self.optimizer.optimize(input_vals_list)
        else:
            logger.log("Not Optimizing")
        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(input_vals_list)
        logger.log("Computing KL after")
        mean_kl = self.optimizer.constraint_val(input_vals_list)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()


    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )





