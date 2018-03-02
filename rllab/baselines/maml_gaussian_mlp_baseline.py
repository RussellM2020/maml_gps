import numpy as np

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.regressors.maml_gaussian_mlp_regressor import MAMLGaussianMLPRegressor
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.optimizers.quad_dist_expert_optimizer import QuadDistExpertOptimizer
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian  # This is just a util class. No params.

from collections import OrderedDict
from sandbox.rocky.tf.misc import tensor_utils
from tensorflow.contrib.layers.python import layers as tf_layers
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors

from sandbox.rocky.tf.core.utils import make_input, make_dense_layer, forward_dense_layer, make_param_layer, \
    forward_param_layer

import tensorflow as tf

class MAMLGaussianMLPBaseline(Baseline, Parameterized, Serializable):

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            num_seq_inputs=1,
            learning_rate=0.01,
            algo_discount=0.99,
            hidden_sizes=(32,32),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.identity,
            init_std=1.0,

    ):
        Serializable.quick_init(self, locals())

        self.env_spec = env_spec
        obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim
        self.n_hidden = len(hidden_sizes)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.input_shape = (None, 2*obs_dim+3,)
        self.learning_rate = learning_rate
        self.algo_discount = algo_discount
        self.max_path_length = 100


        self.all_params = self.create_MLP(
            name="mean_baseline_network",
            output_dim=1,
            hidden_sizes=hidden_sizes,
        )
        self.input_tensor, _ = self.forward_MLP('mean_baseline_network', self.all_params, reuse=None)
        forward_mean = lambda x, params, is_train: self.forward_MLP('mean_baseline_network',all_params=params, input_tensor=x, is_training=is_train)[1]

        init_log_std = np.log(init_std)
        self.all_params['std_param'] = make_param_layer(
            num_units=1,
            param=tf.constant_initializer(init_log_std),
            name="output_bas_std_param",
            trainable=False,
        )
        forward_std = lambda x, params: forward_param_layer(x, params['std_param'])
        self.all_param_vals = None

        self.learning_rate_per_param = OrderedDict(zip(self.all_params.keys(),
                                                       [tf.Variable(self.learning_rate * tf.Variable(tf.ones_like(self.all_params[key]) if True else [[-15000.],[-19000.],[-20000.]]))
                                                                                         for key in self.all_params.keys()]))
        self.accumulation = OrderedDict(zip(self.all_params.keys(),[tf.Variable(tf.zeros_like(self.all_params[key])) for key in self.all_params.keys()]))
        self.momentum = 0.5  # 0.6 - 0.975

        self._forward = lambda enh_obs, params, is_train: (forward_mean(enh_obs, params, is_train), forward_std(enh_obs, params))

        self._dist = DiagonalGaussian(1)

        self._cached_params = {}

        super(MAMLGaussianMLPBaseline, self).__init__(env_spec)

        predict_sym = self.predict_sym(enh_obs_vars=self.input_tensor)
        mean_var = predict_sym['mean']
        log_std_var = predict_sym['log_std']

        self._init_f_dist = tensor_utils.compile_function(
            inputs=[self.input_tensor],
            outputs=[mean_var,log_std_var],
        )
        self._cur_f_dist = self._init_f_dist
        self.initialized = False
        # self.momopt = tf.train.MomentumOptimizer(learning_rate=0.000001, momentum=0.999)
        self.momopt = tf.train.AdamOptimizer(name="Adam2")


    @property
    def vectorized(self):
        return True


    def set_init_surr_obj(self, input_list, surr_obj_tensor):
        """ Set the surrogate objectives used the update the policy
        """
        self.input_list_for_grad = input_list
        self.surr_obj = surr_obj_tensor

    def fit_train_baseline(self, paths, repeat=100):
        if 'surr_obj' not in dir(self):
            assert False, "why didn't we define it already"
        param_keys = self.all_params.keys()

        sess = tf.get_default_session()
        obs = np.concatenate([np.clip(p["observations"],-10,10) for p in paths])
        obs2 = np.concatenate([np.square(np.clip(p["observations"],-10,10)) for p in paths])
        al = np.concatenate([np.arange(len(p["rewards"])).reshape(-1, 1)/100.0 for p in paths])
        al2 =al**2
        al3 = al**3
        # al0 = al**0
        returns = np.concatenate([p["returns"] for p in paths])
        # inputs = [np.concatenate([al,al2,al3],axis=1)] + [returns]
        inputs = [np.concatenate([obs,obs2,al,al2,al3],axis=1)] + [returns]

        if 'lr_train_step' not in dir(self) :
            gradients = dict(zip(param_keys, tf.gradients(self.surr_obj, [self.all_params[key] for key in param_keys])))  #+[self.learning_rate_per_param[key] for key in self.learning_rate_per_param.keys()])))
            postupdate_params = OrderedDict(zip(param_keys, [self.all_params[key] - self.learning_rate_per_param[key]*gradients[key] for key in param_keys]))
            print("debug88\n", self.all_params)
            print("debug89\n", postupdate_params)
            predicted_returns_sym, _ = self.predict_sym(enh_obs_vars = self.input_list_for_grad[0],all_params=postupdate_params)
            print("debug01\n", self.input_list_for_grad[0])
            print("debug02\n", self.input_list_for_grad[1])
            loss_after = tf.reduce_mean(tf.square(predicted_returns_sym['mean'] - tf.reshape(self.input_list_for_grad[1], [-1,1])) + 0.0 * predicted_returns_sym['log_std'])
            self.lr_train_step = self.momopt.minimize(loss=loss_after, var_list=[self.learning_rate_per_param[key] for key in self.learning_rate_per_param.keys()])
            # self.lr_train_step = self.momopt.minimize(loss=loss_after) #, var_list=[self.learning_rate_per_param[key] for key in self.learning_rate_per_param.keys()])
                                            # OrderedDict(zip(param_keys, [self.all_params[key] - self.learning_rate_per_param[key] * gradients[key] for key in param_keys]))
            # pull new param vals out of tensorflow, so gradient computation only done once
            # these are the updated values of the params after the gradient step

        uninit_vars = []
        for var in tf.global_variables():
            # note - this is hacky, may be better way to do this in newer TF.
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)
        sess.run(tf.variables_initializer(uninit_vars))
        feed_dict = dict(list(zip(self.input_list_for_grad, inputs)))

        for _ in range(repeat):
            if _ in [0,repeat-1]:
                print("debug99", sess.run(self.learning_rate_per_param).items())
            sess.run(self.lr_train_step,feed_dict=feed_dict)

            # self.all_param_vals, self.learning_rate_per_param_vals = sess.run(self.all_fast_params_tensor2,
            #                                 feed_dict=dict(list(zip(self.input_list_for_grad, inputs))))
            # self.assign_params(self.all_params, self.all_param_vals)
            # self.assign_lr(self.learning_rate_per_param, self.learning_rate_per_param_vals)
            #
        #
        # enh_obs = self.input_list_for_grad[0]
        # info, _ = self.predict_sym(enh_obs_vars=enh_obs, is_training=False)
        #
        # outputs = [info['mean'], info['log_std']]
        #
        # self._cur_f_dist = tensor_utils.compile_function(
        #     inputs=[self.input_tensor],
        #     outputs=outputs,
        # )





    @overrides
    def fit(self, paths, log=True, repeat=35):  # TODO REVERT repeat=10000
        # return True
        if 'surr_obj' not in dir(self):
            assert False, "why didn't we define it already"
        if not self.initialized:
            # self.learning_rate = 0.1 * self.learning_rate
            repeat = 1000
            self.lr_mult = 0.5
        """Equivalent of compute_updated_dists"""
        update_param_keys = self.all_params.keys()
        no_update_param_keys = []

        sess = tf.get_default_session()
        if 'init_params_tensor' not in dir(self):
            self.init_params_tensor = OrderedDict(zip(update_param_keys, [self.all_params[key] for key in update_param_keys]))
        self.init_param_vals = sess.run(self.init_params_tensor)
        self.init_accumulation_vals = sess.run(self.accumulation)
        obs = np.concatenate([np.clip(p["observations"],-10,10) for p in paths])
        obs2 = np.concatenate([np.square(np.clip(p["observations"],-10,10)) for p in paths])
        al = np.concatenate([np.arange(len(p["rewards"])).reshape(-1, 1)/100.0 for p in paths])
        al2 =al**2
        al3 = al**3
        # al0 = al**0
        # print("debug43", np.shape(obs))
        returns = np.concatenate([p["returns"] for p in paths])  #TODO: do we need to reshape the returns here?
        # print("debug11", np.shape(obs))
        inputs = [np.concatenate([obs,obs2,al,al2,al3],axis=1)] + [returns]
        # inputs = [np.concatenate([al,al2,al3],axis=1)] + [returns]
        #
        # if self.all_param_vals is not None:
        #     self.assign_params(self.all_params,self.all_param_vals)

        if 'all_fast_params_tensor' not in dir(self) or self.all_fast_params_tensor is None:
            gradients = dict(zip(update_param_keys, tf.gradients(self.surr_obj, [self.all_params[key] for key in update_param_keys])))
            new_accumulation = {key:self.momentum * self.accumulation[key] + gradients[key] for key in update_param_keys}
            fast_params_tensor = OrderedDict(zip(update_param_keys, [self.all_params[key] - self.lr_mult * self.learning_rate_per_param[key]*new_accumulation[key] for key in update_param_keys]))
            for k in no_update_param_keys:
                fast_params_tensor[k] = self.all_params[k]
            self.all_fast_params_tensor = (fast_params_tensor, new_accumulation)
            # pull new param vals out of tensorflow, so gradient computation only done once
            # these are the updated values of the params after the gradient step
        for _ in range(repeat):
            self.all_param_vals, self.accumulation_vals = sess.run(self.all_fast_params_tensor,
                                           feed_dict=dict(list(zip(self.input_list_for_grad, inputs))))
            self.assign_params(self.all_params, self.all_param_vals)
            self.assign_accumulation(self.accumulation, self.accumulation_vals)


        # if init_param_values is not None:
        #     self.assign_params(self.all_params, init_param_values)

        inputs = tf.split(self.input_tensor, 1, 0)  #TODO: how to convert this since we don't need to calculate multiple updates simultaneously
        enh_obs = inputs[0]
        info, _ = self.predict_sym(enh_obs_vars=enh_obs, all_params=self.all_param_vals,is_training=False)

        outputs = [info['mean'], info['log_std']]

        self._cur_f_dist = tensor_utils.compile_function(
            inputs=[self.input_tensor],
            outputs=outputs,
        )
        if not self.initialized:
            self.init_param_vals = sess.run(self.init_params_tensor)
            self.all_fast_params_tensor = None
            self.lr_mult = 1.0
            self.initialized=True

        self.assign_accumulation(self.accumulation, self.init_accumulation_vals)

    def get_variable_values(self, tensor_dict):
        sess = tf.get_default_session()
        result = sess.run(tensor_dict)
        return result

    def assign_params(self, tensor_dict, param_values):
        if 'assign_placeholders' not in dir(self):
            # make computation graph, if it doesn't exist; then cache it for future use.
            self.assign_placeholders = {}
            self.assign_ops = {}
            for key in tensor_dict.keys():
                self.assign_placeholders[key] = tf.placeholder(tf.float32)
                self.assign_ops[key] = tf.assign(tensor_dict[key], self.assign_placeholders[key])

        # print("debug78,", tensor_dict.keys())
        # print("debug79,", tensor_dict)
        # print("debug80,", param_values)

        feed_dict = {self.assign_placeholders[key]:param_values[key] for key in tensor_dict.keys()}
        sess = tf.get_default_session()
        sess.run(self.assign_ops, feed_dict)

    def assign_lr(self, tensor_dict, param_values):
        if 'assign_lr_placeholders' not in dir(self):
            # make computation graph, if it doesn't exist; then cache it for future use.
            self.assign_lr_placeholders = {}
            self.assign_lr_ops = {}
            for key in tensor_dict.keys():
                self.assign_lr_placeholders[key] = tf.placeholder(tf.float32)
                self.assign_lr_ops[key] = tf.assign(tensor_dict[key], self.assign_lr_placeholders[key])

        feed_dict = {self.assign_lr_placeholders[key]:param_values[key] for key in tensor_dict.keys()}
        sess = tf.get_default_session()
        sess.run(self.assign_lr_ops, feed_dict)

    def assign_accumulation(self, tensor_dict, param_values):
        if 'assign_acc_placeholders' not in dir(self):
            # make computation graph, if it doesn't exist; then cache it for future use.
            self.assign_acc_placeholders = {}
            self.assign_acc_ops = {}
            for key in tensor_dict.keys():
                self.assign_acc_placeholders[key] = tf.placeholder(tf.float32)
                self.assign_acc_ops[key] = tf.assign(tensor_dict[key], self.assign_acc_placeholders[key])

        feed_dict = {self.assign_acc_placeholders[key]:param_values[key] for key in tensor_dict.keys()}
        sess = tf.get_default_session()
        sess.run(self.assign_acc_ops, feed_dict)


    @overrides
    def predict(self, path):
        # flat_obs = self.env_spec.observation_space.flatten_n(path['observations'])
        obs = np.clip(path['observations'],-10,10)
        obs2 = np.square(obs)
        # al = np.zeros(shape=(len(path["rewards"]),1))
        al = np.arange(len(path["rewards"])).reshape(-1, 1)/100.0
        al2 = al**2
        al3 = al**3
        # al0 = al**0

        enh_obs = np.concatenate([obs, obs2, al, al2, al3],axis=1)
        # enh_obs = np.concatenate([al, al2, al3],axis=1)
        # print("debug24", enh_obs)
        # print("debug24.1", np.shape(enh_obs))
        result = self._cur_f_dist(enh_obs)
        if len(result) == 2:
            means, log_stds = result
        else:
            raise NotImplementedError('Not supported.')
        return np.reshape(means, [-1])

    def meta_predict(self, observations):
        # flat_obs = self.env_spec.observation_space.flatten_n(path['observations'])
        obs = np.zeros(shape=np.shape(observations))
        obs2 = obs
        # al = np.zeros(shape=(len(path["rewards"]),1))
        al = np.zeros(shape=(len(observations),1))
        al2 = al
        al3 = al
        # al0 = al

        enh_obs = np.concatenate([obs, obs2, al, al2, al3],axis=1)
        # enh_obs = np.concatenate([al, al2, al3],axis=1)
        # print("debug24", enh_obs)
        # print("debug24.1", np.shape(enh_obs))
        result = self._cur_f_dist(enh_obs)
        if len(result) == 2:
            means, log_stds = result
        else:
            raise NotImplementedError('Not supported.')
        return np.reshape(log_stds, [-1])


    @property
    def distribution(self):
        return self._dist

    def get_params_internal(self, all_params=False, **tags):
        if tags.get('trainable', False):
            params = tf.trainable_variables()
        else:
            params = tf.global_variables()

        params = [p for p in params if p.name.startswith('mean_baseline_network') or p.name.startswith('output_bas_std_param')]
        params = [p for p in params if 'Adam' not in p.name]

        return params


        # This makes all of the parameters.
    def create_MLP(self, name, output_dim, hidden_sizes,
                   hidden_W_init=tf_layers.xavier_initializer(), hidden_b_init=tf.zeros_initializer(),
                   output_W_init=tf_layers.xavier_initializer(), output_b_init=tf.zeros_initializer(),
                   weight_normalization=False,
                   ):
        all_params = OrderedDict()

        cur_shape = self.input_shape
        with tf.variable_scope(name):
            for idx, hidden_size in enumerate(hidden_sizes):
                W, b, cur_shape = make_dense_layer(
                    cur_shape,
                    num_units=hidden_size,
                    name="hidden_%d" % idx,
                    W=hidden_W_init,
                    b=hidden_b_init,
                    weight_norm=weight_normalization,
                )
                all_params['W' + str(idx)] = W
                all_params['b' + str(idx)] = b
            W, b, _ = make_dense_layer(
                cur_shape,
                num_units=output_dim,
                name='output',
                W=output_W_init,
                b=output_b_init,
                weight_norm=weight_normalization,
            )
            all_params['W' + str(len(hidden_sizes))] = W
            all_params['b' + str(len(hidden_sizes))] = b

        return all_params

    def forward_MLP(self, name, all_params, input_tensor=None,
                    batch_normalization=False, reuse=True, is_training=False):
        # is_training and reuse are for batch norm, irrelevant if batch_norm set to False
        # set reuse to False if the first time this func is called.
        with tf.variable_scope(name):
            if input_tensor is None:
                l_in = make_input(shape=self.input_shape, input_var=None, name='input')
            else:
                l_in = input_tensor

            l_hid = l_in

            for idx in range(self.n_hidden):
                l_hid = forward_dense_layer(l_hid, all_params['W' + str(idx)], all_params['b' + str(idx)],
                                            batch_norm=batch_normalization,
                                            nonlinearity=self.hidden_nonlinearity,
                                            scope=str(idx), reuse=reuse,
                                            is_training=is_training
                                            )
            output = forward_dense_layer(l_hid, all_params['W' + str(self.n_hidden)],
                                         all_params['b' + str(self.n_hidden)],
                                         batch_norm=False, nonlinearity=self.output_nonlinearity,
                                         )
            return l_in, output



    def get_params(self, all_params=False, **tags):
        """
        Get the list of parameters (symbolically), filtered by the provided tags.
        Some common tags include 'regularizable' and 'trainable'
        """
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_params:
            self._cached_params[tag_tuple] = self.get_params_internal(all_params, **tags)
        return self._cached_params[tag_tuple]

    def get_param_values(self, all_params=False, **tags):
        params = self.get_params(all_params, **tags)
        param_values = tf.get_default_session().run(params)
        return flatten_tensors(param_values)

    def switch_to_init_dist(self):
        # switch cur baseline distribution to pre-update baseline
        self._cur_f_dist = self._init_f_dist
        self.all_param_vals = None
        self.assign_params(self.all_params,self.init_param_vals)
        self.assign_accumulation(self.accumulation, self.init_accumulation_vals)
        sess = tf.get_default_session()
        print("debug, accum vals", sess.run(self.accumulation))

    def predict_sym(self, enh_obs_vars, all_params=None, is_training=True):
        """equivalent of dist_info_sym, this function constructs the tf graph, only called
        during beginning of meta-training"""
        return_params = True
        if all_params is None:
            return_params = False
            all_params = self.all_params
            if self.all_params is None:
                assert False, "Shouldn't get here"


        mean_var, std_param_var = self._forward(enh_obs=enh_obs_vars, params=all_params, is_train=is_training)

        if return_params:
            return dict(mean=mean_var, log_std=std_param_var), all_params
        else:
            return dict(mean=mean_var, log_std=std_param_var)

    def updated_predict_sym(self, baseline_pred_loss, enh_obs_vars, params_dict=None, accumulation_sym=None):
        """ symbolically create post-fitting baseline predict_sym, to be used for meta-optimization.
        Equivalent of updated_dist_info_sym"""
        old_params_dict = params_dict
        start_params_dict = old_params_dict
        if old_params_dict is None:
            old_params_dict = self.all_params
        param_keys = self.all_params.keys()

        update_param_keys = param_keys
        no_update_param_keys = []
        grads = tf.gradients(baseline_pred_loss, [old_params_dict[key] for key in update_param_keys])
        gradients = dict(zip(update_param_keys, grads))
        if accumulation_sym is not None:
            new_accumulation_sym = {key:self.momentum * accumulation_sym[key] + gradients[key] for key in update_param_keys}
            params_dict = dict(zip(update_param_keys, [old_params_dict[key] - self.learning_rate_per_param[key] * new_accumulation_sym[key] for key in update_param_keys]))
        else:
            new_accumulation_sym = None
            params_dict = dict(zip(update_param_keys, [old_params_dict[key] - self.learning_rate_per_param[key] * gradients[key] for key in update_param_keys]))
        # for key in update_param_keys:
        #     old_params_dict[key] = params_dict[key]
        for k in no_update_param_keys:
            params_dict[k] = old_params_dict[k]
        return (self.predict_sym(enh_obs_vars=enh_obs_vars, all_params=params_dict), new_accumulation_sym)

    def build_adv_sym(self,enh_obs_vars,rewards_vars, returns_vars, all_params, baseline_pred_loss=None, repeat=20):  # path_lengths_vars was before all_params
        # assert baseline_pred_loss is None, "don't give me baseline pred loss"
        updated_params = all_params
        predicted_returns_sym, _ = self.predict_sym(enh_obs_vars=enh_obs_vars, all_params=updated_params)
        returns_vars_ = tf.reshape(returns_vars, [-1,1])
        # accumulation_sym = {key:tf.Variable(self.accumulation[key]) for key in self.accumulation.keys()}
        accumulation_sym = self.accumulation
        for _ in range(repeat):
            baseline_pred_loss = tf.reduce_mean(tf.square(predicted_returns_sym['mean'] - returns_vars_) + 0.0 * predicted_returns_sym['log_std'])
            (predicted_returns_sym, updated_params), accumuluation_sym = self.updated_predict_sym(baseline_pred_loss=baseline_pred_loss, enh_obs_vars=enh_obs_vars, params_dict=updated_params, accumulation_sym=accumulation_sym)  # TODO: do we need to update the params here?

        organized_rewards = tf.reshape(rewards_vars, [-1,self.max_path_length])
        organized_pred_returns = tf.reshape(predicted_returns_sym['mean'], [-1,self.max_path_length])
        organized_pred_returns_ = tf.concat((organized_pred_returns[:,1:], tf.reshape(tf.zeros(tf.shape(organized_pred_returns[:,0])),[-1,1])),axis=1)

        deltas = organized_rewards + self.algo_discount * organized_pred_returns_ - organized_pred_returns
        adv_vars = tf.map_fn(lambda x: discount_cumsum_sym(x, self.algo_discount), deltas)

        adv_vars = tf.reshape(adv_vars, [-1])
        adv_vars = (adv_vars - tf.reduce_mean(adv_vars))/tf.sqrt(tf.reduce_mean(adv_vars**2))  # centering advantages
        adv_vars = adv_vars + predicted_returns_sym['log_std'][0]

        return adv_vars

    @overrides
    def set_param_values(self, flattened_params, **tags):
        raise NotImplementedError("todo")

        # @overrides
        # def fit(self, paths, log=True):  # aka compute updated baseline
        #     # self._preupdate_params = self._regressor.get_param_values()
        #
        #     param_keys = self.all_params.keys()
        #     update_param_keys = param_keys
        #     no_update_param_keys = []
        #     sess = tf.get_default_session()
        #
        #     observations = np.concatenate([p["observations"] for p in paths])
        #     returns = np.concatenate([p["returns"] for p in paths])
        #
        #     inputs = observations + returns
        #
        #
        #     learning_rate = self.learning_rate
        #     if self.all_param_vals is not None:
        #         self.assign_params(self.all_params, self.all_param_vals)
        #
        #     if "fit_tensor" not in dir(self):
        #         gradients = dict(zip(update_param_keys, tf.gradients(self._regressor.loss_sym, [self.all_params[key] for key in update_param_keys])))
        #         self.fit_tensor = OrderedDict(zip(update_param_keys,
        #                                              [self.all_params[key] - learning_rate * gradients[key] for key in
        #                                               update_param_keys]))
        #         for k in no_update_param_keys:
        #             self.fit_tensor[k] = self.all_params[k]
        #
        #     self.all_param_vals = sess.run(self.fit_tensor, feed_dict = dict(list(zip(self.input_list_for_grad, inputs))))
        #
        #
        #     inputs = self.input_tensor
        #     task_inp = inputs
        #     output = self.predict_sym(task_inp, dict(),all_params=self.all_param_vals, is_training=False)
        #
        #
        #     self._regressor._f_predict = tensor_utils.compile_function(inputs=[self.input_tensor], outputs=output)


        #
    # def revert(self):
    #     # assert self._preupdate_params is not None, "already reverted"
    #     if self._preupdate_params is None:
    #         return
    #     else:
    #         self._regressor.set_param_values(self._preupdate_params)
    #         self._preupdate_params = None

    # def compute_updated_baseline(self, samples):
    #     """ Compute fast gradients once per iteration and pull them out of tensorflow for sampling with the post-update policy.
    #     """
    #     num_tasks = len(samples)
    #     param_keys = self.all_params.keys()
    #     update_param_keys = param_keys
    #     no_update_param_keys = []
    #
    #     sess = tf.get_default_session()
    #
    #
    #
    #     for i in range(num_tasks):
    #
    #
    #     self._cur_f_dist = tensor_utils.compile_function



def discount_cumsum_sym(var, discount):
    # y[0] = x[0] + discount * x[1] + discount**2 * x[2] + ...
    # y[1] = x[1] + discount * x[2] + discount**2 * x[3] + ...
    discount = tf.cast(discount, tf.float32)
    range_ = tf.cast(tf.range(tf.size(var)), tf.float32)
    var_ = var * tf.pow(discount, range_)
    return tf.cumsum(var_,reverse=True) * tf.pow(discount,-range_)




