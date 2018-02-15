import numpy as np
import scipy.optimize
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import logger
from sandbox.rocky.tf.misc import tensor_utils
from collections import OrderedDict




class QuadDistExpertOptimizer(Serializable):
    """
    Runs Tensorflow optimization on a quadratic loss function, under the kl constraint

    """

    def __init__(
            self,
            name,
            max_opt_itr=20,
            initial_penalty=1.0,
            min_penalty=1e-2,
            max_penalty=1e6,
            increase_penalty_factor=2,
            decrease_penalty_factor=0.5,
            max_penalty_itr=10,
            adapt_penalty=True,
            adam_steps=5,
    ):
        Serializable.quick_init(self, locals())
        self._name = name
        self._max_opt_itr = max_opt_itr
        self._penalty = initial_penalty
        self._initial_penalty = initial_penalty
        self._min_penalty = min_penalty
        self._max_penalty = max_penalty
        self._increase_penalty_factor = increase_penalty_factor
        self._decrease_penalty_factor = decrease_penalty_factor
        self._max_penalty_itr = max_penalty_itr
        self._adapt_penalty = adapt_penalty

        self._opt_fun = None
        self._target = None
        self._max_constraint_val = None
        self._constraint_name = None
        self._adam_steps = adam_steps
        self._correction_term = 0


    def update_opt(self, loss, target, leq_constraint, inputs, constraint_name="constraint", *args, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """
        constraint_term, constraint_value = leq_constraint
        with tf.variable_scope(self._name):
            penalty_var = tf.placeholder(tf.float32, tuple(), name="penalty")
        penalized_loss = loss + penalty_var * constraint_term

        self._target = target
        self._max_constraint_val = constraint_value
        self._constraint_name = constraint_name

        self._inputs = inputs
        self._loss = loss
        self._adam = tf.train.AdamOptimizer()
        # self._adam = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.5)
        # self._train_step = self._adam.minimize(self._loss)

        if "correction_term" in kwargs:
            self._correction_term = kwargs["correction_term"]
        else:
            self._correction_term = None

        # gradients = self._adam.compute_gradients(loss=self._loss, var_list=[self._target.all_params[key] for key in self._target.all_params.keys()])
        self._gradients = self._adam.compute_gradients(loss=self._loss)
        # self._dummy_gradients=tf.gradients(ys=self._loss,xs=[self._target.all_params[key] for key in self._target.all_params.keys()])
        if self._correction_term is None:
            self._train_step = self._adam.apply_gradients(self._gradients)
        else:
            print("debug1", self._gradients)
            print("debug2", self._correction_term)
            self.new_gradients = []
            for ((grad, var), corr) in zip(self._gradients, self._correction_term):
                self.new_gradients.append((grad + corr, var))
            print("debug3", self.new_gradients)
            self._train_step = self._adam.apply_gradients(self.new_gradients)

        # initialize Adam variables
        uninit_vars = []
        sess = tf.get_default_session()
        if sess is None:
            sess = tf.Session()
        for var in tf.global_variables():
            # note - this is hacky, may be better way to do this in newer TF.
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)
        sess.run(tf.variables_initializer(uninit_vars))




        def get_opt_output():
            params = target.get_params(trainable=True)
            grads = tf.gradients(penalized_loss, params)
            for idx, (grad, param) in enumerate(zip(grads, params)):
                if grad is None:
                    grads[idx] = tf.zeros_like(param)
            flat_grad = tensor_utils.flatten_tensor_variables(grads)
            return [
                tf.cast(penalized_loss, tf.float64),
                tf.cast(flat_grad, tf.float64),
            ]

        self._opt_fun = ext.lazydict(
            f_loss=lambda: tensor_utils.compile_function(inputs, loss, log_name="f_loss"),
            f_constraint=lambda: tensor_utils.compile_function(inputs, constraint_term, log_name="f_constraint"),
            f_penalized_loss=lambda: tensor_utils.compile_function(
                inputs=inputs + [penalty_var],
                outputs=[penalized_loss, loss, constraint_term],
                log_name="f_penalized_loss",
            ),
            f_opt=lambda: tensor_utils.compile_function(
                inputs=inputs + [penalty_var],
                outputs=get_opt_output(),
            )
        )

    def loss(self, inputs):
        return self._opt_fun["f_loss"](*inputs)

    def constraint_val(self, inputs):
        return self._opt_fun["f_constraint"](*inputs)

    def optimize(self, input_vals_list,):
        sess = tf.get_default_session()
        feed_dict = dict(list(zip(self._inputs, input_vals_list)))
        # print("debug01", sess.run(self._gradients, feed_dict=feed_dict))
        # numeric_grad = compute_numeric_grad(loss=self._loss, params=self._target.all_params, feed_dict=feed_dict)
        # print("debug02", numeric_grad)
        for _ in range(self._adam_steps):
            if _ in [0,1,100]:
                print("debug04 loss",sess.run(self._loss, feed_dict=feed_dict))

                # print("debug01", sess.run(self._gradients, feed_dict=feed_dict))
                # print("debug01.1", sess.run(self._dummy_gradients, feed_dict=feed_dict))
                # print("debug02", sess.run(self._correction_term, feed_dict=feed_dict))
                # print("debug03", sess.run(self.new_gradients, feed_dict=feed_dict))
            sess.run(self._train_step, feed_dict=feed_dict)


def compute_numeric_grad(loss, params, feed_dict, epsilon=1e-4):
    sess = tf.get_default_session()
    loss_theta = sess.run(loss, feed_dict=feed_dict)
    output = OrderedDict({})
    for key in params.keys():
        shape = sess.run(tf.shape(params[key]))
        output[key] = np.zeros(shape=shape,dtype=np.float32)
        assert len(shape) < 3, "not supported"
        if len(shape) == 1:
            for i in range(len(shape)):
                sess.run(tf.assign(params[key][i], params[key][i]+epsilon))
                loss_thetaeps = sess.run(loss, feed_dict=feed_dict)
                output[key][i] = (loss_thetaeps-loss_theta)/epsilon
                sess.run(tf.assign(params[key][i], params[key][i]-epsilon))
        if len(shape) == 2:
            a,b = shape
            # print("debug05",sess.run(params[key]))
            for i in range(a*b):
                j = i % a
                k = int((i-j)/a)
                sess.run(tf.assign(params[key], params[key]+eps_j_k(epsilon,a,b,j,k)))
                loss_thetaeps = sess.run(loss, feed_dict=feed_dict)
                output[key][j][k] = (loss_thetaeps-loss_theta)/epsilon
                if key == "W0" and j ==0 and k == 0:
                    print("debug1", key, j, k, output[key][j][k])
                    print(loss_theta,loss_thetaeps)
                sess.run(tf.assign(params[key], params[key]-eps_j_k(epsilon,a,b,j,k)))
    return output

def eps_j_k(epsilon,a,b,j,k):
    out = np.zeros(shape=(a,b),dtype=np.float32)
    out[j][k]=epsilon
    return out

def eps_i(epsilon,a,i):
    out = np.zeros(shape=(a,),dtype=np.float32)
    out[i] = epsilon
    return out