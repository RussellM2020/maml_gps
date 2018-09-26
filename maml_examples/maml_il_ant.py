
from sandbox.rocky.tf.algos.maml_il import MAMLIL
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.maml_gaussian_mlp_baseline import MAMLGaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.optimizers.quad_dist_expert_optimizer import QuadDistExpertOptimizer
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
#from rllab.envs.mujoco.ant_env_dense import  AntEnvRandGoalRing
from rllab.envs.mujoco.ant_env_sparse import  AntEnvRandGoalRing
from sandbox.rocky.tf.envs.base import TfEnv
# import lasagne.nonlinearities as NL
import sandbox.rocky.tf.core.layers as L

from rllab.envs.gym_env import GymEnv
#from maml_examples.reacher_env import ReacherEnv
#from rllab.envs.mujoco.pusher_env import PusherEnv


from maml_examples.maml_experiment_vars import MOD_FUNC
import numpy as np
import random as rd

#from examples.trpo_push_obj import
EXPERT_TRAJ_LOCATION_DICT = '/root/code/rllab/saved_expert_traj/Expert_trajs_dense_ant/'
#EXPERT_TRAJ_LOCATION_DICT = '/home/russellm/iclr18/maml_gps/saved_expert_traj/Expert_trajs_dense_ant/'

import tensorflow as tf
import time


beta_steps = 1
adam_steps_list = [50]
updateMode = 'vec'
adam_curve = None
fast_learning_rates = [1.0]

env_option = ''
# mode = "ec2"
mode = 'ec2'
extra_input = "onehot_exploration" # "onehot_exploration" "gaussian_exploration"
# extra_input = None
extra_input_dim = 5

fast_batch_size_list = [20]  # 20 # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]  #inner grad update size
meta_batch_size_list = [40]  # 40 @ 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
max_path_length = 200  # 100
num_grad_updates = 1
meta_step_size = 0.01
pre_std_modifier = 1.0
post_std_modifier_train = 0.00001
post_std_modifier_test = 0.00001
l2loss_std_mult = 1.0
ism = ''
#importance_sampling_modifier_list = ['']  #'', 'clip0.5_'
limit_demos_num = 40  # 40
test_goals_mult = 1
bas_lr = 0.01 # baseline learning rate
momentum=0.5
bas_hnl = tf.nn.relu
hidden_layers = (100,100)

basas = 60 # baseline adam steps
use_corr_term = True
# seeds = [1,2,3,4,5,6,7]  #,2,3,4,5,6,7,8] #, 2,3,4,5,6,7,8]
seeds = [1]  #,2,3,4,5,6,7,8] #, 2,3,4,5,6,7,8]
use_maml = True
test_on_training_goals = False

for seed in seeds:            
  for fast_batch_size in fast_batch_size_list:
    for meta_batch_size in meta_batch_size_list:    
      for fast_learning_rate in fast_learning_rates:
        for adam_steps in adam_steps_list:
                                                    
          stub(globals())
          tf.set_random_seed(seed)
          np.random.seed(seed)
          rd.seed(seed)          
          env = TfEnv(normalize(AntEnvRandGoalRing()))    
          policy = MAMLGaussianMLPPolicy(
              name="policy",
              env_spec=env.spec,
              grad_step_size=fast_learning_rate,
              hidden_nonlinearity=tf.nn.relu,
              hidden_sizes=(100, 100),
              std_modifier=pre_std_modifier,
              # metalearn_baseline=(bas == "MAMLGaussianMLP"),
              extra_input_dim=(0 if extra_input is None else extra_input_dim),
              updateMode = updateMode,
              num_tasks = meta_batch_size
          )
         
          
          baseline = LinearFeatureBaseline(env_spec=env.spec)
         
          algo = MAMLIL(
              env=env,
              policy=policy,
              #policy=None,
              #oad_policy='/home/alvin/maml_rl/data/local/R7-IL-0918/R7_IL_200_40_1_1_dem40_ei5_as50_basl_1809_04_27/itr_24.pkl',
              baseline=baseline,
              batch_size=fast_batch_size,  # number of trajs for alpha grad update
              max_path_length=max_path_length,
              meta_batch_size=meta_batch_size,  # number of tasks sampled for beta grad update
              num_grad_updates=num_grad_updates,  # number of alpha grad updates
              n_itr=200, #100
              make_video=False,
              use_maml=use_maml,
              use_pooled_goals=True,
              use_corr_term=use_corr_term,
              test_on_training_goals=test_on_training_goals,
              metalearn_baseline=False,
              # metalearn_baseline=False,
              limit_demos_num=limit_demos_num,
              test_goals_mult=test_goals_mult,
              step_size=meta_step_size,
              plot=False,
              beta_steps=beta_steps,
              adam_curve=adam_curve,
              adam_steps=adam_steps,
              pre_std_modifier=pre_std_modifier,
              l2loss_std_mult=l2loss_std_mult,
              importance_sampling_modifier=MOD_FUNC[ism],
              post_std_modifier_train=post_std_modifier_train,
              post_std_modifier_test=post_std_modifier_test,
              expert_trajs_dir=EXPERT_TRAJ_LOCATION_DICT,
              #[env_option+"."+mode+goals_suffix],
              expert_trajs_suffix="",
              seed=seed,
              extra_input=extra_input,
              extra_input_dim=(0 if extra_input is None else extra_input_dim),
              updateMode = updateMode
          )
          run_experiment_lite(
              algo.train(),
              n_parallel=10,
              snapshot_mode="all",
              python_command='python3',
              seed=seed,
              exp_name='sparse_parallelSampling_c48',
              exp_prefix='Maml_il_ant',
              plot=False,
              sync_s3_pkl=True,
              mode=mode,
              terminate_machine=True,
          )

