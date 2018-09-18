from rllab.misc import logger
from rllab.sampler.base import BaseSampler
from rllab.sampler import parallel_sampler
from rllab.sampler.stateful_pool import singleton_pool
from rllab.misc.tensor_utils import flatten_tensors
import tensorflow as tf
import numpy as np
import itertools


def worker_init_tf(G):
    G.sess = tf.Session()
    G.sess.__enter__()


def worker_init_tf_vars(G):
    G.sess.run(tf.global_variables_initializer())


class BatchSampler(BaseSampler):
    def __init__(self, algo, n_envs=1):
        super(BatchSampler, self).__init__(algo)
        self.n_envs = n_envs

    def start_worker(self):
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf)
        parallel_sampler.populate_task(self.algo.env, self.algo.policy)
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf_vars)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr, reset_args=None,  return_dict=False, log_prefix='',extra_input=None,extra_input_dim=None,  save_img_obs=False, preupdate=True, numTrajs_perTask = None):
        if extra_input is not None:
            assert False, "not implemented"
        # if not preupdate:
        #     assert False, "not implemented"
        init_policy_params = cur_policy_params = self.algo.policy.get_param_values()
        if hasattr(self.algo.env,"get_param_values"):
            try:
                cur_env_params = self.algo.env.get_param_values()
            except:
                cur_env_params = None
        else:
            cur_env_params = None
        import time
        start = time.time()
        if type(reset_args) != list and type(reset_args)!=np.ndarray:
            reset_args = [reset_args]*self.n_envs

        # if not preupdate:
        #     assert  hasattr(self.algo.policy, 'all_param_vals')
        #     assert  self.algo.policy.all_param_vals is not None
        #     cur_policy_params = [flatten_tensors(x.values()) for x in self.algo.policy.all_param_vals]


            #all_param_vals have the updated values of the parameters, set in compute_updated_dist
            
      
        # if hasattr(self.algo.policy, 'all_param_vals'): #TODO: RK, need to make this less hacky and still work with non-maml policies
        #     if self.algo.policy.all_param_vals:
        #         cur_policy_params = [flatten_tensors(x.values()) for x in self.algo.policy.all_param_vals]
        #     else:
        #         cur_policy_params = [cur_policy_params]*self.n_envs
        # else:
        cur_policy_params = [cur_policy_params]*self.n_envs

        # do tasks sequentially and parallelize within rollouts per task.
        paths = {}
        #if preupdate:
        
    
        for i in range(self.n_envs):

            if preupdate:
                policy_params = cur_policy_params[i]
            else:
            
                policy_params = flatten_tensors(self.algo.policy.all_param_vals[i].values())  



            paths_i = parallel_sampler.sample_paths(
                policy_params=policy_params,
                
                env_params=cur_env_params,
                max_samples=self.algo.batch_size / self.n_envs,
                max_path_length=self.algo.max_path_length,
                scope=self.algo.scope,
                reset_arg=reset_args[i],
                taskIdx = i,
                show_prog_bar=False,
                preupdate = preupdate,
            )
            if numTrajs_perTask !=None:
                paths[i] = paths_i[:numTrajs_perTask]
            else:
                paths[i] = paths_i


        total_time = time.time() - start
        logger.record_tabular(log_prefix+"TotalExecTime", total_time)

        if not return_dict:
            flatten_list = lambda l: [item for sublist in l for item in sublist]
            paths = flatten_list(paths.values())

        #self.algo.policy.set_param_values(init_policy_params)

        # currently don't support not whole paths (if desired, add code to truncate paths)
        assert self.algo.whole_paths

        return paths
