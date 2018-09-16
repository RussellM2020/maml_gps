
CHEETAH_GOALS_LOCATION = '/home/rosen/maml_rl/saved_goals/cheetah/goals_pool1.pkl'
CHEETAH_GOALS_LOCATION_EC2 = '/root/code/rllab/saved_goals/cheetah/goals_pool1.pkl'
CHEETAH_GOALS_LOC_INCREMENTAL = '/home/rosen/maml_rl/saved_expert_traj/cheetah_sparse/incremental_vels/goals_pool.pkl'
CHEETAH_GOALS_LOC_SPARSE = '/home/rosen/maml_rl/saved_expert_traj/cheetah_sparse/orig_-1_1/goals_pool.pkl'

ENV_OPTIONS = {
    '':'half_cheetah.xml',

}

default_cheetah_env_option = ''

EXPERT_TRAJ_LOCATION_DICT = {
    ".local": "/home/rosen/maml_rl/saved_expert_traj/cheetah/CH-ET4-individual/",
    ".local_sparse": "/home/rosen/maml_rl/saved_expert_traj/cheetah_sparse/-1_1/",
    ".local_sparse_1": "/home/rosen/maml_rl/saved_expert_traj/cheetah_sparse/orig_-1_1/",
    ".ec2_sparse": "/root/code/rllab/saved_expert_traj/cheetah_sparse/-1_1/",
    ".ec2_sparse_1": "/root/code/rllab/saved_expert_traj/cheetah_sparse/orig_-1_1/",
    ".local_sparse_dummy": "/home/rosen/maml_rl/saved_expert_traj/cheetah_sparse/dummy/",
    ".local_incremental": "/home/rosen/maml_rl/saved_expert_traj/cheetah_sparse/incremental_vels/",
    ".ec2": "/root/code/rllab/saved_expert_traj/cheetah/CH-ET4-individual/",
    ".local.noise0.1": "/home/rosen/maml_rl/saved_expert_traj/cheetah/CH-ET-E1.6-noise0.1/",
    ".ec2.noise0.1": "/root/code/rllab/saved_expert_traj/cheetah/CH-ET-E1.6-noise0.1/",
    ".local.noise0.1.small": "/home/rosen/maml_rl/saved_expert_traj/cheetah/CH-ET-E1.7-noise0.1-small/",
    ".ec2.noise0.1.small": "/root/code/rllab/saved_expert_traj/cheetah/CH-ET-E1.7-noise0.1-small/",
}

