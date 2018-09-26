import numpy as np

def dummy_expert_loader(path):
    return DummyExpert(path)

class DummyExpert():
    def __init__(self, path):
        goal_num = int(path.split('/')[-3])
        import joblib
        self.goal = joblib.load('/home/russellm/abhishek_sandbox/maml_imitation/saved_expert_traj/point/ETs_E1_randstart_test1/goals_pool.pkl')['goals_pool'][goal_num]
    
    def detstep(self, obs):
        return (self.goal - obs)

    def getLogProb(self, ob, action):
        return -np.linalg.norm(self.detstep(ob) - action, axis=1)**2 / np.linalg.norm(self.detstep(ob), axis=-1)
