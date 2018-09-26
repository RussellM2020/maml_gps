import pickle
import numpy as np
parallelData = pickle.load(open('parallelPaths.pkl', 'rb'))
vecData = pickle.load(open('vecPaths.pkl', 'rb'))



for _id in range(40):
    for batch in range(2):

            print('#############################')
        print(np.sum(vecData[_id][batch]['rewards']))
        print(np.sum(parallelData[_id][batch]['rewards']))

