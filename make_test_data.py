import numpy as np

input_features = np.array([
    [[1,2,0], [1,1,3], [0,2,2]], 

    [[0,2,1], [0,3,2], [1,1,0]],

    [[1,2,1], [0,1,3], [3,3,2]]

])

kernel = np.array([
   [
       [[1,1], [2,2]],
       [[1,1], [1,1]],
       [[0,1], [1,0]]
   ],


   [
       [[1, 0], [0,1]],
       [[2,1], [2,1]],
       [[1,2], [2,0]]
   ] 

])

weight_save = np.transpose(kernel, [0,2,3,1]).astype(np.float32)
input_save = np.transpose(input_features, [1,2,0] ).astype(np.float32)

np.save('./test_data/test_input.npy', input_save)
np.save('./test_data/test_weight.npy', weight_save)
np.save('./test_data/test_bias.npy', np.array([[0],[0]]).astype(np.float32))