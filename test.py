import numpy as np
import scipy
image = np.array(
    [[2,1,2,3,1], 
     [3,9,1,1,4], 
     [4,5,0,7,0]])

soble_kernel = np.array([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]])

print(scipy.ndimage.convolve(image,soble_kernel,mode='constant'))

