import numpy as np
import math

E = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

p = [[0.33333334],
 	 [-0.66666667], 
 	 [0.66666667]]

pt = np.transpose(p)


px =  [[0, -p[2][0], p[1][0]],
	   [p[2][0], 0, -p[0][0]],
	   [-p[1][0], p[0][0], 0]]

fi = 1.5707963315726743

ppt = np.matmul(p, pt)

Rp = ppt + np.multiply(math.cos(fi), (np.subtract(E, ppt))) + np.multiply(math.sin(fi), px)

print(Rp)


























