import math
import numpy as np

# Proveravamo da li je matrica rotacije
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = np.float64)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 

def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
    R = np.array(R)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


R = [[ 0.11111111, -0.88888889, -0.44444444],
     [ 0.44444444, 0.44444444, -0.77777778],
     [ 0.88888889, -0.11111111, 0.44444444]]


result = rotationMatrixToEulerAngles(R)
print(result)








