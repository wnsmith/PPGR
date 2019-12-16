import numpy as np
import math

def normalize(p, tolerance=0.00001):
    mag2 = sum(n * n for n in p)
    if abs(mag2 - 1.0) > tolerance:
        mag = sqrt(mag2)
        p = tuple(n / mag for n in p)
    return p

def axisAngle2q(p, fi):
    p = normalize(p)
    x, y, z = p
    fi /= 2
    x = x * math.sin(fi)
    y = y * math.sin(fi)
    z = z * math.sin(fi)
    w = math.cos(fi)
    return [x, y, z, w]


p = [0.33333334, -0.66666667, 0.66666667]

fi = 1.5707963315726743

q = axisAngle2q(p, fi)
print(q)











