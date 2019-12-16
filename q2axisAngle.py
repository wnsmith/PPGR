import numpy
import math

def normalize(p, tolerance=0.00001):
    mag2 = sum(n * n for n in p)
    if abs(mag2 - 1.0) > tolerance:
        mag = math.sqrt(mag2)
        p = tuple(n / mag for n in p)
    return p


def q2axisAngle(q):
    p, w = q[0:3], q[3]
    fi = 2.0 * math.acos(w)
    return normalize(p), fi


q = [0.23570226567262753, -0.4714045242741873, 0.4714045242741873, 0.707106779497348]

(p, fi) = q2axisAngle(q)

print(p)
print(fi)