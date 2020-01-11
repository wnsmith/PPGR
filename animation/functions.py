import numpy as np
import math
import numpy.linalg as LA


def normalize(v):
    norm = LA.norm(v)
    if norm == 0:
        return v
    return v / norm
    

def slerp(q1, q2, tm, t):
    
    tmp = np.clip(t, 0, 1)
    
    tmp1 = np.array(q1)
    tmp1 = normalize(tmp1)
    tmp2 = np.array(q2)
    tmp2 = normalize(tmp2)
    cos0 = np.dot(tmp1, tmp2)
     
        
    if cos0 < 0:
        q1 = -q1
        cos0 = -cos0
    
    if cos0 > 0.95: 
        return normalize(np.array(q1) * (1-t) + np.array(q2) * t)
    
    phi0 = math.acos(cos0)
    
    qs = ((math.sin(phi0 * (1 - t/tm))) /
          math.sin(phi0)) * np.asarray(tmp1) + \
              ((math.sin(phi0 * (t/tm))) /
               math.sin(phi0)) * np.asarray(tmp2)
              
    return qs 
    
def Euler2A(fi, teta, psi):
    
    Rz = np.array([[math.cos(psi), -math.sin(psi), 0],
                   [math.sin(psi), math.cos(psi), 0],
                   [0, 0, 1]])

    Ry = np.array([[math.cos(teta), 0, math.sin(teta)],
                   [0, 1, 0],
                   [-math.sin(teta), 0, math.cos(teta)]])

    Rx = np.array([[1, 0, 0],
                   [0, math.cos(fi), -math.sin(fi)],
                   [0, math.sin(fi), math.cos(fi)]])

    return Rz.dot(Ry).dot(Rx)

def AxisAngle(A):
    lambdas, vector = np.linalg.eig(A, )
    for i in range(len(lambdas)):
        if round(lambdas[i], 6) == 1.0:
            p = np.real(vector[:, i])
            break

    p1 = p[0]
    p2 = p[1]
    p3 = p[2]
    tmp = np.array([p1, p2, p3])
    while True:
        if np.sum(p * tmp) == 1.0:
            tmp = tmp + np.array([0.1, 0, 0])
        else:
            u = np.cross(p, tmp)
            break

    u = np.cross(p, np.array([p1, p2, -p3]))
    u = u/math.sqrt(u[0]**2+u[1]**2+u[2]**2)

    up = A.dot(u)

    fi = round(math.acos(np.sum(u*up)), 5)
    if round(np.sum(np.cross(u, up)*p), 5) < 0:
        p = (-1)*p

    return [p, fi]



def Rodrigez(p, fi):
    
    if (p[0]**2 + p[1]**2 + p[2]**2) != 1:
        p = p/math.sqrt(p[0]**2+p[1]**2+p[2]**2)

    p1 = p[0]
    p2 = p[1]
    p3 = p[2]

    Px = np.array([[0, -p3, p2],
                   [p3, 0, -p1],
                   [-p2, p1, 0]])

    E = np.eye(3)
    p = np.reshape(p, (3, 1))
    Rp = p.dot(p.T) + math.cos(fi)*(E - p.dot(p.T)) + math.sin(fi)*Px
    return Rp

def A2Euler(A):
    fi, teta, psi = 0, 0, 0
    if A[2, 0] < 1:
        if A[2, 0] > -1:
            psi = math.atan2(A[1, 0], A[0, 0])
            teta = math.asin((-1)*A[2, 0])
            fi = math.atan2(A[2, 1], A[2, 2])
        else:
            psi = math.atan2((-1)*A[0, 1], A[1, 1])
            teta = math.pi/2.0
            fi = 0.0
    else:
        psi = math.atan2((-1)*A[0, 1], A[1, 1])
        teta = (-1.0)*math.pi/2.0
        fi = 0

    return([fi, teta, psi])

def AngleAxis2Q(p, fi):
    w = math.cos(fi/2.0)
    norm = np.linalg.norm(p)
    if norm != 0:
        p = p/norm
    [x, y, z] = math.sin(fi/2.0) * p
    return [x, y, z, w]


def Q2AxisAngle(q):
    norm = np.linalg.norm(q)
    if norm != 0:
        q = q/norm

    fi = 2*math.acos(q[3])
    if abs(q[3]) == 1:
        p = [1, 0, 0]
    else:
        norm = np.linalg.norm(np.array([q[0], q[1], q[2]]))
        p = np.array([q[0], q[1], q[2]])
        if norm != 0:
            p = p / norm

    return [p, fi]