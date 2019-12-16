import math
import numpy as np
from sklearn import preprocessing

def euler2A(t):
	Mx = np.array([[1, 0, 0], [0, math.cos(t[0]), -math.sin(t[0])], [0, math.sin(t[0]), math.cos(t[0])]])
	My = np.array([[math.cos(t[1]), 0, math.sin(t[1])], [0, 1, 0], [-math.sin(t[1]), 0, math.cos(t[1])]])
	Mz = np.array([[math.cos(t[2]), -math.sin(t[2]), 0], [math.sin(t[2]), math.cos(t[2]), 0], [0, 0, 1]])
	return np.dot(Mz, np.dot(My, Mx))


def normal(X):
	return math.sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2])

def normalise(M):
	M = np.array(M)
	norm = math.sqrt(sum([i*i for i in M]))
	return [i/norm for i in M]



def axisAngle(matrix):
	T = np.subtract(matrix, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
	r1 = np.squeeze(np.array(T[:1][:3]))
	r2 = np.squeeze(np.array(T[1:2][:3]))
	r3 = np.squeeze(np.array(T[2:3][:3]))
	Z = np.cross(r1, r2)
	Z_norm = normal(Z)


	T = (1/Z_norm)*Z
	K = r2
	UP = np.matmul(matrix, K)
	K_norm = normal(K)
	UP_norm = normal(UP)
	SP = K.dot(UP) / (K_norm * UP_norm)
	angle = math.acos(SP)
	m_t = [[K[0], K[1], K[2]], [UP[0], UP[1], UP[2]], [T[0], T[1], T[2]]]
	det = np.linalg.det(m_t)

	if det < 0:
		T = -T
	return [[x] for x in T], angle


def rodrigez(M, angle):
	M = normalise(M)
	M_transposed = np.transpose(M)
	M_x =  [[0, -M[2][0], M[1][0]],[M[2][0], 0, -M[0][0]],[-M[1][0], M[0][0], 0]]
	K = np.matmul(M, M_transposed)
	I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
	result = K + np.multiply(math.cos(angle), (np.subtract(I, K))) + np.multiply(math.sin(angle), M_x)
	return result


def A2Euler(M):
    if(rotation(M)):
        M = np.array(M)
        w = math.sqrt(M[0,0] * M[0,0] +  M[1,0] * M[1,0])
        inc = w < 0.000001
        if  not inc :
            x = math.atan2(M[2,1] , M[2,2])
            y = math.atan2(-M[2,0], w)
            z = math.atan2(M[1,0], M[0,0])
        else :
            x = math.atan2(-M[1,2], M[1,1])
            y = math.atan2(-M[2,0], w)
            z = 0

        return np.array([x, y, z])
    else:
        return ("---")

 

def rotation(M):
    M_transposed = np.transpose(M)
    r = np.dot(M_transposed, M)
    I = [[1,0,0],[0,1,0],[0,0,1]]
    x = np.linalg.norm(I - r)
    return x < 0.000001


def AxisAngle2Q(M, fi):
	M = normalise(M)
	x = M[0]
	y = M[1]
	z = M[2]
	fi = fi / 2
	x = x * math.sin(fi)
	y = y * math.sin(fi) 
	z = z * math.sin(fi) 
	return [x, y, z, math.cos(fi)]

def Q2AxisAngle(Q):
    fi = 2.0 * math.acos(Q[3])
    return np.array(normalise(Q[0:3])),fi

def main():

	angle = [-math.atan(1/4), -math.asin(8/9), math.atan(4)]

	#euler2A
	#print("euler2A: ")
	#A = euler2A(angle)
	#print(A)
	#print()

	#matrix = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]
	#matrix = np.array([[3/4, 1/4, math.sqrt(6)/4], [1/4, 3/4, -math.sqrt(6)/4], [-math.sqrt(6)/4, math.sqrt(6)/4, 2/4]])
	matrix = [[1/9, 8/9, -4/9], [-4/9, 4/9, 7/9], [8/9, 1/9, 4/9]]


	#axisAngle
	print("axisAngle: ")
	M, fi = axisAngle(matrix)
	print(axisAngle(matrix))
	print()

	#rodrigez
	print("rodrigez: ")
	R = rodrigez(M, fi)
	print(R)
	print()

	#A2Euler
	print("A2Euler: ")
	A_i = A2Euler(matrix)
	print(A_i)
	print()

	#AxisAngle2Q
	print("AxisAngle2Q: ")
	q = AxisAngle2Q(M, fi)
	print(q)
	print()

	#Q2AxisAngle
	print("Q2AxisAngle: ")
	print(Q2AxisAngle(q))


if __name__ == '__main__':
    main()
