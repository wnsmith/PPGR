import numpy as np
import matplotlib.pyplot as plt
import math


# A(1, 2, 1)
# B(-1, 0, 1)
# C(2.4, 1.3, 1)
# D(3, 3, 1)
# E(1, 1.5, 1)
# F(-2.3, 2.4, 1)


# Ap(-2, -1, 1)
# Bp(-1.4, 2, 1)
# Cp(-0.3, 1.3, 1)
# Dp(-1.6, 3, 1)
# Ep(0.3, 2, 1)
# Fp(-1.2, 1.4, 1)

#XYZ = np.array([[1, -1, 2.4, 3, 1, -2.3], [2, 0, 1.3, 3, 1.5, 2.4], [1, 1, 1, 1, 1, 1]])
#XYZ_prim = np.array([[-2, -1.4, -0.3, -1.6, 0.3, -1.2], [-1, 2, 1.3, 3, 2, 1.4], [1, 1, 1, 1, 1, 1]])


XYZ = np.array([[-3, -2, 1, -7, 2], [2, 5, 0, 3, 1], [1, 2, 3, 1, 2]])
XYZ_prim = np.array([[13, 30, 15, 17, 14], [5, 17, 7, 4, 9.02], [8, 11, 20, 11, 11]])

XYZsh = np.array([[1.1, -1, 2.4, 3, 1, -2.3], [2, 0, 1.3, 3.1, 1.5, 2.4], [1, 1, 1, 1, 1, 1]])
XYZ_primsh = np.array([[-2, -1.4, -0.3, -1.6, 0.3, -1.2], [-1, 2, 1.3, 3, 2, 1.4], [1, 1, 1, 1, 1, 1]])
#A(1.1, 2, 1)
#D(3, 3.1, 1)

new_coords = np.array([[0, 1, 2], [-1, 0, 3], [0, 0, 1]])

newXYZ = np.dot(new_coords, XYZ)
newXYZ_prim = np.dot(new_coords, XYZ_prim)



np.set_printoptions(suppress=True)

def naive(XYZ, XYZ_prim):

	a = np.array([[XYZ[0][0], XYZ[0][1], XYZ[0][2]], [XYZ[1][0], XYZ[1][1], XYZ[1][2]], [XYZ[2][0], XYZ[2][1], XYZ[2][2]]])
	b = np.array([XYZ[0][3], XYZ[1][3], XYZ[2][3]])

	ap = np.array([[XYZ_prim[0][0], XYZ_prim[0][1], XYZ_prim[0][2]], [XYZ_prim[1][0], XYZ_prim[1][1], XYZ_prim[1][2]], [XYZ_prim[2][0], XYZ_prim[2][1], XYZ_prim[2][2]]])
	bp = np.array([XYZ_prim[0][3], XYZ_prim[1][3], XYZ_prim[2][3]])


	coeffs1 = np.linalg.solve(a, b)
	coeffs2 = np.linalg.solve(ap, bp)

	P = np.zeros((3, 3))
	P_prim = np.zeros((3, 3))

	for i in range(3):
		P[:, i] = coeffs1[i]*XYZ[:, i]
		P_prim[:, i] = coeffs2[i]*XYZ_prim[:, i]

	S = np.linalg.inv(P)
	R = np.dot(P_prim, S)
	return R


def dlt(XYZ, XYZ_prim, n):
	A = coresp(XYZ, XYZ_prim, n)

	U, D, V = np.linalg.svd(A, full_matrices = True)

	V = [[V[8][0], V[8][1], V[8][2]], [V[8][3], V[8][4], V[8][5]], [V[8][6], V[8][7], V[8][8]]]
	F = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

	return np.dot(F, V)


def center_of_mass(XYZ, n):
	xs = 0
	ys = 0
	for i in range(n):
		xs += XYZ[0][i]/XYZ[2][i]
		ys += XYZ[1][i]/XYZ[2][i]

	return xs/n, ys/n

def euclidian(Ax, Ay, Bx, By):
	return math.sqrt((Ax-Bx)*(Ax-Bx) + (Ay-By)*(Ay-By))

def coresp(XYZ, XYZ_prim, n):
	A = np.zeros((2*n, 9))
	r = 0
	for i in range(n):
		matrix = np.zeros((2, 9))

		matrix[0, :] = np.array([0, 0, 0, -XYZ_prim[2][i]*XYZ[0][i], -XYZ_prim[2][i]*XYZ[1][i], -XYZ_prim[2][i]*XYZ[2][i], XYZ_prim[1, i]*XYZ[0][i], XYZ_prim[1, i]*XYZ[1][i], XYZ_prim[1, i]*XYZ[2][i]])
		matrix[1, :] = np.array([XYZ_prim[2][i]*XYZ[0][i], XYZ_prim[2][i]*XYZ[1][i], XYZ_prim[2][i]*XYZ[2][i], 0, 0, 0, -XYZ_prim[0][i]*XYZ[0][i], -XYZ_prim[0][i]*XYZ[1][i], -XYZ_prim[0][i]*XYZ[2][i]])

		A[r, :] = matrix[0, :]
		A[r+1, :] = matrix[1, :]
		r+=2

	return A



def draw_naive(XYZ, W, XYZ_prim, W_prim):
	plt.plot([XYZ[0][0],XYZ[0][1],XYZ[0][2], W[0]], [XYZ[1][0],XYZ[1][1],XYZ[1][2], W[1]], 'go-', label='line 1', linewidth=2)
	plt.plot([XYZ[0][0], W[0]], [XYZ[1][0], W[1]], "go-", label = "line 2", linewidth = "2") 


	plt.plot([XYZ_prim[0][0],XYZ_prim[0][1],XYZ_prim[0][2], W_prim[0]], [XYZ_prim[1][0],XYZ_prim[1][1],XYZ_prim[1][2], W_prim[1]], 'go-', label='line 1', linewidth=2, color = "blue")
	plt.plot([XYZ_prim[0][0], W_prim[0]], [XYZ_prim[1][0], W_prim[1]], "go-", label = "line 2", linewidth = "2", color = "blue") 

	plt.show()
	


def draw_dlt(XYZ, XYZ_prim):
	plt.plot([XYZ[0][0],XYZ[0][1],XYZ[0][2], XYZ[0][3]], [XYZ[1][0],XYZ[1][1],XYZ[1][2], XYZ[1][3]], 'go-', label='line 1', linewidth=2)
	plt.plot([XYZ[0][0], XYZ[0][3]], [XYZ[1][0], XYZ[1][3]], "go-", label = "line 2", linewidth = "2") 


	plt.plot([XYZ_prim[0][0],XYZ_prim[0][1],XYZ_prim[0][2], XYZ_prim[0][3]], [XYZ_prim[1][0],XYZ_prim[1][1],XYZ_prim[1][2], XYZ_prim[1][3]], 'go-', label='line 1', linewidth=2, color = "blue")
	plt.plot([XYZ_prim[0][0], XYZ_prim[0][3]], [XYZ_prim[1][0], XYZ_prim[1][3]], "go-", label = "line 2", linewidth = "2", color = "blue") 



	plt.show()




def normalisation(XYZ, n):

	Tx, Ty = center_of_mass(XYZ, n)
	
	g = np.array([[1, 0, -Tx], [0, 1, -Ty], [0, 0, 1]])

	total = 0
	for i in range(n):
		total += euclidian(XYZ[0][i]/XYZ[2][i], XYZ[1][i]/XYZ[2][i], Tx, Ty)


	s = np.array([[math.sqrt(2)/(total/n), 0, 0], [0, math.sqrt(2)/(total/n), 0], [0, 0, 1]])

	t = np.dot(s, g)
	normalised = np.dot(t, XYZ)

	return normalised, t


def normalisedDLT(XYZ, XYZ_prim, n):

	XYZn, T = normalisation(XYZ, n)
	XYZ_primn, Tp = normalisation(XYZ_prim, n)

	V = dlt(XYZn, XYZ_primn, n)
	R = np.linalg.multi_dot([np.linalg.inv(Tp), V, T])
	return R

def coeff(matrix1, matrix2):
	return matrix1[0][1]/matrix2[0][1]



print("naive")
R1 = naive(XYZ, XYZ_prim)
print(R1)
draw_dlt(XYZ, XYZ_prim)



print("\n__________________________________________________________\n")
print("DLT algorithm / (A, B, C, D)")
R2 = dlt(XYZ, XYZ_prim, 4)
print(R2)
#draw_dlt(XYZ, XYZ_prim)
print("\n./vs naive")
print(np.dot(R2, coeff(R1, R2)))



print("\n__________________________________________________________\n")
print("DLT algorithm / (A, B, C, D, E)")
R3 = dlt(XYZ, XYZ_prim, 5)
print(R3)
#draw_dlt(XYZ, XYZ_prim)
print("\n./vs naive")
print(np.dot(R3, coeff(R1, R3)))

print("\n__________________________________________________________\n")



'''
#A(1.1, 2, 1)
#D(3, 3.1, 1)
print("DLT algorithm / ~ A(1.1, 2) i D(3, 3.1)")
R4 = dlt(XYZsh, XYZ_primsh, 4)
print(R4)
#draw_dlt(XYZsh, XYZ_primsh)
print("\n")
print("\n./vs naive")
print(np.dot(R4, coeff(R1, R4)))
print("\n__________________________________________________________\n")
'''


print("NDLT algorithm / (A, B, C, D)")
R5 = normalisedDLT(XYZ, XYZ_prim, 4)
print(R5)
print("\n\n")
print("\n./vs naive")
print(np.dot(R5, coeff(R1, R5)))
print("\n__________________________________________________________\n")



print("NDLT algorithm / (A, B, C, D, E)")
R6 = normalisedDLT(XYZ, XYZ_prim, 5)
print(R6)
print("\n./vs naive")
print(np.dot(R6, coeff(R1, R6))) 
print("\n__________________________________________________________\n")


print("DLT algorithm / new coords")
dltNew = dlt(newXYZ, newXYZ_prim, 4)
print(dltNew)

print("\nDLT algorithm / old coords")
dltOld = np.linalg.multi_dot([np.linalg.inv(new_coords), dltNew, new_coords])
print(dltOld)

print("\nmatrix")
print(R2)

print("\nscaled matrix")
print(np.dot(dltOld, coeff(R1, dltOld)))
print("\n__________________________________________________________\n")



print("NDLT algorithm / new coords")
ndltNew = normalisedDLT(newXYZ, newXYZ_prim, 4)
print(ndltNew)

print("\nNDLT algorirthm / old coords")
ndltOld = np.linalg.multi_dot([np.linalg.inv(new_coords), dltNew, new_coords])
print(ndltOld)

print("\nmatrix")
print(R2)

print("\nscaled matrix")
print(np.dot(ndltOld, coeff(R1, ndltOld)))





print("\n__________________________________________________________\n")
