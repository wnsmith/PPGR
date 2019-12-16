import numpy as np
import math

# proverimo da li je det A = 1
def is_orthogonal(A):
	det = np.linalg.det(A)
	# print(det)
	if det >= 1 - 0.01 and det <= 1:
		return 1
	return 0;



R = [[ 0.11111111, -0.88888889, -0.44444444],
     [ 0.44444444, 0.44444444, -0.77777778],
     [ 0.88888889, -0.11111111, 0.44444444]]

E = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]


res = np.matmul(R, np.transpose(R))
# print(res)

# print(is_orthogonal(R))
# jeste AAt = E i detA = 1


p = np.subtract(R, E)
p = np.array(p)
# print(p)

v1 = np.squeeze(np.array(p[:1][:3]))
# print(v1)
v2 = np.squeeze(np.array(p[1:2][:3]))
# print(v2)

x = np.cross(v1, v2)
# print(x)

norma_x = math.sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])
# print(norma_x)

# v oko koga rotiramo (sopstveni vektor za lamda = 1)
p = (1/norma_x) * x
# print(p)

u = v2
# print(u)

up = np.matmul(R, u)
# print(up)

# u * up
norm_u = math.sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2])
norm_up = math.sqrt(up[0]*up[0] + up[1]*up[1] + up[2]*up[2])

skalarni_proizvod = u.dot(up) / (norm_u * norm_up)

fi = math.acos(skalarni_proizvod)
# print(fi)

mesoviti = [[u[0], u[1], u[2]],
			[up[0], up[1], up[2]],
			[p[0], p[1], p[2]]]

det = np.linalg.det(mesoviti)
# print(det)

if det < 0:
	p = -p

print(p)
print(fi)
