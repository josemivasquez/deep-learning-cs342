# Location is [x, z, y]
# 2D trajectory
import numpy as np
norm = np.linalg.norm
# import numpy.linalg.solve as solve

def proj_op(u, v):
    # Proy of u in v
    return (np.dot(u, v) / norm(v)**2) * v

class Trajectory:
    def __init__(self, start, end):
        start = np.array(start)
        end = np.array(end)
        
        self.start = start
        self.end = end

        self.director = (end - start) / norm(end - start)
        self.L = 5
    
    def next(self, pos):
        proj = self.get_proj(pos)
        return self.move(proj)

    def get_proj(self, x):
        return self.start + proj_op(x - self.start, self.end - self.start)

    def move(self, point):
        return point + self.L * self.director

# def unit_vector(v):
#     return v / norm(v)
# def orthogonal(v):
#     return np.array([-v[1], v[0]])

# class CoorSystem:
#     def __init__(self, origin, i, j):
#         self.origin = origin
#         self.i = i
#         self.j = j
#     def convert(self, x):
#         return solve(
#             np.transpose(np.array([self.i, self.j])),
#             x - self.origin
#         )
#     def back(self, x):
#         return self.origin + self.i * x[0] + self.j * x[i]

# def distance_eq_gen(a, b, m):
#     return lambda x: 4*(m**2)*(x**3) - 4*b*m*x + 2*x - 2*a

# class Trajectory:
#     def __init__(self, start, end, k=2):
#         self.start = start
#         self.end = end
#         self.k = k
#         self.L = 4

#         i = unit_vector(end - start)
#         j = orthogonal(i)
#         if k < 0:
#             j *= -1
#         origin = (start + end) / 2
#         origin -= j * k

#         self.system = CoorSystem(origin, i, j)
#         self.m = k / ((norm(end - start) / 2) ** 2)
    
#     def next(self, pos) -> Tuple[float]:
#         pos = self.system.convert(pos)
#         proj = self.get_projection(pos)
#         response = self.move(proj, L)
#         response = self.system.back(response)

#         return response
    
#     def get_projection(self, pos) -> Tuple[float]:
#         if pos[1] > 0:
#             a = np.sign(pos[0]) * (pos[1] ** (1/2))
#         else:
#             a = 0

#         b = pos[0]
#         eq = distance_eq_gen(a, b, self.m)
#         x_proj = bisection(eq, a, b, 10**-3)
#         proj = np.array(x, x**2)
#         return proj

#     def move(self, x, L):
#         pass


# def bisection(eq, a, b, tol):
#     if eq(a) > 0:
#         a, b = b, a

#     assert eq(a) < 0
#     assert eq(b) > 0
    
#     while True:
#         med = (a + b) / 2
#         ev = eq(med)
#         error = abs(ev)
#         if error < tol:
#             break

#         if ev < 0:
#             a = med
#         else:
#             b = med

#     return med