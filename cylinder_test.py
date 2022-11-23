
import open3d as o3d
import copy
import numpy as np
import math
from numba import njit, prange
from angle import *
from get_cylinder import create_cylinder_two_point


def make_point_cloud(npts, center, radius, colorize):
    pts = np.random.uniform(-radius, radius, size=[npts, 3]) + center
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if colorize:
        colors = np.random.uniform(0.0, 1.0, size=[npts, 3])
        cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud



def rotation_from_two_vectors(a, b):
    res = (b-a)
    res_norm = res/np.linalg.norm(res)
    alpha = np.arcsin(-res_norm[1])
    beta = np.arctan(-res_norm[0]/res_norm[2])
    gamma = 0

    return alpha, beta, gamma


def rotation_matrix_numpy(axis, theta):
    mat = np.eye(3, 3)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def PointsInCircum(punkt, r, n=100):
    return [(punkt[0]+np.cos(2*np.pi/n*x)*r, punkt[1]+np.sin(2*np.pi/n*x)*r, punkt[2]) for x in range(0, n)]



mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
    radius=0.01, height=4.0)
mesh_cylinder.compute_vertex_normals()
mesh_cylinder.paint_uniform_color([1, 0, 0])


p1_sq = np.array([4, 2, 1])
p2_sq = np.array([3, 3, 3])

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.3, origin=[0, 0, 0])
mesh_frame_p1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.3, origin=[0, 0, 0])
mesh_frame_p2 = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.3, origin=p2_sq)

vertices = np.asarray(mesh_cylinder.vertices)
vertex_normals = np.asarray(mesh_cylinder.vertex_normals)

print("We draw a few primitives using collection.")

origin_sq = np.array([0, 0, 0])
origin = np.expand_dims(origin_sq, axis=0)

p1 = np.expand_dims(p1_sq, axis=0)
p2 = np.expand_dims(p2_sq, axis=0)

pcd_1 = o3d.geometry.PointCloud()
pcd_1.points = o3d.utility.Vector3dVector(p1)

pcd_2 = o3d.geometry.PointCloud()
pcd_2.points = o3d.utility.Vector3dVector(p2)

alpha, beta, gamma = rotation_from_two_vectors(p1_sq, p2_sq)
angle = np.array([alpha, beta, gamma])

p3 = np.squeeze(p1-1*(p1-p2)/2)
p4 = np.squeeze(p2-p1)
p4_norm = p4/np.linalg.norm(p4)

origin_p1_p2 = np.vstack([origin, p1, p2, p3, p4, p4_norm])

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(origin_p1_p2),
    lines=o3d.utility.Vector2iVector([[1, 2], [0, 3], [0, 5]])
)

mesh_frame_p3 = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.3, origin=p1_sq)

z = np.asarray([0,0,1]).astype(np.float64)
# R = rotation_matrix_numpy(p3, angle_1)
R = rod_rot(z,p4_norm)
# cyl_rot = copy.deepcopy(mesh_cylinder).rotate(R).translate(p3)
cyl_rot = create_cylinder_two_point(p1_sq,p2_sq)
o3d.visualization.draw([line_set, pcd_1, pcd_2, mesh_frame_p3,
                       mesh_frame_p1, mesh_frame_p2, cyl_rot], show_ui=True)
