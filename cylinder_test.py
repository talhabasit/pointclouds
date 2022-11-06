
import open3d as o3d
import copy
import numpy as np
import math
from numba import njit,prange



def make_point_cloud(npts, center, radius, colorize):
    pts = np.random.uniform(-radius, radius, size=[npts, 3]) + center
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if colorize:
        colors = np.random.uniform(0.0, 1.0, size=[npts, 3])
        cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud

def vector_angle(u, v):
    return np.arccos(np.dot(u,v) / (np.linalg.norm(u)* np.linalg.norm(v)))

def GetAngle(a, b):
    length = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)
    rise = b[1] - a[1]
    run = math.sqrt((length**2) - (rise**2))
    angle = math.degrees(math.atan(rise/run))
    return angle



def rotation_from_two_vectors(a,b):
    
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a,b)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = np.eye(3) + vx + np.dot(np.dot(vx,vx),((1-c)/(s**2)))
    
    return r



@njit()
def rotation_matrix_numpy(axis, theta):
    mat = np.eye(3,3)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01,
                                                          height=4.0)

mesh_cylinder.compute_vertex_normals()
mesh_cylinder.paint_uniform_color([1, 0, 0])
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.3, origin=[0, 0, 0])
mesh_frame_p1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.3, origin=[1, 2, 1])
mesh_frame_p2 = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.3, origin=[2, 2, 2])

vertices= np.asarray(mesh_cylinder.vertices)
vertex_normals= np.asarray(mesh_cylinder.vertex_normals)

print("We draw a few primitives using collection.")
p1_sq=np.array([1,2,1])
p2_sq=np.array([2,2,2])

origin_sq=np.array([0,0,0])
origin=np.expand_dims(origin_sq,axis=0)

p1=np.expand_dims(p1_sq,axis=0)
p2=np.expand_dims(p2_sq,axis=0)

pcd_1= o3d.geometry.PointCloud()
pcd_1.points=o3d.utility.Vector3dVector(p1)

pcd_2= o3d.geometry.PointCloud()
pcd_2.points=o3d.utility.Vector3dVector(p2)

angle_1=np.deg2rad(GetAngle(np.squeeze(p1),np.squeeze(p2)))
angle = np.array([-np.pi/3 ,angle_1,0])


p1_p2=np.vstack([p1,p2])
origin_p1_p2=np.vstack([origin,p1,p2])
                  
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(origin_p1_p2),
    lines=o3d.utility.Vector2iVector([[0,1],[0,2],[1,2]])
)


R= rotation_from_two_vectors(p1_sq,p2_sq)

p3= np.squeeze(p1-1*(p1-p2)/2)
# R= rotation_matrix_numpy(p3,1.8*(-np.pi/2))
cyl_rot=copy.deepcopy(mesh_cylinder).rotate(R).translate(p3)

o3d.visualization.draw([line_set,pcd_1,pcd_2,mesh_frame, mesh_frame_p1,mesh_frame_p2,cyl_rot],show_ui=True)



