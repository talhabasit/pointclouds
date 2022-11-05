
import open3d as o3d
import copy
import numpy as np
import math

mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01,
                                                          height=4.0)

def vector_angle(u, v):
    return np.arccos(np.dot(u,v) / (np.linalg.norm(u)* np.linalg.norm(v)))

def GetAngle(a, b):
    length = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)
    rise = b[1] - a[1]
    run = math.sqrt((length**2) - (rise**2))
    angle = math.degrees(math.atan(rise/run))
    return angle

mesh_cylinder.compute_vertex_normals()
mesh_cylinder.paint_uniform_color([1, 0, 0])
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0, 0, 0])



print("We draw a few primitives using collection.")
p1=np.expand_dims(np.array([1,1,1]),axis=0)
p2=np.expand_dims(np.array([2,2,2]),axis=0)
pcd_1= o3d.geometry.PointCloud()
pcd_1.points=o3d.utility.Vector3dVector(p1)
pcd_2= o3d.geometry.PointCloud()
pcd_2.points=o3d.utility.Vector3dVector(p2)

angle_1=np.deg2rad(GetAngle(np.squeeze(p1),np.squeeze(p2)))
angle = np.array([angle_1 ,-angle_1,0])
cc=np.vstack([p1,p2])
R = pcd_1.get_rotation_matrix_from_yxz(angle)

def make_point_cloud(npts, center, radius, colorize):
    pts = np.random.uniform(-radius, radius, size=[npts, 3]) + center
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if colorize:
        colors = np.random.uniform(0.0, 1.0, size=[npts, 3])
        cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud

sphere=make_point_cloud(1024,p1,2,True)

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(cc),
    lines=o3d.utility.Vector2iVector([[0,1]])
)

p3= p1-1*(p1-p2)/2
cyl_rot=copy.deepcopy(mesh_cylinder).translate(np.squeeze(p3)).paint_uniform_color([0.1, 0.9, 0.1]).rotate(R)

o3d.visualization.draw_geometries([line_set,pcd_1,pcd_2, mesh_frame,cyl_rot])



