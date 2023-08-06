
import open3d as o3d
import numpy as np


from angle import *
from get_cylinder import create_cylinder_two_point

"""This script can be used to test the cylinder creation and rotation functions"""


# Define the two points p1 and p2 in 3D space
p1_sq = np.array([4, 2, 1])
p2_sq = np.array([3, 3, 3])


# Convert the points to numpy arrays
p1 = np.expand_dims(p1_sq, axis=0)
p2 = np.expand_dims(p2_sq, axis=0)

# Create the point clouds
pcd_1 = o3d.geometry.PointCloud()
pcd_1.points = o3d.utility.Vector3dVector(p1)

pcd_2 = o3d.geometry.PointCloud()
pcd_2.points = o3d.utility.Vector3dVector(p2)

# Define the cylinder
p3 = np.squeeze(p1-1*(p1-p2)/2)
p4 = np.squeeze(p2-p1)
p4_norm = p4/np.linalg.norm(p4)

# Define the origin and the points for the lineset
origin_sq = np.array([0, 0, 0])
origin = np.expand_dims(origin_sq, axis=0)
origin_p1_p2 = np.vstack([origin, p1, p2, p3, p4, p4_norm])

# Create the lineset
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(origin_p1_p2),
    lines=o3d.utility.Vector2iVector([[1, 2],[0, 1], [0, 3], [0, 2]])
)

# Create a coordinate frame for the point p1
mesh_frame_p1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.3, origin=[0, 0, 0])

# Create a coordinate frame for the point p2
mesh_frame_p2 = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.3, origin=p2_sq)

# Create a coordinate frame for the point p3
mesh_frame_p3 = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.3, origin=p1_sq)

# Create a cylinder and a coordinate frame
mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
    radius=0.1, height=np.linalg.norm(p2_sq - p1_sq))
mesh_cylinder.compute_vertex_normals()
mesh_cylinder.paint_uniform_color([1, 0, 0])
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.3, origin=[0, 0, 0])

cyl_rot = create_cylinder_two_point(p1_sq,p2_sq,offset=0)
o3d.visualization.draw([line_set, pcd_1, pcd_2, mesh_frame_p3,
                       mesh_frame_p1, mesh_frame_p2,mesh_cylinder ,cyl_rot], show_ui=True)
