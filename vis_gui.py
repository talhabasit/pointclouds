import open3d as o3d
import copy

mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.3,
                                                          height=4.0)

mesh_cylinder.compute_vertex_normals()
mesh_cylinder.paint_uniform_color([0, 0, 0])
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[-2, -2, -2])

print("We draw a few primitives using collection.")
p1=[1,1,1]
p2=[2,2,2]
cyl_rot=copy.deepcopy(mesh_cylinder).translate(p2).paint_uniform_color([0.1, 0.9, 0.1]).transform([[1, 0, 1, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

o3d.visualization.draw_geometries([mesh_cylinder, mesh_frame,cyl_rot])



