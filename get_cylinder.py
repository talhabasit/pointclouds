import open3d as o3d
import numpy as np
from angle import rod_rot
from numba import jit
import time

def create_cylinder_two_point(p1,p2,radius=0.1):
	mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(p2-p1)+.05)
	mesh_cylinder.compute_vertex_normals()
	mesh_cylinder.paint_uniform_color([1, 0, 0])
	mid_point = np.squeeze(p1-1*(p1-p2)/2)
	resultant_norm = (p2-p1)/np.linalg.norm(p2-p1)
	z = np.eye(3)[2]
	R = rod_rot(z,resultant_norm)
	cyl_rot = mesh_cylinder.rotate(R).translate(mid_point)
	return cyl_rot

def main():
	p1_sq = np.array([4, 2, 1])
	p2_sq = np.array([3, 3, 3])
	start  = time.time_ns()
	cyl_rot = create_cylinder_two_point(p1_sq,p2_sq)
	print((time.time_ns()-start)/1e6)
	mesh_frame_p1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
		size=0.3, origin=p1_sq)
	mesh_frame_p2 = o3d.geometry.TriangleMesh.create_coordinate_frame(
		size=0.3, origin=p2_sq)
	o3d.visualization.draw([cyl_rot,mesh_frame_p1,mesh_frame_p2], show_ui=True)
	


if __name__=="__main__":
	main()
	

