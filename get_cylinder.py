import open3d as o3d
import numpy as np
from angle import rod_rot
from numba import jit
import time

def create_cylinder_two_point(p1,p2,radius=0.1,offset=0.0):
	mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(p2-p1)+offset)
	mesh_cylinder.paint_uniform_color([1, 0, 0])
	mid_point = np.squeeze(p1-1*(p1-p2)/2) # mid point between p1 and p2
	resultant_norm = (p2-p1)/np.linalg.norm(p2-p1) # resultant unit vector
	z = np.array([0,0,1]).astype(np.float64) # The cylinder is created around the z axis
	# start = time.monotonic_ns()
	R = rod_rot(z,resultant_norm) # rotation matrix
	# print((time.monotonic_ns()-start))
	cyl_rot = mesh_cylinder.rotate(R).translate(mid_point) # rotate and translate
	return cyl_rot

def main(): #Testng cylinder transformations
	#Define two arbitrary points
	p1_sq = np.array([4, 2, 1])
	p2_sq = np.array([1, 2, 3])
	#Create cylinder between these 
	cyl_rot = create_cylinder_two_point(p1_sq,p2_sq)
	#Plot cylinder and coordinate frames
	mesh_frame_p1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
		size=0.3, origin=p1_sq)
	mesh_frame_p2 = o3d.geometry.TriangleMesh.create_coordinate_frame(
		size=0.3, origin=p2_sq)
	o3d.visualization.draw([cyl_rot,mesh_frame_p1,mesh_frame_p2], show_ui=True)
	


if __name__=="__main__":
	main()
	

