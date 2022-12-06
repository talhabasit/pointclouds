import cProfile
import pstats
import open3d as o3d
import numpy as np
import cv2
from get_cylinder import create_cylinder_two_point
from nuitrack_test_3d import depth_from_x_y_z_joints
from read_calib_file import get_intrinsics_from_json
import plyfile as ply
from functools import cache
from numba import njit
import os
import time

from tkinter import filedialog as fd 


file_name = os.path.basename(__file__)


forearm_color = np.array([1,0,0])
upperarm_color = np.array([0,255,0])


intrinsics,fx,fy,cx,cy, width, height = get_intrinsics_from_json(3)


@cache
def load_joints_as_pts(filename):
#wrist elbow shoulder
	joints = np.load(f"./joints/{filename}.npy").astype(np.float64)
	joints = depth_from_x_y_z_joints(joints)/1000

	return joints[0],joints[1],joints[2]

@njit(parallel = True,fastmath = True)
def return_class_column_array(cylinder_list,pcd_points,class_label):
	x = np.zeros(pcd_points.shape[0])
	x[cylinder_list] = class_label
	return x
	
def save_array_with_labels():
	pass
	

def write_3d_point_cloud_to_ply(path_to_ply_file, coordinates, colors=None,
								extra_properties=None,
								extra_properties_names=None, 
								comments=[],
								text = True):
	"""
	Write a 3D point cloud to a ply file.

	Args:
		path_to_ply_file (str): path to a .ply file
		coordinates (array): numpy array of shape (n, 3) containing x, y, z coordinates
		colors (array): numpy array of shape (n, 3) or (n, 1) containing either
			r, g, b or gray levels
		extra_properties (array): optional numpy array of shape (n, k)
		extra_properties_names (list): list of k strings with the names of the
			(optional) extra properties
		comments (list): list of strings containing the ply header comments
	"""
	points = coordinates
	dtypes = [('x', coordinates.dtype),
			  ('y', coordinates.dtype),
			  ('z', coordinates.dtype)]

	if colors is not None:
		if colors.shape[1] == 1:  # replicate grayscale 3 times
			colors = np.column_stack([colors] * 3)
		elif colors.shape[1] != 3:
			raise Exception('Error: colors must have either 1 or 3 columns')
		points = np.column_stack((points, colors))
		dtypes += [('red', colors.dtype),
				   ('green', colors.dtype),
				   ('blue', colors.dtype)]

	if extra_properties is not None:
		points = np.column_stack((points, extra_properties))
		dtypes += [(s, extra_properties.dtype) for s in extra_properties_names]

	tuples = [tuple(x) for x in points]
	plydata = ply.PlyElement.describe(np.asarray(tuples, dtype=dtypes),
										  'vertex')
	ply.PlyData([plydata], comments=comments,text=text).write(path_to_ply_file) 

def create_pcd_from_img_depth(img_color, img_depth, downsample=False, ds_factor=0.1):
	color_raw = o3d.geometry.Image(img_color)
	depth_raw = o3d.geometry.Image(img_depth)

	rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
		color_raw, depth_raw, depth_trunc=10, convert_rgb_to_intensity=False)

	temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
		rgbd_image, intrinsic=intrinsics, project_valid_depth_only=True)
	if downsample:
		temp_pcd = temp_pcd.voxel_down_sample(voxel_size=ds_factor)
	return temp_pcd


def main():
	filename = fd.askopenfilename()
	filename = filename.split("/")[-1].split(".npy")[0]
	p1,p2,p3 = load_joints_as_pts(filename)


	# write_3d_point_cloud_to_ply("test.ply",x,extra_properties=labels,extra_properties_names=["label"])

	forearm_cyl = create_cylinder_two_point(p1,p2,.1)
	upperarm_cyl = create_cylinder_two_point(p2,p3,.1)
 
	forearm_cyl.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	upperarm_cyl.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
 
	# mesh_frame_p1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=p1)
	# mesh_frame_p2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=p2)
	# mesh_frame_p3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=p3)
 
	img_load=np.load(f"./pcd/rgb/{filename}.npy")
	depth_load=np.load(f"./pcd/depth/{filename}.npy")
	img_load=cv2.cvtColor(img_load,cv2.COLOR_BGR2RGB)

	pcd = create_pcd_from_img_depth(img_load,depth_load)
	# pcd = o3d.geometry.PointCloud()
	# depth_raw = o3d.geometry.Image(depth_load)
	# pcd = pcd.create_from_depth_image(depth_raw,intrinsics)
	# pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0.0, 1.0, size=[np.asarray(pcd.points).shape[0], 3]))
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	pcd_points = np.asarray(pcd.points)

	# pcd= pcd.uniform_down_sample(every_k_points=100)
	#pcd= pcd.random_down_sample(0.1)
	
	# pcd_down_sample = pcd.uniform_down_sample(every_k_points=100)
	
	forearm_bb = forearm_cyl.get_oriented_bounding_box()
	upperarm_bb = upperarm_cyl.get_oriented_bounding_box()
	
	list_forearm = forearm_bb.get_point_indices_within_bounding_box(pcd.points)
	list_upperarm = upperarm_bb.get_point_indices_within_bounding_box(pcd.points)
	same_elements = np.intersect1d(list_forearm,list_upperarm)
	
	for x in list(same_elements):
		list_upperarm.remove(x)
 
 
	forearm_pcd = pcd.select_by_index(list_forearm)
	upperarm_pcd = pcd.select_by_index(list_upperarm)
	forearm_pcd.paint_uniform_color([0, 0, 1])
	upperarm_pcd.paint_uniform_color([1, 0, 0])

	forearm_label = return_class_column_array(np.array(list_forearm),pcd_points,10)
	
	print(f"Same points :{same_elements.shape[0]}")
	# o3d.io.write_point_cloud("./test.pts",forearm_pcd)
	o3d.visualization.draw([forearm_pcd,upperarm_pcd,upperarm_bb,forearm_bb,pcd], show_ui=True)
	# o3d.visualization.draw([test], show_ui=True)
	# o3d.visualization.draw([pcd,upperarm_bb,forearm_bb], show_ui=True)

if __name__ == '__main__':
	with cProfile.Profile() as pr:
		main()

	stats = pstats.Stats(pr)
	stats.sort_stats(pstats.SortKey.TIME)
	#stats.print_stats()
	stats.dump_stats(filename=f"./profiling_runs/{file_name}_{time.time_ns()}.prof")
	


		