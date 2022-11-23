
from PyNuitrack import py_nuitrack
import cv2
import numpy as np
import open3d as o3d
import os
import concurrent.futures as mt
import time
from read_calib_file import get_intrinsics_from_json
import copy
import sys
from numba import njit,prange,jit


joints_color = np.array([255,0,0])

intrinsics,fx,fy,cx,cy = get_intrinsics_from_json(1)

def convert_to_o3d(image,depth,skeletons,first_call):
	start = time.time()
	all_joints=[]
	if skeletons[1]:
		first_skeleton = skeletons[2][0]
		for joints in range(1,21):
			all_joints.append(first_skeleton[joints].projection)
	else:
		# all_joints=[np.random.rand(3) for _ in range(1,21)]
		all_joints=[np.zeros(3,) for _ in range(1,21)]
	all_joints = np.squeeze(np.asanyarray(all_joints)/1000.0)

	if first_call:
		pcd_joints= o3d.geometry.PointCloud()
		pcd_joints.create_from_depth_image()
		pcd_joints.points = o3d.utility.Vector3dVector(all_joints)
	else:
		pcd_joints.points=o3d.utility.Vector3dVector(all_joints)

	color_raw = o3d.geometry.Image(image)
	depth_raw = o3d.geometry.Image(depth)

	rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,depth_trunc=10,convert_rgb_to_intensity=True)

	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,intrinsic= intrinsics, project_valid_depth_only = False)
	print(time.time()-start)
	return pcd,pcd_joints


@njit(cache=True,parallel=True,fastmath=True)
def depth_from_x_y_z_joints(joint_array:np.ndarray):

    for i in prange(joint_array.shape[0]):
        joint_array[i,0] =  (joint_array[i,0]-cx)*joint_array[i,2]/fx
        joint_array[i,1] =  (joint_array[i,1]-cy)*joint_array[i,2]/fy
    return joint_array


def init_nuitrack():
    
	nuitrack = py_nuitrack.Nuitrack()
	nuitrack.init()

	# ---enable if you want to use face tracking---
	#nuitrack.set_config_value("Faces.ToUse", "true")
	nuitrack.set_config_value("DepthProvider.Depth2ColorRegistration", "true")
	nuitrack.set_config_value("Realsense2Module.Depth.ProcessWidth", "1280")
	nuitrack.set_config_value("Realsense2Module.Depth.ProcessHeight", "720")
	nuitrack.set_config_value("Realsense2Module.Depth.ProcessMaxDepth", "7000")
	nuitrack.set_config_value("Realsense2Module.RGB.ProcessWidth", "1280")
	nuitrack.set_config_value("Realsense2Module.RGB.ProcessHeight", "720")

	devices = nuitrack.get_device_list()

	for i, dev in enumerate(devices):
		print(dev.get_name(), dev.get_serial_number())
		if i == 0:
			#dev.activate("ACTIVATION_KEY") #you can activate device using python api
			print(dev.get_activation())
			nuitrack.set_device(dev)

	nuitrack.create_modules()
	nuitrack.run()
	return nuitrack

def return_joints(skeleton_data=np.tile(np.zeros(3,),(21,1)), skeleton = False):
	if skeleton:
		all_joints= np.zeros((21,3))
		first_skeleton = skeleton_data
		for joint_number in range(1,21):
			all_joints[joint_number,:]=first_skeleton[joint_number].projection
		all_joints = depth_from_x_y_z_joints(all_joints)
		all_joints = all_joints/1000.0
		return  all_joints
	else:
		return skeleton_data 

def create_pcd_from_img_depth(img_color,img_depth,downsample=False, ds_factor = 0):
	color_raw = o3d.geometry.Image(img_color)
	depth_raw = o3d.geometry.Image(img_depth)

	rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,depth_trunc=7,convert_rgb_to_intensity=False)

	temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,intrinsic= intrinsics, project_valid_depth_only = False)
	if downsample:
		temp_pcd = temp_pcd.voxel_down_sample(voxel_size=ds_factor)	
	return temp_pcd

def main():
	vis = o3d.visualization.VisualizerWithKeyCallback()
	vis.create_window()
	nuitrack = init_nuitrack()
	# for i in range(360):
 
	first_call = True
	rd_options = vis.get_render_option()
	rd_options.point_size = 3
	rd_options.show_coordinate_frame = True
 
	def key_action_callback(vis, action, mods):
		if action == 1:  # key down
			vis.destroy_window()
			vis.close()
			nuitrack.release()
			os._exit(0)
		return True

	
	while 1:
    
		start = time.time_ns()
		nuitrack.update()
		data = nuitrack.get_skeleton()
		img_depth = nuitrack.get_depth_data()
		img_color = nuitrack.get_color_data()

		if img_depth.size and img_color.size:
			if data[1] != 0:
				bed = True
				all_joints = return_joints(data[2][0],bed)
			else:
				all_joints= return_joints()

			if first_call:
				pcd_joints= o3d.geometry.PointCloud()
				pcd_joints.color=o3d.utility.Vector3dVector(np.tile(joints_color,(21,1)))
				pcd_joints.points = o3d.utility.Vector3dVector(all_joints)
			else:
				pcd_joints.points=o3d.utility.Vector3dVector(all_joints)
    
			pcd_joints.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

			img_color = cv2.cvtColor(img_color,cv2.COLOR_BGR2RGB)
			temp_pcd = create_pcd_from_img_depth(img_color,img_depth)

			vis.register_key_action_callback(ord("Q"),key_action_callback)

			if first_call:
				pcd = o3d.geometry.PointCloud()
				pcd.points = temp_pcd.points
				pcd.colors = temp_pcd.colors
				pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
			else:
				pcd.points = temp_pcd.points
				pcd.colors = temp_pcd.colors
				pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

			if first_call:
				vis.add_geometry(pcd_joints)
				vis.add_geometry(pcd)
				first_call = False
			else:
				vis.update_geometry(pcd_joints)
				vis.update_geometry(pcd)
		vis.poll_events()
		vis.update_renderer()
			
		print(f"{(time.time_ns()-start)/1e6} ms")
	#nuitrack.release()
	print("while end")
 
if __name__=="__main__":
	main()