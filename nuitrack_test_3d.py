
from PyNuitrack import py_nuitrack
import cv2
import numpy as np
import open3d as o3d
import os
import concurrent.futures as mt
import time
from read_calib_file import get_intrinsics_from_json
import copy



intrinsics = get_intrinsics_from_json(1)

def convert_to_o3d(image,depth,skeletons,first_call):
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

	return pcd,pcd_joints

def key_action_callback(vis, action, mods):
	if action == 1:  # key down
		vis.close()
		vis.destroy_window()
	return True

def main():
	o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
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

	vis = o3d.visualization.VisualizerWithKeyCallback()
	vis.create_window()
	ctr = vis.get_view_control()

	# for i in range(360):
	first_call = True
	mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0, 0, 0])
	while 1:
     
		start = time.time()
		nuitrack.update()
		data = nuitrack.get_skeleton()
		img_depth = nuitrack.get_depth_data()
		img_color = nuitrack.get_color_data()

		if img_depth.size and img_color.size:

			all_joints=[]
			if data[1]:
				first_skeleton = data[2][0]
				for joints in range(1,21):
					all_joints.append(first_skeleton[joints].real)
			else:
				# all_joints=[np.random.rand(3) for _ in range(1,21)]
				all_joints=[np.zeros(3,) for _ in range(1,21)]
			all_joints = np.squeeze(np.asanyarray(all_joints)/1000.0)
			all_joints[:,2] = all_joints[:,2]*-1
			if first_call:
				pcd_joints= o3d.geometry.PointCloud()
				pcd_joints.points = o3d.utility.Vector3dVector(all_joints)
			else:
				pcd_joints.points=o3d.utility.Vector3dVector(all_joints)
    
			#pcd_joints.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
   
			color_raw = o3d.geometry.Image(img_color)
			depth_raw = o3d.geometry.Image(img_depth)

			rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,depth_trunc=7,convert_rgb_to_intensity=False)

			temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,intrinsic= intrinsics, project_valid_depth_only = False)

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
				vis.add_geometry(mesh_frame)
				first_call = False
			else:
				vis.update_geometry(pcd_joints)
				vis.update_geometry(pcd)
		vis.poll_events()
		vis.update_renderer()
			
		print(time.time()-start)

	nuitrack.release()
 
if __name__=="__main__":
    main()