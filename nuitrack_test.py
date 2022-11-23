
from PyNuitrack import py_nuitrack
import cv2
from itertools import cycle
import numpy as np
import time
import open3d as o3d
import os
import concurrent.futures as mt
import copy
import pandas
import sys
import pickle
import json

currentdir=os.path.dirname(__file__)
save_data = True

def draw_skeleton(data,image):
	point_color = (255, 255, 255)
	for skeleton in data.skeletons:
		p1=(round(skeleton.right_shoulder.projection[0]),round(skeleton.right_shoulder.projection[1]))
		p2=(round(skeleton.right_elbow.projection[0]),round(skeleton.right_elbow.projection[1]))
		p3=(round(skeleton.right_wrist.projection[0]),round(skeleton.right_wrist.projection[1]))
		cv2.circle(image, p1, 8, point_color, -1)
		cv2.circle(image, p2, 8, point_color, -1)
		cv2.circle(image, p3, 8, point_color, -1)
		cv2.line(image,p1 ,p2 ,color= point_color)
		cv2.line(image,p2 ,p3 ,color= point_color)
		print("skeleton")

def save_skeleton_as_array(data,timestamp,timens):
	for skeleton in data.skeletons:
		if save_data:
			with open("C:/Users/Basit/Desktop/PyNuitrack/Nuitrack/joints/{}_{}.npy".format(timestamp,timens),"wb") as f:
				skeleton_data = ([skeleton.right_wrist.projection[0],
				skeleton.right_wrist.projection[1],
				skeleton.right_wrist.projection[2]],
				[skeleton.right_elbow.projection[0],
				skeleton.right_elbow.projection[1],
				skeleton.right_elbow.projection[2]],
				[skeleton.right_shoulder.projection[0],
				skeleton.right_shoulder.projection[1],
				skeleton.right_shoulder.projection[2]])
				sekel=np.asanyarray(skeleton_data)/1000
				np.save(f,sekel)
    
def save_skeleton_as_dict(data,timens):
	if data.skeletons:
		if save_data:
			with open("./joints/{}_{}.json".format(data.timestamp,timens),"wb") as f:
				f.write(json.dumps(data._asdict()))
				print("saved_skeleton")

def save_npy(data,x,y,z,timestamp,timens):
	xyz=np.column_stack((x,y,z))
	if data.skeletons:
		if save_data:
			with open("C:/Users/Basit/Desktop/PyNuitrack/Nuitrack/pcd/{}_{}.npy".format(timestamp,timens),"wb") as f:
				np.save(f,xyz)
			# print(f"{timestamp}_saved")

def save_depth_as_array(data,array,timestamp,timens):
	if data.skeletons:
		if save_data:
			with open("./pcd/depth/{}_{}.npy".format(timestamp,timens),"wb") as f:
				np.save(f,array)
			# print(f"{timestamp}_saved")
   
def save_rgb_as_array(data,array,timestamp,timens):
	if data.skeletons:
			if save_data:
				with open("./pcd/rgb/{}_{}.npy".format(timestamp,timens),"wb") as f:
					np.save(f,array)
				# print(f"{timestamp}_saved")

def Savepcd(data,x,y,z,timestamp,timens):
	"""
	Takes XYZ data as Z16, converts it to a PCD and writes it to a file 
	"""
 
	xyz=np.column_stack((x,y,z))
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(xyz)
	#pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	if data.skeletons:
		if save_data:
			o3d.io.write_point_cloud("./pcd/{}_{}.ply".format(timestamp,timens), pcd, write_ascii = True)
 

def convert_depth_frame_to_pointcloud(depth_image):
	"""
	Convert the depthmap to a 3D point cloud
 
	Parameters:
	-----------
	depth_frame 	 	 : nuitrack depth frame 
						   The depth_frame containing the depth map
	fx,fy,ppx,ppy		 : Rectified focus and Principal points from the Intrinsic calibration data found in realsense

	Intrinsics from the Realsense calibration file for 1280x720

	fx=646.358
	fy=646.358
	ppx=643.229
	ppy=360.057


	Return:
	----------
	x : array
		The x values of the pointcloud in meters
	y : array
		The y values of the pointcloud in meters
	z : array
		The z values of the pointcloud in meters
	"""
	fx=646.358
	fy=646.358
	ppx=643.229
	ppy=360.057

	[height, width] = depth_image.shape

	nx = np.linspace(0, width-1, width)
	ny = np.linspace(0, height-1, height)
	u, v = np.meshgrid(nx, ny)
	x = (u.flatten() - ppx)/fx
	y = (v.flatten() - ppy)/fy

	z = np.transpose(depth_image).flatten()/1000
	x = np.multiply(x,z)
	y = np.multiply(y,z)

	"""x = x[np.nonzero(z)]
	y = y[np.nonzero(z)]
	z = z[np.nonzero(z)]"""

	return x, y, z


def main():

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

	modes = cycle(["depth", "color"])
	mode = next(modes)
	first_call = True
	while 1:
		
		timens = time.time_ns()

		key = cv2.waitKey(1)

		nuitrack.update()
		data = nuitrack.get_skeleton()
		img_depth = nuitrack.get_depth_data()
		img_color = nuitrack.get_color_data()
		
		x,y,z=convert_depth_frame_to_pointcloud(img_depth)

		if img_depth.size:
			
			with mt.ThreadPoolExecutor() as pool:#Tested---> consistent 1ms improvement

				task1=pool.submit(save_depth_as_array,data,img_depth,data.timestamp,timens)
	
				task2=pool.submit(save_rgb_as_array,data,img_color,data.timestamp,timens)

				save_skeleton_as_dict(data,timens)
	
				if mode == "depth":
					task3=pool.submit(draw_skeleton,data,img_depth)
				else:
					task3=pool.submit(draw_skeleton,data,img_color)
     
				futures=[task1,task2,task3] #Python internal datatype may incur overhead
				done, _ = mt.wait(futures, return_when=mt.ALL_COMPLETED)
				if done:
					pass
		
			cv2.normalize(img_depth, img_depth, 0, 255, cv2.NORM_MINMAX)#scale depth with depth-min/max and between (0,255)

			img_depth = np.array(cv2.cvtColor(img_depth,cv2.COLOR_GRAY2RGB), dtype=np.uint8)

		if key == 32:
			mode = next(modes)
		if mode == "depth":
			if img_depth.size:
				cv2.imshow('Image', img_depth)
		if mode == "color":
			if img_color.size:
				cv2.imshow('Image', img_color)
		if key == 27:
			break
		print("{} ms".format((time.time_ns()-timens)/1e6))

	nuitrack.release()

if __name__=="__main__":
	sys.exit(main())