from cmath import pi
import open3d as o3d
import numpy as np
import pickle 
import cv2 
import os 
import natsort as ns
from open3d import io 
import math as m
from open3d.visualization import draw_geometries_with_vertex_selection



w=1280
h=720

fx=646.358
fy=646.358
ppx=643.229
ppy=360.057

x=(ppx+ppy)/1000

joints_dir = "./joints/"

def read_data():
    joints_list=[]

    for file in ns.natsorted(os.listdir(joints_dir)):
        filename = os.path.join(joints_dir,file)
        joints_list.append(np.loadtxt(filename))
    
    joints_list=np.asanyarray(ns.natsorted(joints_list))
    
    return joints_list

#read_joints=read_data()

intrinsics=o3d.camera.PinholeCameraIntrinsic(w,h,fx,fy,ppx,ppy)


img_load=np.load("C:/Users/talha/Desktop/Nuitrack/Nuitrack/pcd/rgb/1666035128205312_1666035162425369800.npy")
depth_load=np.load("C:/Users/talha/Desktop/Nuitrack/Nuitrack/pcd/depth/1666035128205312_1666035162425369800.npy")
img_load=cv2.cvtColor(img_load,cv2.COLOR_BGR2RGB)

color_raw = o3d.geometry.Image(img_load)
depth_raw = o3d.geometry.Image(depth_load)



joint=np.loadtxt("C:/Users/talha/Desktop/Nuitrack/Nuitrack/joints/1666035128205312_1666035162425369800.txt")
#lineset=io.read_line_set("C:/Users/talha/Desktop/Nuitrack/PyNuitrack/Nuitrack/joints/1666035128205312_1666035164807039300.txt")
# joints_image=o3d.geometry.Image(np.float16(joint))


pcd_joints= o3d.geometry.PointCloud()
pcd_joints.points = o3d.utility.Vector3dVector(joint)
pcd_joints.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
# pcd_joints.transform([[1, 0, 0, -x], [0, -1, 0, 0.5], [0, 0, -1, -0.1], [0, 0, 0, 1]])

# pcd_joints.transform([[1, 0, 0, -x], [0, -1, 0, 0.5], [0, 0, -1, -0.1], [0, 0, 0, 1]])
# Best config till now

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

	z = np.transpose(depth_image).flatten()
	x = np.multiply(x,z)
	y = np.multiply(y,z)

	"""x = x[np.nonzero(z)]
	y = y[np.nonzero(z)]
	z = z[np.nonzero(z)]"""

	return x, y, z


# a,b,c=convert_depth_frame_to_pointcloud(joint)
# xyz=np.column_stack((a,b,c))
# pcd_joints.points = o3d.utility.Vector3dVector(xyz)
# pcd_joints.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,depth_trunc=3,convert_rgb_to_intensity=False)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,intrinsic= intrinsics)
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def distance_two_points(p1,p2):
    subt=np.subtract(p2,p1)
    dist = np.linalg.norm(subt,ord=2)
    return np.asanyarray([dist,dist,-dist])

points=np.asanyarray(pcd_joints.points)
p0=points[0,:]
p1=points[1,:]
p2=points[2,:]

extent=np.asanyarray([0.3,0.3,0.3])
angle=np.arccos(np.sum(np.multiply(p0,p1))/(np.linalg.norm(p0,ord=2)*np.linalg.norm(p1,ord=2)))
rotation=pcd.get_rotation_matrix_from_xyz((p0))
rotation=Ry(np.rad2deg(angle))*Rx(np.rad2deg(angle))*Rz(np.rad2deg(angle))

center=p0
bbox=o3d.geometry.OrientedBoundingBox(center,rotation,extent)


# vis = o3d.visualization.VisualizerWithKeyCallback()
# vis.create_window()
# vis.add_geometry(pcd_joints)
# vis.add_geometry(pcd)

# def key_action_callback(vis, action, mods):
# 	print(action)
# 	if action == 1:  # key down
# 		vis.destroy_window()
# 	return True

# key_press=vis.register_key_action_callback(ord("Q"),key_action_callback)


# while 1:
# 	vis.poll_events()
# 	vis.update_renderer()
# 	# if 
	

# y=o3d.visualization.draw([pcd_joints,pcd,bbox],show_ui=True)
y=o3d.visualization.draw([pcd_joints,pcd],show_ui=True)
# y=o3d.visualization.draw([pcd_joints],show_ui=True)

#shoulder=(-0.52, 0.28, -0.82)
#elbow=(-0.43, 0.11, -0.58)
#wrist=(-0.33, -0.00, -0.38)



