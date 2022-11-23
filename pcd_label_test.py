import open3d as o3d
import numpy as np
import cv2
from get_cylinder import create_cylinder_two_point
from nuitrack_test_3d import depth_from_x_y_z_joints,create_pcd_from_img_depth
from read_calib_file import get_intrinsics_from_json

forearm_color = np.array([1,0,0])
upperarm_color = np.array([0,255,0])


intrinsics,fx,fy,cx,cy = get_intrinsics_from_json(1)





def load_joints_as_pts():
#wrist elbow shoulder
	joints = np.load("./joints/1669222128156672_1669222104307698700.npy").astype(np.float64)
	joints = depth_from_x_y_z_joints(joints)/1000

	return joints[0],joints[1],joints[2]






def main():
	
	p1,p2,p3 = load_joints_as_pts()

	forearm_cyl = create_cylinder_two_point(p1,p2,.15)
	upperarm_cyl = create_cylinder_two_point(p2,p3,.15)
	forearm_cyl.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	upperarm_cyl.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	mesh_frame_p1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
		size=0.3, origin=p1)
	mesh_frame_p2 = o3d.geometry.TriangleMesh.create_coordinate_frame(
		size=0.3, origin=p2)
	mesh_frame_p3 = o3d.geometry.TriangleMesh.create_coordinate_frame(
		size=0.3, origin=p3)
 
	img_load=np.load("./pcd/rgb/1669222128156672_1669222104307698700.npy")
	depth_load=np.load("./pcd/depth/1669222128156672_1669222104307698700.npy")
	img_load=cv2.cvtColor(img_load,cv2.COLOR_BGR2RGB)

	pcd = create_pcd_from_img_depth(img_load,depth_load)
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

	
	forearm_bb = forearm_cyl.get_oriented_bounding_box()
	upperarm_bb = upperarm_cyl.get_oriented_bounding_box()
	
	forearm_pcd = pcd.select_by_index(forearm_bb.get_point_indices_within_bounding_box(pcd.points))
	upperarm_pcd = pcd.select_by_index(upperarm_bb.get_point_indices_within_bounding_box(pcd.points))
	print(forearm_pcd)

	o3d.io.write_point_cloud("./test.pts",forearm_pcd)
	o3d.visualization.draw([forearm_pcd,upperarm_pcd], show_ui=True)
	


if __name__=="__main__":
	main()
	


		