import open3d as o3d
import numpy as np
import cv2
from get_cylinder import create_cylinder_two_point
from nuitrack_3D import depth_from_x_y_z_joints
from read_calib_file import get_intrinsics_from_json
import plyfile as ply
import os
import time
import sys


file_name = os.path.basename(__file__)


forearm_color = np.array([1, 0, 0])
upperarm_color = np.array([0, 1, 0])

intrinsics, fx, fy, cx, cy, width, height = get_intrinsics_from_json(1)


def load_joints_as_pts(filename):
    # Load the numpy file containing the joints
    joints = np.load(f"./joints/{filename}.npy").astype(np.float64)
    # Convert the joints into depth values
    joints = depth_from_x_y_z_joints(joints) / 1000.0

    return joints[0], joints[1], joints[2]


# @njit(parallel = True,fastmath = True)
def return_class_column_array(cylinder_list, pcd_points, class_label):
    # Create a numpy array of zeros with the same length as the pcd_points array
    x = np.zeros(pcd_points.shape[0])
    # Set the values of the array at the index positions in the cylinder_list to the class_label
    x[cylinder_list] = class_label
    return x


def write_3d_point_cloud_to_ply(
    path_to_ply_file,
    coordinates,
    colors=None,
    extra_properties=None,
    extra_properties_names=None,
    comments=[],
    text=True,
):
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
    dtypes = [
        ("x", coordinates.dtype),
        ("y", coordinates.dtype),
        ("z", coordinates.dtype),
    ]

    if colors is not None:
        # Check that the colors are valid
        if colors.shape[1] == 1:  # replicate grayscale 3 times
            colors = np.column_stack([colors] * 3)
        elif colors.shape[1] != 3:
            raise Exception("Error: colors must have either 1 or 3 columns")
        # Add the colors to the points array
        points = np.column_stack((points, colors))
        dtypes += [
            ("red", colors.dtype),
            ("green", colors.dtype),
            ("blue", colors.dtype),
        ]

    if extra_properties is not None:
        # Add the extra properties to the points array
        points = np.column_stack((points, extra_properties))
        dtypes += [(s, extra_properties.dtype) for s in extra_properties_names]

    # Convert the points array to a list of tuples
    tuples = [tuple(x) for x in points]
    # Create a ply element using the tuples
    plydata = ply.PlyElement.describe(np.asarray(tuples, dtype=dtypes), "vertex")
    # Write the ply file
    ply.PlyData([plydata], comments=comments, text=text).write(path_to_ply_file)


def create_pcd_from_img_depth(img_color, img_depth, downsample=False, ds_factor=0.1):
    # Create an Open3D Image object from the color image
    color_raw = o3d.geometry.Image(img_color)
    # Create an Open3D Image object from the depth image
    depth_raw = o3d.geometry.Image(img_depth)

    # Create an Open3D RGBDImage object from the color and depth images
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_trunc=10, convert_rgb_to_intensity=True
    )

    # Create a point cloud from the RGBDImage object
    temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic=intrinsics, project_valid_depth_only=True
    )
    # Downsample the point cloud
    if downsample:
        temp_pcd = temp_pcd.voxel_down_sample(voxel_size=ds_factor)
    # Return the point cloud
    return temp_pcd


def create_tk_dialog():
    """Create a tkinter dialog to ask the user to choose a file to display, as well as whether to downsample the pointcloud"""
    import tkinter as tk
    from tkinter import simpledialog
    from tkinter import filedialog as fd, messagebox

    parent = tk.Tk("Choose a file to display")  # Create the object
    parent.overrideredirect(1)  # Avoid it appearing and then disappearing quickly
    parent.withdraw()  # Hide the parent window
    filename = fd.askopenfilename(
        parent=parent,
        title="Choose a file to display",
        filetypes=[("npy files", ".npy")],
    )

    if not ".npy" in filename:
        print("No file selected")
        sys.exit()

    filename = filename.split("/")[-1].split(".npy")[0]
    p1, p2, p3 = load_joints_as_pts(filename)

    # Create a dialog to ask user to choose from a list of options

    response = messagebox.askyesnocancel(
        "Downsampling", "Do you want to downsample the pointcloud?"
    )
    if response and response is not None:
        float_input = simpledialog.askfloat(
            "Enter a float Value",
            "Enter the number of points to keep from the original pointcloud",
            parent=parent,
            maxvalue=10000,
            minvalue=0.0,
            initialvalue=4096,
        )
        if float_input is None:
            sys.exit()
    elif response is None:
        print("Cancelled")
        sys.exit()
    else:
        float_input = None
        print("No downsampling")

    parent.destroy()  # Destroy the object

    return p1, p2, p3, filename, response, float_input


def main():
    p1, p2, p3, filename, down, down_percent = create_tk_dialog()
    forearm_cyl = create_cylinder_two_point(p1, p2, radius=0.1)
    upperarm_cyl = create_cylinder_two_point(p2, p3, radius=0.1, offset=0.1)

    forearm_cyl.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    upperarm_cyl.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # mesh_frame_p1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=p1)
    # mesh_frame_p2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=p2)
    # mesh_frame_p3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=p3)

    img_load = np.load(f"./pcd/rgb/{filename}.npy")
    depth_load = np.load(f"./pcd/depth/{filename}.npy")
    img_load = cv2.cvtColor(img_load, cv2.COLOR_BGR2RGB)

    pcd = create_pcd_from_img_depth(img_load, depth_load)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([0, 0, 0.5]), (np.asarray(pcd.points).shape[0], 1))
    )
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # pcd= pcd.uniform_down_sample(every_k_points=50)
    if down and down is not None:
        pcd = pcd.farthest_point_down_sample(4096)

    # pcd_down_sample = pcd.uniform_down_sample(every_k_points=1000)

    forearm_bb = forearm_cyl.get_oriented_bounding_box()
    upperarm_bb = upperarm_cyl.get_oriented_bounding_box()

    list_forearm = forearm_bb.get_point_indices_within_bounding_box(pcd.points)
    list_upperarm = upperarm_bb.get_point_indices_within_bounding_box(pcd.points)
    # same_elements = np.intersect1d(list_forearm,list_upperarm)#intersection of two lists
    same_elements = list(set(list_forearm).intersection(list_upperarm))

    for x in same_elements:
        list_upperarm.remove(x)

    forearm_pcd = pcd.select_by_index(list_forearm)
    upperarm_pcd = pcd.select_by_index(list_upperarm)

    forearm_pcd.paint_uniform_color(forearm_color)
    upperarm_pcd.paint_uniform_color(upperarm_color)

    print(f"Same points :{same_elements.__len__()}")
    vis = o3d.visualization.draw(
        [
            forearm_cyl,
            upperarm_cyl,
            forearm_pcd,
            upperarm_pcd,
            upperarm_bb,
            forearm_bb,
            pcd,
        ],
        show_ui=True,
    )


if __name__ == "__main__":
    # import cProfile
    # import pstats
    # with cProfile.Profile() as pr:
    main()

    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # #stats.print_stats()
    # stats.dump_stats(filename=f"./profiling_runs/{file_name}_{time.time_ns()}.prof")
