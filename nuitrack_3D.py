from PyNuitrack import py_nuitrack
import cv2
import numpy as np
import open3d as o3d
import os
import time
from utils.read_calib_file import get_intrinsics_from_json
from numba import njit, prange

file_name = os.path.basename(__file__)

joints_color = np.array([255, 0, 0])

intrinsics, fx, fy, cx, cy, width, height = get_intrinsics_from_json(1)
fps = 60  # limited to 60 fps by the RGB camera depth can go up to 90 fps with resolution 848x480 and lower


def convert_to_o3d(image, depth, skeletons, first_call):
    start = time.time()
    all_joints = []
    if skeletons[1]:
        first_skeleton = skeletons[2][0]
        for joints in range(1, 21):
            all_joints.append(first_skeleton[joints].projection)
    else:
        # all_joints=[np.random.rand(3) for _ in range(1,21)]
        all_joints = [
            np.zeros(
                3,
            )
            for _ in range(1, 21)
        ]
    all_joints = np.squeeze(np.asanyarray(all_joints) / 1000.0)

    if first_call:
        pcd_joints = o3d.geometry.PointCloud()
        pcd_joints.create_from_depth_image()
        pcd_joints.points = o3d.utility.Vector3dVector(all_joints)
    else:
        pcd_joints.points = o3d.utility.Vector3dVector(all_joints)

    color_raw = o3d.geometry.Image(image)
    depth_raw = o3d.geometry.Image(depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_trunc=10, convert_rgb_to_intensity=True
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic=intrinsics, project_valid_depth_only=False
    )
    print(time.time() - start)
    return pcd, pcd_joints


@njit(parallel=True, fastmath=True)
def depth_from_x_y_z_joints(joint_array: np.ndarray):
    """This function is used to convert the joints from x,y,z to depth
    Note: parallel=True is used to parallelize the for loop, may not be faster for small arrays
    """
    for i in prange(joint_array.shape[0]):
        joint_array[i, 0] = (joint_array[i, 0] - cx) * joint_array[i, 2] / fx
        joint_array[i, 1] = (joint_array[i, 1] - cy) * joint_array[i, 2] / fy
    return joint_array


def init_nuitrack():
    """Intitialize nuitrack and set the config values
    Returns: nuitrack object"""
    nuitrack = py_nuitrack.Nuitrack()
    nuitrack.init()

    # ---enable if you want to use face tracking---
    # nuitrack.set_config_value("Faces.ToUse", "true")
    nuitrack.set_config_value("DepthProvider.Depth2ColorRegistration", "true")
    nuitrack.set_config_value("Realsense2Module.Depth.ProcessWidth", f"{width}")
    nuitrack.set_config_value("Realsense2Module.Depth.ProcessHeight", f"{height}")
    nuitrack.set_config_value("Realsense2Module.Depth.ProcessMaxDepth", "7000")
    nuitrack.set_config_value("Realsense2Module.Depth.Preset", "2")
    nuitrack.set_config_value("Realsense2Module.Depth.FPS", f"{fps}")
    nuitrack.set_config_value("Realsense2Module.RGB.ProcessWidth", f"{width}")
    nuitrack.set_config_value("Realsense2Module.RGB.ProcessHeight", f"{height}")
    nuitrack.set_config_value("Realsense2Module.RGB.FPS", f"{fps}")
    # nuitrack.set_config_value(
    #     "Segmentation.Background.BackgroundMode", "static_first_frame"
    # )
    # nuitrack.set_config_value("Segmentation.Background.CalibrationFramesNumber", "20")

    devices = nuitrack.get_device_list()

    for i, dev in enumerate(devices):
        print(dev.get_name(), dev.get_serial_number())
        if i == 0:
            # dev.activate("ACTIVATION_KEY") #you can activate device using python api
            print(dev.get_activation())
            nuitrack.set_device(dev)

    nuitrack.create_modules()
    nuitrack.run()
    return nuitrack


def return_joints(
    skeleton_data=np.tile(
        np.zeros(shape=(3,)),
        (21, 1),
    ),
    skeleton=False,
):
    """This function is used to return the joints from the nuitrack data
    Returns: joints in depth or x,y,z format if a user is in frame
    otherwise it returns the zero array
    """

    if skeleton:
        all_joints = np.zeros((21, 3))
        first_skeleton = skeleton_data
        for joint_number in range(1, 21):
            if first_skeleton[joint_number].confidence > 0.6:
                all_joints[joint_number, :] = first_skeleton[joint_number].projection
        all_joints = depth_from_x_y_z_joints(all_joints)
        all_joints = all_joints / 1000.0  # convert to meters
        return all_joints
    else:
        return skeleton_data


def create_pcd_from_img_depth(img_color, img_depth, downsample=False, ds_factor=0.1):
    # This code takes an image and depth map, and creates a point cloud from them.
    # We use the intrinsic matrix to create an RGBD image, which is then converted
    # into a point cloud. If the downsample flag is set, we downsample the point cloud
    # by the given factor (0.1 means 10x downsample).

    # Create o3d image objects from the given numpy arrays
    color_raw = o3d.geometry.Image(img_color)
    depth_raw = o3d.geometry.Image(img_depth)

    # Create an RGBD image from the color and depth images, truncating the depth
    # at 3.5 meters and not converting the RGB image to intensity
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_trunc=3.5, convert_rgb_to_intensity=False
    )

    # Create a point cloud from the RGBD image, using the given intrinsics
    # We only project the valid depth pixels
    temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic=intrinsics, project_valid_depth_only=True
    )
    # If the downsample flag is set, downsample the point cloud
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
    rd_options.show_coordinate_frame = (
        True  # should show the coordinate frame but doesnt work for some reason
    )

    def key_action_callback(vis, action, mods):
        if action == 1:  # key down
            vis.destroy_window()
            vis.close()
            nuitrack.release()
            # os._exit(0)
        return False

    vis.register_key_action_callback(ord("Q"), key_action_callback)

    while 1:
        # Main loop
        start = time.time_ns()
        nuitrack.update()
        data = nuitrack.get_skeleton()
        img_depth = nuitrack.get_depth_data()
        img_color = nuitrack.get_color_data()
        # TODO: add lines to the joint to make it easier to see
        if img_depth.size and img_color.size:
            if data[1] != 0:
                bed = True
                all_joints = return_joints(data[2][0], bed)
            else:
                all_joints = return_joints()

            if first_call:
                # Create the point cloud from the joints during the first call
                pcd_joints = o3d.geometry.PointCloud()
                pcd_joints.points = o3d.utility.Vector3dVector(all_joints)
            else:
                # Only Update the point cloud from the joints during subsequent calls
                pcd_joints.points = o3d.utility.Vector3dVector(all_joints)

            pcd_joints.paint_uniform_color([1, 0, 0])
            pcd_joints.transform(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            )

            # Realsense uses BGR
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

            temp_pcd = create_pcd_from_img_depth(img_color, img_depth)

            if first_call:
                pcd = o3d.geometry.PointCloud()
                # This is a dumb and sad workaround but it "works"
                # TODO: figure out how to change this to directly use the Array
                pcd.points = temp_pcd.points
                # pcd.colors = temp_pcd.colors
                pcd.transform(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
                )
            else:
                pcd.points = temp_pcd.points
                # pcd.colors = temp_pcd.colors # uncomment if you want to color the point cloud
                pcd.transform(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
                )

            if first_call:
                # Add the point cloud to the visualizer during the first call
                vis.add_geometry(pcd_joints)
                vis.add_geometry(pcd)

                first_call = False
            else:
                # Only Update the point cloud during subsequent calls
                vis.update_geometry(pcd_joints)
                vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()

        print(f"{(time.time_ns()-start)/1e6} ms")


if __name__ == "__main__":
    # import cProfile
    # import pstats

    # with cProfile.Profile() as pr:
    main()

    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    # stats.dump_stats(
    # 	filename=f"./profiling_runs/{file_name}_{time.time_ns}.prof")
