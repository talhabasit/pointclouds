
import concurrent.futures as mt

import json
import os
import time
from itertools import cycle

import cv2
import numpy as np
import open3d as o3d

from PyNuitrack import py_nuitrack
from read_calib_file import get_intrinsics_from_json

file_name  = os.path.basename(__file__)
currentdir = os.path.dirname(__file__)
save_data = True


def draw_skeleton(data, image):
    point_color = (255, 255, 255)
    for skeleton in data.skeletons:
        p1 = (round(skeleton.right_shoulder.projection[0]), round(
            skeleton.right_shoulder.projection[1]))
        p2 = (round(skeleton.right_elbow.projection[0]), round(
            skeleton.right_elbow.projection[1]))
        p3 = (round(skeleton.right_wrist.projection[0]), round(
            skeleton.right_wrist.projection[1]))
        cv2.circle(image, p1, 8, point_color, -1)
        cv2.circle(image, p2, 8, point_color, -1)
        cv2.circle(image, p3, 8, point_color, -1)
        cv2.line(image, p1, p2, color=point_color,thickness=2)  
        cv2.line(image, p2, p3, color=point_color,thickness=2)


def save_skeleton_as_array(data, timestamp, timens):
    for skeleton in data.skeletons:
        if save_data:
            if not os.path.exists("./joints"):
                os.makedirs("./joints")
            with open("./joints/{}_{}.npy".format(timestamp, timens), "wb") as f:
                skeleton_data = ([skeleton.right_wrist.projection[0],
                                  skeleton.right_wrist.projection[1],
                                  skeleton.right_wrist.projection[2]],
                                 [skeleton.right_elbow.projection[0],
                                  skeleton.right_elbow.projection[1],
                                  skeleton.right_elbow.projection[2]],
                                 [skeleton.right_shoulder.projection[0],
                                  skeleton.right_shoulder.projection[1],
                                  skeleton.right_shoulder.projection[2]])
                sekel = np.asanyarray(skeleton_data)
                np.save(f, sekel)


def save_skeleton_as_dict(data, timens):
    if data.skeletons:
        if save_data:
            with open("./joints/{}_{}.json".format(data.timestamp, timens), "wb") as f:
                f.write(json.dumps(data._asdict()))
                print("saved_skeleton")



def save_depth_as_array(data, array, timestamp, timens):
    if data.skeletons:
        if save_data:
            if not os.path.exists("./pcd/depth"):
                os.makedirs("./pcd/depth")
            with open("./pcd/depth/{}_{}.npy".format(timestamp, timens), "wb") as f:
                np.save(f, array)
            # print(f"{timestamp}_saved")


def save_rgb_as_array(data, array, timestamp, timens):
    if data.skeletons:
        if save_data:
            if not os.path.exists("./pcd/rgb"):
                os.makedirs("./pcd/rgb")
            with open("./pcd/rgb/{}_{}.npy".format(timestamp, timens), "wb") as f:
                np.save(f, array)
            # print(f"{timestamp}_saved")


def Savepcd(data, x, y, z, timestamp, timens):
    """
    Takes XYZ data as Z16, converts it to a PCD and writes it to a file 
    """

    xyz = np.column_stack((x, y, z))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if data.skeletons:
        if save_data:
            o3d.io.write_point_cloud(
                "./pcd/{}_{}.ply".format(timestamp, timens), pcd, write_ascii=True)


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
    fx = 646.358
    fy = 646.358
    ppx = 643.229
    ppy = 360.057

    [height, width] = depth_image.shape

    nx = np.linspace(0, width-1, width)
    ny = np.linspace(0, height-1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - ppx)/fx
    y = (v.flatten() - ppy)/fy

    z = np.transpose(depth_image).flatten()/1000
    x = np.multiply(x, z)
    y = np.multiply(y, z)

    """x = x[np.nonzero(z)]
	y = y[np.nonzero(z)]
	z = z[np.nonzero(z)]"""

    return x, y, z


def init_nuitrack():
    _, _, _, _, _, width, height = get_intrinsics_from_json(1)
    fps = 60 

    nuitrack = py_nuitrack.Nuitrack()
    nuitrack.init()

    # ---enable if you want to use face tracking---
    #nuitrack.set_config_value("Faces.ToUse", "true")
    nuitrack.set_config_value("DepthProvider.Depth2ColorRegistration", "true")
    nuitrack.set_config_value("Realsense2Module.Depth.ProcessWidth", f"{width}")
    nuitrack.set_config_value("Realsense2Module.Depth.ProcessHeight", f"{height}")
    nuitrack.set_config_value("Realsense2Module.Depth.ProcessMaxDepth", "5000")
    nuitrack.set_config_value("Realsense2Module.Depth.Preset", "2")
    nuitrack.set_config_value("Realsense2Module.Depth.FPS", f"{fps}")
    nuitrack.set_config_value("Realsense2Module.RGB.ProcessWidth", f"{width}")
    nuitrack.set_config_value("Realsense2Module.RGB.ProcessHeight", f"{height}")
    nuitrack.set_config_value("Realsense2Module.RGB.FPS", f"{fps}")

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


def main():

    nuitrack = init_nuitrack()

    modes = cycle(["depth", "color"])
    mode = next(modes)
    first_call = True
    win_name = "Skeletal Tracking"
    while 1:

        timens = time.time_ns()

        key = cv2.waitKey(1)

        nuitrack.update()
        data = nuitrack.get_skeleton()
        img_depth = nuitrack.get_depth_data()
        img_color = nuitrack.get_color_data()

        if img_depth.size:

            with mt.ThreadPoolExecutor() as pool:  # Tested---> consistent 1ms improvement

                task1 = pool.submit(save_depth_as_array,
                                    data, img_depth, data.timestamp, timens)

                task2 = pool.submit(save_rgb_as_array, data,
                                    img_color, data.timestamp, timens)

                task3 = pool.submit(save_skeleton_as_array,
                                    data, data.timestamp, timens)

                # Python internal datatype may incur overhead
                futures = [task1, task2, task3]
                done, _ = mt.wait(futures, return_when=mt.ALL_COMPLETED)
                # if done:
                #     pass

            if mode == "depth":
                draw_skeleton(data, img_depth)
            else:
                draw_skeleton(data, img_color)

            # scale depth with depth-min/max and between (0,255)
            cv2.normalize(img_depth, img_depth, 0, 255, cv2.NORM_MINMAX)

            img_depth = np.array(cv2.cvtColor(
                img_depth, cv2.COLOR_GRAY2RGB), dtype=np.uint8)


        if key == 32:
            mode = next(modes)
        if mode == "depth":
            if img_depth.size:
                cv2.imshow(winname=win_name, mat = img_depth)
        if mode == "color":
            if img_color.size:
                cv2.imshow(winname=win_name,mat = img_color)
        if key == 27:
            break
        
        print("{} ms".format((time.time_ns()-timens)/1e6))

    nuitrack.release()


if __name__ == '__main__':
    # import cProfile
    # import pstats

    # with cProfile.Profile() as pr:
    main()

    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # #stats.print_stats()
    # stats.dump_stats(filename=f"./profiling_runs/{file_name}_{time.time_ns()}.prof")
    
