import numpy as np
import cv2
from get_cylinder import create_cylinder_two_point
from nuitrack_3D import depth_from_x_y_z_joints
from read_calib_file import get_intrinsics_from_json
import os
import open3d as o3d
import sys
from Visualize_saved_joints import create_pcd_from_img_depth, load_joints_as_pts
from tqdm import tqdm


"""This Script is used to label the pcds with the help of the joints and the cylinders created from the joints"""

file_name = os.path.basename(__file__)


forearm_color = np.array([1, 0, 0])
upperarm_color = np.array([0, 1, 0])

intrinsics, fx, fy, cx, cy, width, height = get_intrinsics_from_json(1)


def save_labeled_pcds(
    csv=False,
    ply=True,
    rgb=False,
):
    """This function is used to save the labeled pcds as ply or csv files
    Args:
        csv (bool, optional): If True, the pcds are saved as csv files. Defaults to False.
        ply (bool, optional): If True, the pcds are saved as ply files. Defaults to True.
        rgb (bool, optional): Include RGB values. Defaults to False.
    """
    for filename in tqdm(os.listdir("./joints")):
        if not ".npy" in filename:
            print("No joints file found")
            sys.exit()

        filename = filename.split("/")[-1].split(".npy")[0]
        p1, p2, p3 = load_joints_as_pts(filename)

        forearm_cyl = create_cylinder_two_point(p1, p2, radius=0.1)
        upperarm_cyl = create_cylinder_two_point(p2, p3, radius=0.1, offset=0.1)

        forearm_cyl.transform(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        upperarm_cyl.transform(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )

        img_load = np.load(f"./pcd/rgb/{filename}.npy")
        depth_load = np.load(f"./pcd/depth/{filename}.npy")
        img_load = cv2.cvtColor(img_load, cv2.COLOR_BGR2RGB)

        pcd = create_pcd_from_img_depth(img_load, depth_load)
        if not rgb:
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(np.array([1, 1, 1]), (np.asarray(pcd.points).shape[0], 1))
            )

        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Uncomment one of the following to downsample the point cloud

        # pcd = pcd.farthest_point_down_sample(4096)

        # pcd = pcd.uniform_down_sample(every_k_points=1000)

        # Get the indices of the points within the bounding box of the cylinders

        forearm_bb = forearm_cyl.get_oriented_bounding_box()
        upperarm_bb = upperarm_cyl.get_oriented_bounding_box()

        list_forearm = forearm_bb.get_point_indices_within_bounding_box(pcd.points)
        list_upperarm = upperarm_bb.get_point_indices_within_bounding_box(pcd.points)

        # Get the intersection of the two lists and remove the points from the forearm list

        # same_elements = np.intersect1d(list_forearm,list_upperarm)#intersection of two lists
        same_elements = list(set(list_forearm).intersection(list_upperarm))

        for x in same_elements:
            list_forearm.remove(x)

        forearm_pcd = pcd.select_by_index(list_forearm)
        upperarm_pcd = pcd.select_by_index(list_upperarm)

        forearm_pcd.paint_uniform_color(forearm_color)
        upperarm_pcd.paint_uniform_color(upperarm_color)

        points = np.asarray(pcd.points).astype(np.float32)
        colors = np.asarray(pcd.colors).astype(np.float32)

        forearm_set = set(list_forearm)
        upperarm_set = set(list_upperarm)

        if csv:
            import pandas as pd

            pcd_df = pd.DataFrame(np.asarray(pcd.points, dtype=np.float32))
            pcd_df.columns = ["x", "y", "z"]
            pcd_df["label"] = "background"
            pcd_df.loc[list_forearm, "label"] = "forearm"
            pcd_df.loc[list_upperarm, "label"] = "upperarm"

            pcd_df.to_csv(f"./labeled/{filename}.csv", index=True)

        if ply:
            with open(f"./labeled/{filename}.ply", "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write("comment Data Format xyz rgb label-[refer key.json]\n")
                f.write("element vertex {}\n".format(pcd.points.__len__()))
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                if rgb:
                    f.write("property float red\n")
                    f.write("property float green\n")
                    f.write("property float blue\n")
                f.write("property char label \n")
                f.write("end_header\n")
                for i in range(pcd.points.__len__()):
                    x, y, z = points[i]
                    r, g, b = colors[i]
                    if i in forearm_set:
                        label = 1
                        forearm_set.remove(i)
                    elif i in upperarm_set:
                        label = 2
                        upperarm_set.remove(i)
                    else:
                        label = 0
                    if rgb:
                        f.write(f"{x} {y} {z} {r} {g} {b} {label}\n")
                    else:
                        f.write(f"{x} {y} {z} {label}\n")

        # print(f"Same points :{same_elements.__len__()}")


if __name__ == "__main__":
    save_labeled_pcds(csv=False, ply=True, rgb=False)
