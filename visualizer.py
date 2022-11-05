

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d
from open3d.web_visualizer import draw
from open3d.visualization import draw_geometries
import natsort as ns


def visualize_data(point_cloud):
    df = pd.DataFrame(
        data={
            "x": point_cloud[:, 0],
            "y": point_cloud[:, 1],
            "z": point_cloud[:, 2],
        }
    )
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection="3d")

    ax.scatter(
        df["x"], df["y"], df["z"]
    )

    plt.show()

def visualize_pcd(points):
    
    colors = np.random.rand(1000, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    draw(pcd)
    # visualizer = JVisualizer()
    # visualizer.add_geometry(pcd)
    # visualizer.show()



npy_dir = "./pcd/"
joints_dir = "./joints/"

def read_data():
    npy_list=[]
    joints_list=[]

    for file in ns.natsorted(os.listdir(npy_dir)):
        filename = os.path.join(npy_dir,file)
        npy_list.append(np.load(filename,"r"))

    npy_list=np.asanyarray(npy_list)    

    for file in ns.natsorted(os.listdir(joints_dir)):
        filename = os.path.join(joints_dir,file)
        joints_list.append(np.loadtxt(filename))
        
    joints_list=np.asanyarray(ns.natsorted(joints_list))
    
    return npy_list,joints_list

def pcd_from_npy(array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd
    

def lines(line_data:np.ndarray):
    #draw a point set
    points = line_data[0,:,:]
    lines = [
        [0, 1],
        [1, 2]
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    #line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

pcd,joints=read_data()
lines = lines(joints)
points=pcd_from_npy(pcd[0,:,:])

draw_geometries([points,lines])


# visualize_pcd(npy_list[1,:,:])

# xyz=np.load("C:/Users/Basit/Desktop/PyNuitrack/Nuitrack/pcd/1665754747371520_1665754732619735700.npy","r")
# print(xyz)


