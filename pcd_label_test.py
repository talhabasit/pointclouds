import open3d.visualization as viz 
import open3d.tools as tools 
import open3d as o3d

pcd = o3d.io.read_point_cloud("C:/Users/talha/OneDrive/Desktop/Nuitrack/pcd/1662303325355135100.ply")


v=[]
with open("C:/Users/talha/OneDrive/Desktop/Nuitrack/joints/1662303338496000_1662303325359238100.txt","r") as f:
    for line in f:
        v.append(line.split())
        
# with open("C:/Users/talha/OneDrive/Desktop/Nuitrack/joints/1661859883122688_1661859896767854300.txt") as f:
#     w, h = [float(x) for x in next(f).split()] # read first line
#     array = []
#     for line in f: # read rest of lines
#         array.append([int(x) for x in line.split()])
        
o3d.visualization.draw_geometries([pcd])

x=0