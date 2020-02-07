
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

#open3d==0.9.0
#scipy==1.4.1

if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud("G:\\Datasets\\3D\\bu3dfeply\\PLY\\F0001_AN01WH_F3D.ply")
    print(pcd)
    print(np.asarray(pcd.points))
    #o3d.visualization.draw_geometries([pcd])

    rot = R.from_rotvec(np.pi/1 * np.array([0, 0, 1]))
    print(dir(rot))
    rot_pcd = pcd.rotate(rot.as_matrix())

    print("Rotated pointcloud")
    o3d.visualization.draw_geometries([rot_pcd])
