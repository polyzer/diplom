
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import glob
import os
from pathlib import Path
#open3d==0.9.0
#scipy==1.4.1

save_dir = "AUG_PLY"

class Augmenter:
    def __init__(self):
        return      
    def rotate_pcd_by_angle_on_axis(self, pcd, angle, axis = [1,0,0]):
        rot_vec = np.array(axis)
        matrix = R.from_rotvec(angle * rot_vec).as_matrix()
        rot_pcd = pcd.rotate(matrix)
        return rot_pcd

    def save_ply_by_path(self, ply, path):
        o3d.io.write_point_cloud(path, ply)

    def load_ply_by_path(self, path):
        return o3d.io.read_point_cloud(path)

    def get_rotate_range(self, a_1, a_2, steps):
       return np.linspace(a_1, a_2, steps)

    def get_all_ply_files_in_dir_and_subdirs(self, in_dir):
        return Path(in_dir).rglob('*.ply')

if __name__ == "__main__":
    augm = Augmenter()
    all_plys = augm.get_all_ply_files_in_dir_and_subdirs("./PLY")
    rot_range = augm.get_rotate_range(-np.pi/6, np.pi/6, 30)
    new_rot_range = rot_range / np.pi * 180

    x_postfixes = [f"_x_{int(el)}" for el in np.round(new_rot_range)]
    y_postfixes = [f"_y_{int(el)}" for el in np.round(new_rot_range)]
    z_postfixes = [f"_z_{int(el)}" for el in np.round(new_rot_range)]
    x_map = {x_postfixes[i]:  rot_range[i] for i in range(len(rot_range))}
    y_map = {y_postfixes[i]:  rot_range[i] for i in range(len(rot_range))}
    z_map = {z_postfixes[i]:  rot_range[i] for i in range(len(rot_range))}
    
    print(x_map)

    for f_full_path in all_plys:
        ply = augm.load_ply_by_path(str(f_full_path))
        for postf, angle in x_map.items():
            save_name = os.path.basename(f_full_path)[:-4] + postf + ".ply"
            save_path = os.path.join(save_dir, save_name)
            rot_ply = augm.rotate_pcd_by_angle_on_axis(ply, angle, [1,0,0])
            augm.save_ply_by_path(rot_ply, save_path)
            print(f"saved: {save_path}")
        for postf, angle in y_map.items():
            save_name = os.path.basename(f_full_path)[:-4] + postf + ".ply"
            save_path = os.path.join(save_dir, save_name)
            rot_ply = augm.rotate_pcd_by_angle_on_axis(ply, angle, [0,1,0])
            augm.save_ply_by_path(rot_ply, save_path)
            print(f"saved: {save_path}")
        for postf, angle in z_map.items():
            save_name = os.path.basename(f_full_path)[:-4] + postf + ".ply"
            save_path = os.path.join(save_dir, save_name)
            rot_ply = augm.rotate_pcd_by_angle_on_axis(ply, angle, [0,0,1])
            augm.save_ply_by_path(rot_ply, save_path)
            print(f"saved: {save_path}")


    # print(rot_range)
    # print(new_rot_range)
    # print(np.round(new_rot_range))
    # print(x_postfixes)
#     rot = R.from_rotvec(np.pi/12 * np.array([1, 0, 0]))
#    rot_pcd = pcd.rotate(rot.as_matrix())

#     print("Rotated pointcloud")
#     o3d.visualization.draw_geometries([rot_pcd])

#     rot = R.from_rotvec(np.pi/12 * np.array([0, 0, 1]))
#     print(dir(rot))
#     rot_pcd = pcd.rotate(rot.as_matrix())

#     print("Rotated pointcloud")
#     o3d.visualization.draw_geometries([rot_pcd])

#     print("saving")
    #o3d.io.write_point_cloud("./rotated.ply", rot_pcd)