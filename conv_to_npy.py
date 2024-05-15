import numpy as np
import open3d as o3d
import json 
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D 
import plotly.graph_objects as go

path_to_annoation_json = "/home/jkrishna/Downloads/datumaro/annotations/default.json"
path_to_save_txt = "/home/jkrishna/Downloads/datumaro/annotations/"

with open(path_to_annoation_json) as f:
    data = json.load(f)


def load_pcd(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    return points

def voxel_size_from_range(point_range, voxel_size_z):
    # Point cloud range along z-axis / voxel_size is 40
    voxel_size_xy = (point_range[3] - point_range[0]) / (np.ceil((point_range[3] - point_range[0]) / voxel_size_z * 16) / 16)
    voxel_size_xy = (point_range[4] - point_range[1]) / (np.ceil((point_range[4] - point_range[1]) / voxel_size_xy * 16) / 16)
    return voxel_size_xy, voxel_size_z

def scale_annotations(annotations, voxel_size_xy, voxel_size_z, point_range):
    scaled_annotations = []
    for annotation in annotations:
        x, y, z, dx, dy, dz, heading_angle, category_name = annotation
        
        # Scale the coordinates and dimensions
        x = (x - point_range[0]) / voxel_size_xy
        y = (y - point_range[1]) / voxel_size_xy
        z = (z - point_range[2]) / voxel_size_z
        dx /= voxel_size_xy
        dy /= voxel_size_xy
        dz /= voxel_size_z
        
        scaled_annotations.append([x, y, z, dx, dy, dz, heading_angle, category_name])
    
    return scaled_annotations

def main(pcd_path, out_path, point_range, annotations):
    points = load_pcd(pcd_path)
    
    # Ensure the point cloud is within the specified range
    points = points[np.logical_and(np.logical_and(points[:, 0] >= point_range[0], points[:, 0] <= point_range[3]),
                                   np.logical_and(points[:, 1] >= point_range[1], points[:, 1] <= point_range[4]))]
    
    #points[:,3] =0 
    zero_col = np.zeros((points.shape[0],1))
    points = np.append(points, zero_col, axis=1)
    
    # voxel_size_xy, voxel_size_z = voxel_size_from_range(point_range, 0.2)
    
    # Convert the point cloud to voxel representation
    # voxel_coords = ((points[:, :3] - np.array([point_range[0], point_range[1], point_range[2]])) / np.array([voxel_size_xy, voxel_size_xy, voxel_size_z])).astype(np.int32)
    
    # Scale the annotations
    # scaled_annotations = scale_annotations(annotations, voxel_size_xy, voxel_size_z, point_range)
    
    #plot the 3d points and the cuboids
    # fig = go.Figure()
    # data = go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers', marker=dict(size=0.5, color=points[:,2], colorscale='Viridis', opacity=0.8))
    # fig.add_trace(data)
    
    # for annotation in annotations:
    #     x, y, z, dx, dy, dz, heading_angle, category_name = annotation
    #     x = [x-dx/2, x+dx/2, x+dx/2, x-dx/2, x-dx/2, x+dx/2, x+dx/2, x-dx/2]
    #     y = [y-dy/2, y-dy/2, y+dy/2, y+dy/2, y-dy/2, y-dy/2, y+dy/2, y+dy/2]
    #     z = [z, z, z, z, z+dz, z+dz, z+dz, z+dz]
    #     fig.add_trace(go.Mesh3d(x=x, y=y, z=z, showscale=False, flatshading=True, opacity=0.6,
    #     color='#DC143C'))
    
    # fig.show()


    # Save the voxel representation and scaled annotations
    np.save("/home/jkrishna/OpenPCDet/data/custom/points/"+out_path+".npy", points)
    fname = "/home/jkrishna/OpenPCDet/data/custom/labels/"+out_path+".txt" 
    with open(fname, 'w') as f:
        for annotation in annotations:
            f.write(' '.join(map(str, annotation)) + '\n')


if __name__ == '__main__':
    import os 
    import glob 

    labels = ["fence","bollard"]
    pcd_path ="/home/jkrishna/portomated/portomated-ros/src/stack_path_finder/pcd_files/"
    point_range = [-32, -32, -3, 32, 32, 40]  
    i = 0
    for item in data['items']:
        fname = item['id']
        annotation_ = []
        for annotation in item['annotations']:
            x = annotation['position'][0]
            y = annotation['position'][1]
            z = annotation['position'][2]
            dx = annotation['scale'][0]
            dy = annotation['scale'][1]
            dz = annotation['scale'][2]
            rz = annotation['rotation'][2]
            label = labels[int(annotation['label_id'])]
            annotation_.append([x, y, z, dx, dy, dz, rz, label])
        pcd_file = pcd_path+fname+'.pcd'
        newname = str(i).zfill(6)
        main(pcd_file, newname, point_range, annotation_)
        i+=1
    print("Done")