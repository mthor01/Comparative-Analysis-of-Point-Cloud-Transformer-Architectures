"""
This script generates two interactive windows that show the ground truth labels
and the predicted labels for a given scene and model.
""" 

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def label_to_colors(labels):
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab20")
    color_map = {label: cmap(i % 20)[:3] for i, label in enumerate(unique_labels)}
    return np.array([color_map[label] for label in labels]), color_map


def load_npy_point_cloud_with_labels(pts_file, label_file):
    pts = np.load(pts_file)
    labels = np.load(label_file).flatten()
    colors, color_map = label_to_colors(labels)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd, color_map

def show_legend(color_map, class_names):
    plt.ion()
    fig, ax = plt.subplots(figsize=(4, 6))
    legend_elements = [
        Patch(facecolor=color_map[label], edgecolor='black', label=class_names[label])
        for label in sorted(color_map)
    ]
    ax.legend(handles=legend_elements, loc='center')
    ax.axis('off')
    fig.canvas.manager.set_window_title("Class Legend")

    # Move the legend window (only works with some backends, like TkAgg)
    try:
        fig.canvas.manager.window.wm_geometry("+1750+50")  # 800 + 800 + 40 margin
    except Exception as e:
        print("Couldn't reposition matplotlib window:", e)

    plt.show()



def sync_cameras(vis1, vis2):
    ctr1 = vis1.get_view_control()
    ctr2 = vis2.get_view_control()
    params = ctr1.convert_to_pinhole_camera_parameters()
    ctr2.convert_from_pinhole_camera_parameters(params)


def get_cam_params(vis):
    return vis.get_view_control().convert_to_pinhole_camera_parameters()


def cam_params_changed(cam_a, cam_b):
    return not (
        np.allclose(cam_a.intrinsic.intrinsic_matrix, cam_b.intrinsic.intrinsic_matrix) and
        np.allclose(cam_a.extrinsic, cam_b.extrinsic)
    )





# Load point cloud and color map
pcd1, color_map = load_npy_point_cloud_with_labels("Results/point_clouds/coord.npy", "Results/point_clouds/segment.npy")
pcd2, _ = load_npy_point_cloud_with_labels("Results/point_clouds/coord.npy", "Results/point_clouds/segment.npy")

class_names = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
               'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter')





# Show legend window
show_legend(color_map, class_names)

# Create visualizer windows
vis1 = o3d.visualization.Visualizer()
vis2 = o3d.visualization.Visualizer()
vis1.create_window(window_name='Ground Truth Segmentation', width=800, height=800, left=50, top=50)
vis2.create_window(window_name='Predicted Segmentation', width=800, height=800, left=900, top=50)

vis1.add_geometry(pcd1)
vis2.add_geometry(pcd2)

vis1.poll_events()
vis1.update_renderer()
vis2.poll_events()
vis2.update_renderer()
sync_cameras(vis1, vis2)

vis1.register_animation_callback(lambda vis: sync_cameras(vis1, vis2))
vis2.register_animation_callback(lambda vis: sync_cameras(vis1, vis2))

prev_cam1 = get_cam_params(vis1)
prev_cam2 = get_cam_params(vis2)

while True:
    if not vis1.poll_events() or not vis2.poll_events():
        break

    #vis1.poll_events()
    vis1.update_renderer()
    #vis2.poll_events()
    vis2.update_renderer()

    plt.pause(0.001) 

    cam1 = get_cam_params(vis1)
    cam2 = get_cam_params(vis2)

    if cam_params_changed(cam1, prev_cam1):
        vis2.get_view_control().convert_from_pinhole_camera_parameters(cam1)
        prev_cam1 = cam1
        prev_cam2 = cam1
    elif cam_params_changed(cam2, prev_cam2):
        vis1.get_view_control().convert_from_pinhole_camera_parameters(cam2)
        prev_cam1 = cam2
        prev_cam2 = cam2
