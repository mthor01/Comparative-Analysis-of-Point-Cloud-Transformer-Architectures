"""
This script generates two interactive windows that show the ground truth labels
and the predicted labels for a given scene and model. 
It only shows points that were labeled incorrectly.
It also shows the full ground truth labeled point cloud as reference.
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
    fig, ax = plt.subplots()
    fig.set_size_inches(300 / fig.get_dpi(), 300 / fig.get_dpi())

    legend_elements = [
        Patch(facecolor=color_map[label], edgecolor='black', label=class_names[label])
        for label in sorted(color_map)
    ]
    ax.legend(handles=legend_elements, loc='center')
    ax.axis('off')
    fig.canvas.manager.set_window_title("Class Legend")

    try:
        fig.canvas.manager.window.wm_geometry("+50+700") 
    except Exception as e:
        print("Couldn't reposition matplotlib window:", e)

    plt.show()



def sync_cameras(*visualizers):
    if len(visualizers) < 2:
        return
    ctr_main = visualizers[0].get_view_control()
    main_params = ctr_main.convert_to_pinhole_camera_parameters()
    for vis in visualizers[1:]:
        vis.get_view_control().convert_from_pinhole_camera_parameters(main_params)



def get_cam_params(vis):
    return vis.get_view_control().convert_to_pinhole_camera_parameters()


def cam_params_changed(cam_a, cam_b):
    return not (
        np.allclose(cam_a.intrinsic.intrinsic_matrix, cam_b.intrinsic.intrinsic_matrix) and
        np.allclose(cam_a.extrinsic, cam_b.extrinsic)
    )


# Convert to Open3D point clouds
def create_pcd(points, labels):
    colors, _ = label_to_colors(labels)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# Load point cloud and color map
pts = np.load("Results/point_clouds/coord.npy")
gt_labels = np.load("Results/point_clouds/segment.npy").flatten()
pred_labels = np.load("Results/point_clouds/Area_5-conferenceRoom_1_pred.npy").flatten()

# Find indices where labels differ
diff_mask = gt_labels != pred_labels
pts_diff = pts[diff_mask]
gt_diff_labels = gt_labels[diff_mask]
pred_diff_labels = pred_labels[diff_mask]

class_names = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
               'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter')



pcd1 = create_pcd(pts_diff, gt_diff_labels)
pcd2 = create_pcd(pts_diff, pred_diff_labels)
pcd3 = create_pcd(pts, gt_labels)


# Update color map for legend
combined_labels = np.concatenate([gt_labels, pred_labels])
_, color_map = label_to_colors(combined_labels)

# Show legend window
show_legend(color_map, class_names)


# Create visualizer windows
vis1 = o3d.visualization.Visualizer()
vis2 = o3d.visualization.Visualizer()
vis3 = o3d.visualization.Visualizer()
vis1.create_window(window_name='Ground Truth Segmentation', width=600, height=600, left=30, top=50)
vis2.create_window(window_name='Predicted Segmentation', width=600, height=600, left=660, top=50)
vis3.create_window(window_name='Full Ground Truth Scene', width=600, height=600, left=1290, top=50)

vis1.add_geometry(pcd1)
vis2.add_geometry(pcd2)
vis3.add_geometry(pcd3)

vis1.poll_events()
vis1.update_renderer()
vis2.poll_events()
vis2.update_renderer()
vis3.poll_events()
vis3.update_renderer()
sync_cameras(vis1, vis2, vis3)

vis1.register_animation_callback(lambda vis: sync_cameras(vis1, vis2, vis3))
vis2.register_animation_callback(lambda vis: sync_cameras(vis1, vis2, vis3))
vis3.register_animation_callback(lambda vis: sync_cameras(vis1, vis2, vis3))

prev_cam1 = get_cam_params(vis1)
prev_cam2 = get_cam_params(vis2)
prev_cam3 = get_cam_params(vis3)

while True:
    if not vis1.poll_events() or not vis2.poll_events() or not vis3.poll_events():
        break

    vis1.update_renderer()
    vis2.update_renderer()
    vis3.update_renderer()

    plt.pause(0.001)

    cam1 = get_cam_params(vis1)
    cam2 = get_cam_params(vis2)
    cam3 = get_cam_params(vis3)

    if cam_params_changed(cam1, prev_cam1):
        vis2.get_view_control().convert_from_pinhole_camera_parameters(cam1)
        vis3.get_view_control().convert_from_pinhole_camera_parameters(cam1)
        prev_cam1 = cam1
        prev_cam2 = cam1
    elif cam_params_changed(cam2, prev_cam2):
        vis1.get_view_control().convert_from_pinhole_camera_parameters(cam2)
        vis3.get_view_control().convert_from_pinhole_camera_parameters(cam2)
        prev_cam1 = cam2
        prev_cam2 = cam2
    elif cam_params_changed(cam3, prev_cam2):
        vis1.get_view_control().convert_from_pinhole_camera_parameters(cam3)
        vis2.get_view_control().convert_from_pinhole_camera_parameters(cam3)
        prev_cam1 = cam3
        prev_cam2 = cam3

