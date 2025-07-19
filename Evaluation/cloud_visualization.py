import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def get_colors_from_labels(labels, colormap="tab20"):
    cmap = plt.get_cmap(colormap)
    num_classes = labels.max() + 1
    colors = cmap(labels % num_classes)[:, :3]  # drop alpha
    return colors

def create_colored_pointcloud(points, labels):
    colors = get_colors_from_labels(labels)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# --- Load data ---
data = np.load("Results/point_clouds/coord.npy")  # shape: (N, F)
points = data[:, :3]  # XYZ

# Load labels from separate files
gt_labels = np.load("Results/point_clouds/segment.npy")
pred_labels = np.load("Results/point_clouds/segment.npy")

gt_labels = gt_labels.flatten()
pred_labels = pred_labels.flatten()


print("Points shape:", points.shape)        # should be (N, 3)
print("GT labels shape:", gt_labels.shape)  # should be (N,)
print("Pred labels shape:", pred_labels.shape)  # should be (N,)


# --- Create colored point clouds ---
gt_pcd = create_colored_pointcloud(points, gt_labels)
pred_pcd = create_colored_pointcloud(points, pred_labels)

# --- Save as PLY ---
o3d.io.write_point_cloud("gt_colored.ply", gt_pcd)
o3d.io.write_point_cloud("pred_colored.ply", pred_pcd)

print("Saved 'gt_colored.ply' and 'pred_colored.ply' — open both in CloudCompare.")
