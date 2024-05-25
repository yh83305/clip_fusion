import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import torch
import clip
import tqdm
import utils
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
# 设置 NumPy 打印选项以显示完整的数组
torch.set_printoptions(threshold=np.inf)


def search_features(text, image_features):
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text).squeeze()
        text_features = text_features.to(image_features.dtype)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        print("text_features.shape", text_features.shape)
        print("image_features.shape", image_features.shape)
        sim_array = (100.0 * image_features @ text_features).detach().cpu().numpy()
        sim_array = np.nan_to_num(sim_array, nan=0)
        print("sim_array.shape", sim_array.shape)
    visualize_similarity(sim_array)


def visualize_similarity(sim_array):
    # max_z_indices = np.argmax(sim_array, axis=2)
    # max_z_values = np.zeros((sim_array.shape[0], sim_array.shape[1]))
    # # 遍历每个位置，提取 z 方向上取值最大的点的值
    # for i in range(sim_array.shape[0]):
    #     for j in range(sim_array.shape[1]):
    #         max_z_values[i, j] = sim_array[i, j, max_z_indices[i, j]]
    # max_z_values[max_z_values == 0] = 20
    # plt.imshow(max_z_values, cmap='summer', interpolation='nearest')
    # plt.colorbar()  # 添加颜色条
    # plt.title('Heatmap of max_z_values')  # 设置标题
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
    # 获取非零元素的索引
    non_zero_indices = np.nonzero(sim_array)

    # 将非零索引转换为坐标
    points = np.column_stack(non_zero_indices)
    print("points shape:", points.shape)

    voxel_size, vol_bounds, vol_dims, vol_origin = utils.load_volume_settings("result/vol_config")
    print("voxel_size:", voxel_size)
    print("vol_bounds:", vol_bounds)
    print("vol_dims:", vol_dims)
    print("vol_origin:", vol_origin)

    real_points = utils.index_to_position(points, voxel_size, vol_origin)

    # 初始化颜色数组
    colors = np.zeros((len(non_zero_indices[0]), 3))
    print("colors shape:", colors.shape)

    # 根据索引获取对应的值作为颜色
    for idx, (i, j, k) in enumerate(zip(*non_zero_indices)):
        colors[idx] = [(sim_array[i, j, k] - 20) / 10, (30 - sim_array[i, j, k]) / 10,
                       (30 - sim_array[i, j, k]) / 10]

    for thresh in [28, 26, 24, 22, 20]:
        # 创建与points相同形状的布尔数组
        mask = sim_array > thresh
        # mask = select_top_percent(sim_array, percent=10)

        # 将颜色数组中大于30的部分设为红色
        # colors[mask[non_zero_indices]] = [1, 0, 0]
        t_points = real_points[mask[non_zero_indices]]

        if len(t_points) == 0:
            print("no target")
            continue
        else:
            print("t_points shape:", t_points.shape)

            dbscan_mask = dbscan_cluster(t_points, 3 * voxel_size, 5)

            length = len(dbscan_mask)
            print(length)
            if length == 0:
                continue
            # for i in range(length):
            #     change = colors[mask[non_zero_indices]]
            #     change[dbscan_mask[i]] = [1 - i / length, 1 - i / length, i / length]
            #     colors[mask[non_zero_indices]] = change

            change = colors[mask[non_zero_indices]]
            change[dbscan_mask[0]] = [1, 1, 0]
            colors[mask[non_zero_indices]] = change

            cluster_points = t_points[dbscan_mask[0]]

            center = np.mean(cluster_points, axis=0)
            print(center)

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(cluster_points)

            # 生成包围框
            aabb = cloud.get_axis_aligned_bounding_box()

            # 创建 OrientedBoundingBox 对象
            obb = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(aabb)
            line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
            line_set.paint_uniform_color([0, 0, 0])

            # 创建标记球体
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            marker.compute_vertex_normals()
            # 将标记球体移动到中心点的位置
            marker.translate(center)
            marker.paint_uniform_color([1, 1, 0])

            # 创建点云对象
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(real_points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)

            # 创建体素网格
            voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=voxel_size)

            # 创建渲染窗口并添加渲染控制器
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(window_name='Open3D', width=1920, height=1080, visible=True)
            vis.get_render_option().point_size = 3  # 设置点的大小
            vis.get_render_option().point_show_normal = False  # 不显示点的法线

            world_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)
            world_frame.translate([0, 0, 0])

            # 添加点云、中心点和标记球体
            vis.add_geometry(voxel)
            vis.add_geometry(marker)
            vis.add_geometry(world_frame)
            vis.add_geometry(line_set)

            # 运行渲染
            vis.run()
            vis.destroy_window()
            return


def dbscan_cluster(t_points, eps=0.06, min_samples=6):
    """
    参数：
    t_points : numpy数组
        输入的点集，形状为 (n, 3)。
    eps : float, 可选，默认为0.5
        DBSCAN算法中的ε参数，用于确定邻域的大小。
    min_samples : int, 可选，默认为5
        DBSCAN算法中的最小样本数参数。
    """
    # 使用DBSCAN算法对点进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(t_points)

    # 获取聚类的唯一标签
    unique_labels = np.unique(clusters)

    # 初始化掩码列表
    masks = []

    # 根据不同的类别生成掩码
    for label in unique_labels:
        if label != -1:
            # 生成对应类别的掩码
            mask = clusters == label
            masks.append(mask)

    return masks


def select_top_percent(sim_array, percent=10):
    """
    参数：
    sim_array : numpy数组
        输入的相似性数组。
    percent : int, 可选，默认为10
        要选择的百分比。
    """
    # 将相似性数组展平并排序
    flattened_array = np.sort(sim_array.flatten())

    # 计算前百分之几的索引
    top_percent_index = int(len(flattened_array) * (percent / 100))

    # 获取阈值
    threshold_value = flattened_array[-top_percent_index]

    # 生成掩码
    mask = sim_array > threshold_value

    return mask


if __name__ == "__main__":
    # visualization of point clouds.

    real_pcd = o3d.io.read_point_cloud('result/mesh.ply')
    o3d.visualization.draw_geometries([real_pcd])

    # 加载保存的数组
    image_features = torch.load('result/features.pth').to(device)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    while 1:
        text = input("press the sentence:")
        search_features(text, image_features)
