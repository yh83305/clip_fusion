import open3d as o3d

if __name__ == "__main__":
    # visualization of point clouds.
    pcd = o3d.io.read_point_cloud('result/mesh.ply')
    o3d.visualization.draw_geometries([pcd])
    # 体素重构
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                voxel_size=0.01)
    o3d.visualization.draw_geometries([voxel_grid])





