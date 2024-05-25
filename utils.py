import numpy as np
import open3d as o3d
import scipy.linalg as la
import copy
import torch
from addict import Dict
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"


class ForceKeyErrorDict(Dict):
    def __missing__(self, key):
        raise KeyError(key)


def load_yaml(path):
    with open(path, encoding='utf8') as yaml_file:
        config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        config = ForceKeyErrorDict(**config_dict)

    return config


def load_config(args):
    config_dict = load_yaml(args.config)
    # merge args and config
    other_dict = vars(args)
    config_dict.update(other_dict)
    return config_dict


def initialize_pose(color_im, depth_im, cam_intr, vol_bounds, visualize=False):
    pcd = create_pcd(depth_im, cam_intr, color_im, depth_trunc=3)

    plane_frame, inlier_ratio = plane_detection_o3d(pcd,
                                                    max_iterations=1000,
                                                    inlier_thresh=0.005,
                                                    visualize=False)
    cam_pose = la.inv(plane_frame)
    transformed_pcd = copy.deepcopy(pcd).transform(la.inv(plane_frame))
    transformed_pts = np.array(transformed_pcd.points)
    transformed_pts = transformed_pts[transformed_pts[:, 2] > -0.05]

    vol_bnds = np.zeros((3, 2), dtype=np.float32)
    vol_bnds[:, 0] = transformed_pts.min(0)
    vol_bnds[:, 1] = transformed_pts.max(0)
    vol_bnds[0] += vol_bounds[:2]
    vol_bnds[1] += vol_bounds[2:4]
    vol_bnds[2] = vol_bounds[4:]

    if visualize:
        vol_box = o3d.geometry.OrientedBoundingBox()
        vol_box.center = vol_bnds.mean(1)
        vol_box.extent = vol_bnds[:, 1] - vol_bnds[:, 0]
        vol_box.color = [1, 0, 0]
        cam_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)
        world_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)
        world_frame.transform(cam_pose)
        o3d.visualization.draw_geometries([vol_box, transformed_pcd, world_frame, cam_frame])

    return torch.tensor(cam_pose, dtype=torch.float32).to(device)


def plane_detection_o3d(pcd: o3d.geometry.PointCloud,
                        inlier_thresh: float,
                        max_iterations: int = 2000,
                        visualize: bool = False,
                        in_cam_frame: bool = True):
    # http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Plane-segmentation
    plane_model, inliers = pcd.segment_plane(distance_threshold=inlier_thresh,
                                             ransac_n=3,
                                             num_iterations=max_iterations)
    [a, b, c, d] = plane_model  # ax + by + cz + d = 0
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    max_inlier_ratio = len(inliers) / len(np.asarray(pcd.points))

    # sample the inlier point that is closest to the camera origin as the world origin
    inlier_pts = np.asarray(inlier_cloud.points)
    squared_distances = np.sum(inlier_pts ** 2, axis=1)
    closest_index = np.argmin(squared_distances)
    x, y, z = inlier_pts[closest_index]
    origin = np.array([x, y, (-d - a * x - b * y) / (c + 1e-12)])
    plane_normal = np.array([a, b, c])
    plane_normal /= np.linalg.norm(plane_normal)

    if in_cam_frame:
        if plane_normal @ origin > 0:
            plane_normal *= -1
    elif plane_normal[2] < 0:
        plane_normal *= -1

    # randomly sample x_dir and y_dir given plane normal as z_dir
    x_dir = np.array([-plane_normal[2], 0, plane_normal[0]])
    x_dir /= la.norm(x_dir)
    y_dir = np.cross(plane_normal, x_dir)
    plane_frame = np.eye(4)
    plane_frame[:3, 0] = x_dir
    plane_frame[:3, 1] = y_dir
    plane_frame[:3, 2] = plane_normal
    plane_frame[:3, 3] = origin

    if visualize:
        plane_frame_vis = generate_coordinate_frame(plane_frame, scale=0.05)
        cam_frame_vis = generate_coordinate_frame(np.eye(4), scale=0.05)
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, plane_frame_vis, cam_frame_vis])

    return plane_frame, max_inlier_ratio


def create_pcd(depth_im: np.ndarray,
               cam_intr: np.ndarray,
               color_im: np.ndarray = None,
               depth_scale: float = 1,
               depth_trunc: float = 1.5,
               cam_extr: np.ndarray = np.eye(4)):
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_o3d.intrinsic_matrix = cam_intr
    depth_im_o3d = o3d.geometry.Image(depth_im)
    if color_im is not None:
        color_im_o3d = o3d.geometry.Image(color_im)
        rgbd = o3d.geometry.RGBDImage().create_from_color_and_depth(color_im_o3d, depth_im_o3d,
                                                                    depth_scale=depth_scale,
                                                                    depth_trunc=depth_trunc,
                                                                    convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud().create_from_rgbd_image(rgbd, intrinsic_o3d, extrinsic=cam_extr)
    else:
        pcd = o3d.geometry.PointCloud().create_from_depth_image(depth_im_o3d, intrinsic_o3d,
                                                                extrinsic=cam_extr,
                                                                depth_scale=depth_scale,
                                                                depth_trunc=depth_trunc)
    return pcd


def get_volume_setting(vol_bounds, voxel_size):
    vol_bnds = np.array(vol_bounds).reshape(3, 2)
    vol_dims = (vol_bnds[:, 1] - vol_bnds[:, 0]) // voxel_size + 1
    vol_origin = vol_bnds[:, 0]
    return vol_dims, vol_origin


def save_volume_settings(voxel_size, vol_bounds, vol_dims, vol_origin, filename):
    vol_dims_int = [int(dim) for dim in vol_dims]
    vol_origin_formatted = [round(coord, 1) for coord in vol_origin]
    with open(filename, 'w') as f:
        f.write(f"voxel_size: {voxel_size}\n")
        f.write(f"vol_bounds: {vol_bounds}\n")
        f.write(f"vol_dims: {vol_dims_int}\n")
        f.write(f"vol_origin: {vol_origin_formatted}\n")


def load_volume_settings(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 解析每一行并提取相应的值
    voxel_size = float(lines[0].split(": ")[1])
    vol_bounds = eval(lines[1].split(": ")[1])
    vol_dims = eval(lines[2].split(": ")[1])
    vol_origin = eval(lines[3].split(": ")[1])

    return voxel_size, vol_bounds, vol_dims, vol_origin


def index_to_position(index, voxel_size, vol_origin):
    # 将索引乘以体素尺寸，得到相对位置
    relative_position = index * voxel_size
    # 将相对位置加上体素原点，得到真实位置
    real_position = relative_position + vol_origin
    return real_position


def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud.
    """
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
    """Get corners of 3D camera view frustum of depth image
    """
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array([
        (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[0, 0],
        (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[1, 1],
        np.array([0, max_depth, max_depth, max_depth, max_depth])
    ])
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts


def meshwrite(filename, verts, faces, norms, colors):
    """Save a 3D mesh to a polygon .ply file.
    """
    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
            verts[i, 0], verts[i, 1], verts[i, 2],
            norms[i, 0], norms[i, 1], norms[i, 2],
            colors[i, 0], colors[i, 1], colors[i, 2],
        ))

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()


def pcwrite(filename, xyzrgb):
    """Save a point cloud to a polygon .ply file.
    """
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n" % (
            xyz[i, 0], xyz[i, 1], xyz[i, 2],
            rgb[i, 0], rgb[i, 1], rgb[i, 2],
        ))
