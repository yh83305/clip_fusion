"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time
import cv2
import numpy as np
import tqdm
import argparse
# from old import fusion
import fusion2 as fusion
import torch
import utils
import json
import liblzfse
from PIL import Image
from tracker import ICPTracker
from quaternion import as_rotation_matrix, quaternion


def read_metadata(filepath):
    with open(filepath, "r") as f:
        metadata_dict = json.load(f)
    poses = np.array(metadata_dict["poses"])
    init_pose = np.array(metadata_dict["initPose"])
    print(poses, init_pose)
    return poses, init_pose


def load_image(filepath):
    with open(filepath, "rb") as image_file:
        return np.asarray(Image.open(image_file))


def load_depth(filepath):
    with open(filepath, "rb") as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img: np.ndarray = np.frombuffer(decompressed_bytes, dtype=np.float32)

    if depth_img.shape[0] == 960 * 720:
        depth_img = depth_img.reshape((960, 720))  # For a FaceID camera 3D Video
    else:
        depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video
    return depth_img


if __name__ == "__main__":
    print("Initializing settings...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    # standard configs
    parser.add_argument('--config', type=str, default="configs/data3.yaml", help='Path to config file.')
    args = utils.load_config(parser.parse_args())

    n_imgs = args.n_imgs
    cam_intr_cpu = np.loadtxt(args.intrinsics_data_root + "camera-intrinsics.txt", delimiter=' ')
    cam_intr_gpu = torch.from_numpy(cam_intr_cpu).float().to(device)

    voxel_size = args.voxel_size
    vol_bounds = args.vol_bounds
    vol_dims, vol_origin = utils.get_volume_setting(vol_bounds, voxel_size)
    utils.save_volume_settings(voxel_size, vol_bounds, vol_dims, vol_origin, "result/vol_config")

    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolumeTorch(vol_dims, vol_origin, voxel_size, device, margin=3, fuse_color=True,
                                      fuse_feature=False)
    icp_tracker = ICPTracker(device)

    # Loop through RGB-D images and fuse them together
    curr_pose, depth1, color1 = None, None, None
    t0_elapse = time.time()

    poses, init_pose = read_metadata("./data3/metadata")
    for i in tqdm.trange(n_imgs, desc="Fusing frame~"):
        # print("Fusing frame %d/%d" % (i + 1, n_imgs))

        color0 = load_image(args.rgb_data_root + "%d.jpg" % (i))
        _depth0 = load_depth(args.depth_data_root + "%d.depth" % (i))

        # Upscale depth image.
        pil_img = Image.fromarray(_depth0)
        depth0_cp = pil_img.resize((color0.shape[1], color0.shape[0]))
        depth0_cp = np.asarray(depth0_cp)
        depth0 = depth0_cp.copy()
        height, width = depth0.shape[:2]

        # depth0 /= 1000.
        # depth0[depth0 == 65.535] = 0
        # 归一化深度图像到 [0, 255] 范围
        depth0_normalized = cv2.normalize(depth0, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow('Depth0 Image', depth0_normalized)
        cv2.waitKey(1)

        curr_pose = np.eye(4)

        if i == 0:  # initialize pose, need cpu
            qx, qy, qz, qw, px, py, pz = init_pose
        else:  # tracking, need gpu
            qx, qy, qz, qw, px, py, pz = poses[i-1]

        curr_pose[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
        curr_pose[:3, -1] = [px, py, pz]

        tsdf_vol.integrate(depth0,
                           cam_intr_cpu,
                           curr_pose,
                           obs_weight=1.,
                           color_img=color0
                           )

    fps = n_imgs / (time.time() - t0_elapse)
    print(time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    print("Saving features to features.ply...")
    tsdf_vol.save_features("result/features.pth")

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    utils.meshwrite("result/mesh.ply", verts, faces, norms, colors)
