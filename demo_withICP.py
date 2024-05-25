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
from tracker import ICPTracker

if __name__ == "__main__":
    print("Initializing settings...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    # standard configs
    parser.add_argument('--config', type=str, default="configs/data2.yaml", help='Path to config file.')
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
                                      fuse_feature=True)
    icp_tracker = ICPTracker(device)

    # Loop through RGB-D images and fuse them together
    poses = list()
    curr_pose, depth1, color1 = None, None, None
    t0_elapse = time.time()
    for i in tqdm.trange(n_imgs, desc="Fusing frame~"):
        # print("Fusing frame %d/%d" % (i + 1, n_imgs))

        color0 = cv2.cvtColor(cv2.imread(args.rgb_data_root + "%04d-color.jpg" % (i)), cv2.COLOR_BGR2RGB)
        depth0 = cv2.imread(args.depth_data_root + "%04d-depth.png" % (i), -1).astype(float)

        height, width = depth0.shape[:2]
        depth0 /= args.deepth_scale
        depth0[depth0 == 65.535] = 0

        if i == 0:  # initialize pose, need cpu
            curr_pose = utils.initialize_pose(color0, depth0.astype(np.float32), cam_intr_cpu, vol_bounds,
                                              visualize=True)

        else:  # tracking, need gpu
            depth1, color1, vertex01, normal1, mask1 = tsdf_vol.render_model(curr_pose, cam_intr_gpu, height, width,
                                                                             near=0.1, far=5.0, n_samples=args.n_samples)

            depth1_visualized = (depth1.cpu().numpy() - 0.1) / (5.0 - 0.1) * 255
            depth1_visualized = depth1_visualized.astype(np.uint8)

            depth0_visualized = (depth0 / depth0.max() * 255).astype('uint8')

            # 将深度图像显示为灰度图像
            cv2.imshow('Depth0 Visualization', depth0_visualized)
            cv2.waitKey(1)
            cv2.imshow('Depth1 Visualization', depth1_visualized)
            cv2.waitKey(1)

            depth0 = torch.from_numpy(depth0).float().to(device)
            T10 = icp_tracker(depth0, depth1, cam_intr_gpu)  # transform from 0 to 1
            curr_pose = curr_pose @ T10

        poses.append(curr_pose)

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
