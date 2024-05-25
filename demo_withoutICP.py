"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time
import cv2
import numpy as np
import tqdm
# from old import fusion
import fusion2 as fusion
import torch
import utils

if __name__ == "__main__":
    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    print("Estimating voxel volume bounds...")
    n_imgs = 1000
    cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
    vol_bnds = np.zeros((3, 2))
    for i in range(n_imgs):
        # Read depth image and camera pose
        depth_im = cv2.imread("data/frame-%06d.depth.png" % (i), -1).astype(float)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
        cam_pose = np.loadtxt("data/frame-%06d.pose.txt" % (i))  # 4x4 rigid transformation matrix

        # Compute camera view frustum and extend convex hull
        view_frust_pts = utils.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    # ======================================================================================================== #

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    voxel_size = 0.05
    vol_dims = (vol_bnds[:, 1] - vol_bnds[:, 0]) // voxel_size + 1
    vol_origin = vol_bnds[:, 0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolumeTorch(vol_dims, vol_origin, voxel_size, device, margin=3, fuse_color=True,
                                      fuse_feature=True)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for i in tqdm.trange(n_imgs, desc="Fusing frame~"):
        # print("Fusing frame %d/%d" % (i + 1, n_imgs))

        # Read RGB-D image and camera pose
        color_image = cv2.cvtColor(cv2.imread("data/frame-%06d.color.jpg" % (i)), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread("data/frame-%06d.depth.png" % (i), -1).astype(float)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = np.loadtxt("data/frame-%06d.pose.txt" % (i))

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(depth_im, cam_intr, cam_pose, obs_weight=1., color_img=color_image)

    fps = n_imgs / (time.time() - t0_elapse)
    print(time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    print("Saving features to features.ply...")
    tsdf_vol.save_features("result/features.pth")

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    utils.meshwrite("result/mesh.ply", verts, faces, norms, colors)
