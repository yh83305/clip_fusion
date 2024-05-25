# import torch
#
# N = torch.tensor([0, 0, 0, 0, 0])
# mask0 = torch.tensor([True, True, True, True, False])
# mask1 = torch.tensor([True, True, True, False])
# mask2 = torch.tensor([True, True, False])
# val = torch.tensor([1, 2])
#
# change1 = torch.zeros_like(N[mask0][mask1])
# change2 =torch.zeros_like(N[mask0])
# change1[mask2] = val
# change2[mask1] = change1
# N[mask0] = change2
# print(N)

import cv2
depth0 = cv2.imread("/home/ubuntu/Desktop/tsdf-fusion-python/data2/DESK/depth/0600-depth.png")
print(depth0)
depth_gray = (depth0 / depth0.max() * 255).astype('uint8')

cv2.imshow('Depth0 Visualization', depth_gray)
cv2.waitKey(0)
