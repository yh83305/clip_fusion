import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Configure depth and rgb and infrared streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Align objects
# align_to = rs.stream.color
align_to = rs.stream.depth
align = rs.align(align_to)

# Start streaming
profile = pipeline.start(config)

# Depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth scale is: ", depth_scale)

try:
    while True:
        # Frames: depth and rgb and infrared
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Convert images to numpy arrays
        depth_data = np.asanyarray(depth_frame.get_data(), dtype="float16")
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        cv2.imshow('color_image', color_image)
        cv2.waitKey(1)
        cv2.imshow('depth_image', depth_image)
        cv2.waitKey(1)

# Save data
# np.savetxt('./realsense/depth_data.txt', depth_data)
# depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
# cv2.imwrite('./realsense/depth_image.jpg', depth_colormap)
# cv2.imwrite('./realsense/color_image.jpg', color_image)
# cv2.imwrite('./realsense/infrared_image.jpg', infrared_image)

finally:
    # Stop streaming
    pipeline.stop()
