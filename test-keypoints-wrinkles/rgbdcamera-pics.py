import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

# Create output folders if they don't exist
os.makedirs("images", exist_ok=True)
os.makedirs("depth", exist_ok=True)
os.makedirs("depth_normalized", exist_ok=True)

# Configure streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)   # RGB
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)   # Depth

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (for visualization only)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Show images
        cv2.imshow('RGB', color_image)
        cv2.imshow('Depth Colormap', depth_colormap)

        key = cv2.waitKey(1)

        # Save images when 's' is pressed
        if key & 0xFF == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # Save RGB image (8-bit)
            cv2.imwrite(os.path.join("images", f"rgb_{timestamp}.png"), color_image)

            # Save depth image (16-bit, real distances)
            cv2.imwrite(os.path.join("depth", f"depth_{timestamp}.png"), depth_image)

            # Save depth colormap (for visualization)
            cv2.imwrite(os.path.join("depth_normalized", f"depth_colormap_{timestamp}.png"), depth_colormap)

            print(f"✅ Saved images/rgb_{timestamp}.png, depth/depth_{timestamp}.png, depth_normalized/depth_colormap_{timestamp}.png")

        # Quit when 'q' is pressed
        if key & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
