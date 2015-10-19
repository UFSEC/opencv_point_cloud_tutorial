import numpy as np
import cv2

image1_path = '../images/color.png'
image2_path = '../images/depth.png'

window1_title = 'color'
window2_title = 'depth'
window3_title = 'point cloud'

cv2.namedWindow(window1_title, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(window2_title, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(window3_title, cv2.WINDOW_AUTOSIZE)

# Read in the images
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Create the camera calibration matrix
camera_matrix = np.zeros((3,3))
fx = 532.189488 
fy = 532.189488 
cx = 318.839986 
cy = 244.149197
scale = 10000.0

points = []

for v in range(depth_image.height):
  for u in range(depth_image.width):
    Z = depth_image[v,u] / factor
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    points.append(
    

points = cv2.depthTo3d()

# --- Display the Images --- #
cv2.imshow(window1_title, image1)
cv2.imshow(window2_title, image2)

cv2.moveWindow(window1_title, 0, 0)
cv2.moveWindow(window2_title, 0, 500)

# Clean up
cv2.waitKey(0)
cv2.destroyAllWindows()
