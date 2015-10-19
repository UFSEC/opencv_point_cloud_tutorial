import numpy as np
import cv2

image_path = '../images/katowice.jpg'
window_title = 'original'
window2_title = 'keypoints'
window3_title = 'rich_keypoints'

image = cv2.imread(image_path)
orb = cv2.ORB_create()

keypoints = orb.detect(image, None)
keypoints, descriptors = orb.compute(image, keypoints)

# Draw the feature points (just locations) onto the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# Draw the feature points with DRAW_RICH_KEYPOINTS flag enabled
image_with_rich_keypoints = cv2.drawKeypoints(image, keypoints, None, None, 4)

cv2.imshow(window_title, image)
cv2.imshow(window2_title, image_with_keypoints)
cv2.imshow(window3_title, image_with_rich_keypoints)

cv2.waitKey(0)
cv2.destroyAllWindows()
