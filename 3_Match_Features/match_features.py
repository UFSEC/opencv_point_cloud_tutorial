import numpy as np
import cv2

image1_path = '../images/living_room_1.jpg'
image2_path = '../images/living_room_2.jpg'
image3_path = '../images/living_room_3.jpg'
window4_title = 'matches 1 -> 2'
window5_title = 'matches 2 -> 3'

cv2.namedWindow(window4_title, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(window5_title, cv2.WINDOW_AUTOSIZE)

# Read in the images
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)
image3 = cv2.imread(image3_path)

# Initialize the feature detector
orb = cv2.ORB_create()

# --- Get the Features and teh Feature Descriptors --- #
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
keypoints3, descriptors3 = orb.detectAndCompute(image3, None)

# Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches12 = bf.match(descriptors1,descriptors2)
matches23 = bf.match(descriptors2,descriptors3)

# Important to sort by how close the feature descriptors are to each other
matches12 = sorted(matches12, key = lambda x:x.distance)
matches23 = sorted(matches23, key = lambda x:x.distance)

# Draw the matches from one image to the next one
image4 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches12[:20], None)
image5 = cv2.drawMatches(image2, keypoints2, image3, keypoints3, matches23[:20], None)

# --- Display the Images --- #
cv2.imshow(window4_title, image4)
cv2.imshow(window5_title, image5)

# Clean up
cv2.waitKey(0)
cv2.destroyAllWindows()
