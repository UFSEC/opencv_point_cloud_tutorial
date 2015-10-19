import numpy as np
import cv2

image_path = '../images/katowice.jpg'
window_title = 'awesome image'

image = cv2.imread(image_path)
cv2.imshow(window_title, image)

cv2.waitKey(0)
cv2.destroyAllWindows()
