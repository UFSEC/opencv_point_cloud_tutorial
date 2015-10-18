import numpy as np
import cv2

image_path = 'sec_logo.jpg'
gui_title = 'awesome image'

image = cv2.imread(image_path)
cv2.imshow(gui_title, image)
cv2.waitKey(0)
cv2.destroyAllWindows()
