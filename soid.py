import sys
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('image.jpg')            
b,g,r = cv2.split(img)               
source_image = cv2.merge([r,g,b])

dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21) 

b,g,r = cv2.split(dst)                
output_image = cv2.merge([r,g,b])

plt.subplot(211),plt.imshow(source_image)
plt.subplot(212),plt.imshow(output_image)
plt.show()
