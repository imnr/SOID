import sys
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('image.jpg')       # gets the input (source) image for denoising. "image.jpg" is the directory of the target image      
b,g,r = cv2.split(img)              # gets the r,g,b values
source_image = cv2.merge([r,g,b])

dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21) # Denoising function. the values can be changed to get the best result

b,g,r = cv2.split(dst)                
output_image = cv2.merge([r,g,b])

#draw input and output
plt.subplot(211),plt.imshow(source_image)
plt.subplot(212),plt.imshow(output_image)
plt.show()
