"""
Karl Estes
CSC 515 Discussion Post 2
Created: October 20th, 2021
Due: October 21st, 2021

Asignment Prompt
----------------
Many computer vision applications use geometric transformations to change the position, orientation, and size of objects present in a scene.  
One such application where geometric transformations should be applied is counterfeit banknote detection. 

Take a look at [this image](https://frostlor-cdn-prod.courses.csuglobal.edu/lor/resources/src/272137f2-3816-3779-a73f-71882da43fa1/shutterstock227361781--125.jpg). 
Import this image into OpenCV and examine its pixels matrix.  Discuss the translations you would perform to identify whether the banknotes are counterfeit and why 
you would apply them for counterfeit detection.  Describe how you would manually apply the transformations to this image’s pixels matrix? Discuss in detail your 
transformation matrices.  Apply your transformations and attach your transformed image to your post.


File Description
----------------
Python script to perform a series of transformation on the linked image. 

Comment Anchors
---------------
I am using the Comment Anchors extension for Visual Studio Code which utilizes specific keywords
to allow for quick navigation around the file by creating sections and anchor points. Any use
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with 
this extension. To trigger these keywords, they must be typed in all caps. 
"""

import cv2 as cv
import numpy as np

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    Retrieved from https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)
    print(rotation_mat)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

img = cv.imread("money.jpg", 1)
print("Image Dimensions (height, width, channels): ", img.shape)
height, width, channels = img.shape
# scale_percent = 200
# print('Original Dimensions : ',img.shape)

# height_scaled = int(height * scale_percent / 100)
# width_scaled = int(width * scale_percent / 100)

# resized_img = cv.resize(img, (width_scaled, height_scaled), interpolation=cv.INTER_AREA)

# cv.imshow("Scaled", resized_img)
# print('Resized Dimensions : ',resized_img.shape)

# *** Crop Bills ***
bill1 = img[20:,8:45]

rotated_img = rotate_image(img, 170)
print("Roated Image Shape: ", rotated_img.shape)
bill2 = rotated_img[10:84, 20:63]

print("Bill Shapes: ", bill1.shape, bill2.shape)

# Ensure Same size
bill2 = cv.resize(bill2, (37,63))

print("Bill Shapes Resized: ", bill1.shape, bill2.shape)

# Concat cropped bills with separating black lines
bills = np.zeros([bill1.shape[0], (bill1.shape[1]*2) + 5, 3], 'uint8')
bills[:, 0:bill1.shape[1]] = bill1
bills[:, (bill1.shape[1] + 5):] = bill2

# *** NOISE REDUCTION? ***
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

other_kernel = np.array([[1,4,6,4,1],
                        [4,16,24,16,4],
                        [6,24,-476,24,6],
                        [4,16,24,16,4],
                        [1,4,6,4,1]])
other_kernel = other_kernel * (-1/256)

sharpened_bills = cv.filter2D(src=bills, ddepth=-1, kernel=other_kernel)


# Edge Detection
# Convert to grayscale
#bills_gray = cv.cvtColor(bills, cv.COLOR_BGR2GRAY)
# Blurred Image for better edge detection
#bills_blurred = cv.GaussianBlur(bills_gray, (3,3), 0)
#bills_blurred = bills_gray

# # Sobel Edge Detection
# sobelx = cv.Sobel(src=bills_blurred, ddepth=cv.CV_64F, dx=1, dy=0, ksize=3) # Sobel Edge Detection on the X axis
# sobely = cv.Sobel(src=bills_blurred, ddepth=cv.CV_64F, dx=0, dy=1, ksize=3) # Sobel Edge Detection on the Y axis
# sobelxy = cv.Sobel(src=bills_blurred, ddepth=cv.CV_64F, dx=1, dy=1, ksize=3) # Combined X and Y Sobel Edge Detection
# # Display Sobel Edge Detection Images

# cv.imshow("Sobel X", sobelx)
# cv.imshow("Sobel Y", sobely)
# cv.imshow("Sobel XY", sobelxy)

#edges1 = cv.Canny(image=bills_blurred, threshold1=50, threshold2=100)

bills_gray = cv.cvtColor(sharpened_bills, cv.COLOR_BGR2GRAY)
# Blurred Image for better edge detection
bills_blurred = cv.GaussianBlur(bills_gray, (3,3), 0)
edges = cv.Canny(image=bills_blurred, threshold1=50, threshold2=100)


# *** COLOR ***
hsv = cv.cvtColor(bills, cv.COLOR_BGR2HSV)

mask = cv.inRange(hsv, (10, 100, 20), (25, 255, 255))
masked_bills_orange = cv.bitwise_and(bills, bills, mask=mask)

mask = cv.inRange(hsv, (110, 30, 20), (130, 255, 255))
masked_bills_blue = cv.bitwise_and(bills, bills, mask=mask)

mask = cv.inRange(hsv, (40, 50, 20), (80, 255, 255))
masked_bills_green = cv.bitwise_and(bills, bills, mask=mask)

masked_bills = np.concatenate((masked_bills_blue, masked_bills_green, masked_bills_orange),axis=1)

cv.imshow("Original", img)
cv.imshow("Rotated Image", rotated_img)
cv.imshow("Copped Bills", bills)
cv.imshow("Sharpened Image", sharpened_bills)
cv.imshow("Canny Edge Detection", edges)
cv.imshow("HSV Bills", hsv)
cv.imshow("MASK Bills", masked_bills)

cv.imwrite("rotated_image.jpg", rotated_img)
cv.imwrite("cropped_bills.jpg", bills)
cv.imwrite("sharpened_image.jpg", sharpened_bills)
cv.imwrite("canny_edge_detection.jpg", edges)
cv.imwrite("hsv_bills.jpg", hsv)
cv.imwrite("color_masks.jpg", masked_bills)
cv.waitKey(0)
cv.destroyAllWindows()



# Two methods for concatenating images for openCV with Numpy
