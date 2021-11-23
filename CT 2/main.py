"""
Karl Estes
CSC 515 Critical Thinking 2
Created: October 19th, 2021
Due: October 24th, 2021

Asignment Prompt
----------------
Take a look at this [image](https://frostlor-cdn-prod.courses.csuglobal.edu/lor/resources/src/7b4ef85f-d88e-317f-952e-752a28cd944d/shutterstock147979985--250.jpg) of a kitten.  
Being a colored image, this image has three channels, corresponding to the primary colors of red, green, and blue.

1.	Import this image (using the link) into OpenCV and write code to extract each of these channels separately to create 2D images. 
    This means that from the n x n x 3 shaped image, you’ll get 3 matrices of the shape n x n.
2.	Now, write code to merge all these images back into a colored 3D image.
3.	What will the image look like if you exchange the reds with the greens? Write code to merge the 2D images created in step 1 back together, 
    this time swapping out the blue channel with the red channel (GRB).

Be sure to display the resulting images for each step. Your submission should be one executable Python file.

Comment Anchors
---------------
I am using the Comment Anchors extension for Visual Studio Code which utilizes specific keywords
to allow for quick navigation around the file by creating sections and anchor points. Any use
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with 
this extension. To trigger these keywords, they must be typed in all caps. 
"""

import cv2 as cv
import sys
import numpy as np

# Two methods for concatenating images for openCV with Numpy


if __name__ == "__main__":

    print("* * * CSC 515 - Module 1 Portfolio Milestone * * *")
    print("\nImporting image")

    # Load the image
    img = cv.imread("kitten.jpg")

    if img is None:
        sys.exit("Could not load the image. Please ensure image is in same directory as script and is named kitten.jpg")

    print("Loaded image of kitten")

    # Display the image
    cv.imshow("Kitten Image", img)
    print("Press Any key to continue...")
    k = cv.waitKey(0)
    cv.destroyAllWindows()

    # Split into channels
    b,g,r = cv.split(img)
    cv.imshow("Blue Kitten Channel", b)
    cv.imshow("Green Kitten Channel", g)
    cv.imshow("Red Kitten Channel", r)

    # METHOD 2: Using np.concatenate
    # axis=1 puts images horizontal
    rgb_concat = np.concatenate((b,g,r),axis=1)
    # cv.imwrite("bgr_channels.jpg", rgb_concat)

    print("Press Any key to continue...")
    k = cv.waitKey(0)
    cv.destroyAllWindows()

    merged_Img = cv.merge((b,g,r))
    merged_Swap = cv.merge((g,r,b))

    cv.imshow("Merged Image", merged_Img)
    cv.imshow("Merged Image w/ GRB Order", merged_Swap)
    # cv.imwrite("merged.jpg", merged_Img)
    # cv.imwrite("merged_GRB.jpg",merged_Swap)

    print("Press Any key to continue...")
    k = cv.waitKey(0)
    cv.destroyAllWindows()