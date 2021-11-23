"""
Karl Estes
CSC 515 Module 1 Portfolio Milestone
Created: October 12, 2021
Due: October 17th, 2021


Asignment Prompt
----------------
It is time to begin thinking about your Portfolio Project.  In order to complete the Portfolio Project, OpenCV will need to be installed and working properly on your desktop. 

OpenCV (Open-Source Computer Vision Library) is an open source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in commercial products.

For this milestone assignment, install OpenCV based on your specific operating system.  Then, use OpenCV to complete the following:
1.	Write Python code to import [this image](https://frostlor-cdn-prod.courses.csuglobal.edu/lor/resources/src/7985e0c1-89b3-35e6-b709-8c90e828356c/shutterstock93075775--250.jpg)
2.	Write Python code to display the image.
3.	Write Python code to write a copy of the image to any directory on your desktop.

Your submission should be one executable Python file.

File Description
----------------
Basic script to fulfill assignment prarameters

Comment Anchors
---------------
I am using the Comment Anchors extension for Visual Studio Code which utilizes specific keywords
to allow for quick navigation around the file by creating sections and anchor points. Any use
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with 
this extension. To trigger these keywords, they must be typed in all caps. 
"""

import cv2 as cv
import sys
import os

homePath = os.path.expanduser("~")

print("* * * CSC 515 - Module 1 Portfolio Milestone * * *")
print("\nImporting image")

# Load the image
img = cv.imread("image.jpg")

if img is None:
    sys.exit("Could not load the image. Please ensure image is in same directory as script and is named image.jpg")

print("Image Loaded")
# Display the image
cv.imshow("Display window", img)
print("Press s to save image to desktop or any other key to continue...")
k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite(homePath + "/Desktop/image.jpg", img)
    print(f"Images saved to {homePath}/Desktop")





