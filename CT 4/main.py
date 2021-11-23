"""
Karl Estes
CSC 515 Critical Thinking 4
Created: November 1st, 2021
Due: November 7th, 2021

Asignment Prompt
----------------
Image filtering involves the application of window operations that perform useful functions, 
such as noise removal and image enhancement. Compare the effects of Laplacian and Gaussian filters 
on an image for different kernel windows.

This image (Links to an external site.) contains Gaussian noise. In OpenCV, write algorithms for this 
image to do the following:

Apply a Gaussian, a Laplacian, and a Gaussian with Laplacian filter using a 3x3 kernel. 
For Gaussian, think about how to select a good value of sigma for optimal results.

Apply a Gaussian, a Laplacian, and a Gaussian with Laplacian filter using a 5x5 kernel. 
For gaussian, use the same value of sigma you selected in the above step.

Apply a Gaussian, a Laplacian, and a Gaussian with Laplacian filter using a 7x7 kernel. 
For gaussian, use the same value of sigma you selected in the above step.

Output your filter results as 3 x 3 side-by-side subplots to make comparisons easy to inspect visually. 
That is, your subplot should have 3 rows (1 for each kernel size) and 3 columns (1 for each filter type). 
Be sure to include row and column labels.

File Description
----------------
TODO: File Description

Comment Anchors
---------------
I am using the Comment Anchors extension for Visual Studio Code which utilizes specific keywords
to allow for quick navigation around the file by creating sections and anchor points. Any use
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with 
this extension. To trigger these keywords, they must be typed in all caps. 
"""

import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from os import system, name

def clear_terminal():
    """Clears the terminal of all text on Windows/MacOs/Linux"""
    
    # For windows
    if name == 'nt':
        _ = system('cls')
    # Mac and linux 
    else:
        _ = system('clear')

def apply_filters(img, kernel_size: tuple, sigma) -> tuple:
    '''
    Creates 3 copies of img with the following:
        - Gaussian filter
        - Laplacian filter
        - Gaussian and Laplacian filter

    Returns list containing those three images in that specific order
    '''

    # Apply gaussian
    gauss_img = cv.GaussianBlur(img,kernel_size,sigma)

    # Apply laplace
    ddepth = cv.CV_16S
    print(ddepth)
    laplace_img = cv.Laplacian(img,ddepth)

    # Apply both
    gauss_laplace_img = cv.GaussianBlur(img,kernel_size,sigma)
    gauss_laplace_img = cv.Laplacian(gauss_laplace_img,ddepth, ksize=kernel_size[0])

    return (gauss_img,laplace_img,gauss_laplace_img)

def plot_images(col1, col2, col3, sigma):
    fig, axs = plt.subplots(3,3)

    fig.suptitle(f'Gaussian and Laplacian Filters (Sigma = {sigma})')

    for i in range(3):
        for j in range(3):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

    axs[0,0].set_title('Gaussian')
    axs[0,0].imshow(col1[0])
    axs[0,0].set_ylabel("3 X 3 Kernel")
    axs[1,0].imshow(col2[0])
    axs[1,0].set_ylabel("5 X 5 Kernel")
    axs[2,0].imshow(col3[0])
    axs[2,0].set_ylabel("7 X 7 Kernel")

    axs[0,1].set_title("Laplacian")
    axs[0,1].imshow(col1[1])
    axs[1,1].imshow(col2[1])
    axs[2,1].imshow(col3[1])

    axs[0,2].set_title("Gaussian & Laplacian")
    axs[0,2].imshow(col1[2])
    axs[1,2].imshow(col2[2])
    axs[2,2].imshow(col3[2])

    plt.show()




if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Load image
    src = cv.imread("image.jpg")

    if src is None:
        print("ERROR: Unable to load image")
        exit(0)

    src = cv.cvtColor(src, cv.COLOR_RGB2BGR)

    sigma = 1

    # Apply 3x3 Kernel
    images_3_3 = apply_filters(src, (3,3), sigma)

    # Apply 5x5 Kernel
    images_5_5 = apply_filters(src, (5,5), sigma)

    # Apply 7x7 Kernel
    images_7_7 = apply_filters(src, (7,7), sigma)

    plot_images(images_3_3, images_5_5, images_7_7,sigma)