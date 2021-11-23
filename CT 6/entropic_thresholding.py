"""
Karl Estes
CSC 515 Critical Thinking 6 - Adaptive Thresholding Scheme for Simple Objext
Created: November 16th, 2021
Due: November 21st, 2021

Asignment Prompt
----------------
If an image has been preprocessed properly to remove noise, a key step that is generally used when interpreting that image is segmentation. 
Image segmentation is a process in which regions or features sharing similar characteristics are identified and grouped together.

The thresholds in the algorithms discussed in this module were chosen by the designer. In order to make segmentation stronger to variations 
in the scene, the algorithm should be able to select an appropriate threshold automatically using the amount of intensity present in the image. 
The knowledge about the gray values of objects should not be hard-coded into an algorithm. The algorithm should use knowledge about the relative 
characteristics of gray values to select the appropriate threshold.  A thresholding scheme that uses such knowledge and selects a proper threshold 
value for each image without human intervention is called an adaptive thresholding scheme.

Find on the internet (or use a camera to take) three different types of images: an indoor scene, outdoor scenery, and a close-up scene of a single 
object. Implement an adaptive thresholding scheme to segment the images as best as you can.

File Description
----------------
This script contains an implementation of adaptive thresholding with an entropy based approach. The entropy for all threshold possibilities is calculated
and the threshold level corresponding to the maximum entropy is chosen for binarization. Edges of objects are subsequently extracted from the binary 
image and drawn onto a copy of the original

Comment Anchors
---------------
I am using the Comment Anchors extension for Visual Studio Code which utilizes specific keywords
to allow for quick navigation around the file by creating sections and anchor points. Any use
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with 
this extension. To trigger these keywords, they must be typed in all caps. 
"""

import cv2 as cv
import numpy as np
from numpy.lib.type_check import _imag_dispatcher
import progressbar
import math
import argparse
import sys
import matplotlib.pyplot as plt
from os import system, name

def clear_terminal():
    """Clears the terminal of all text on Windows/MacOs/Linux"""
    
    # For windows
    if name == 'nt':
        _ = system('cls')
    # Mac and linux 
    else:
        _ = system('clear')

def find_entropic_threshold(img):
    '''
    Takes a single grayscale image and returns optimal threshold value based on entropy characteristics
    
    Returns tuple -> (k, H_K) 
        - k = threshold
        - H_K = max_entropy'''

    # Generate histogram
    hist = cv.calcHist([img], [0], None, [256], [0,256])

    # Loop through item in histogram and calculate probabilities
    probs = []
    N = img.shape[0] * img.shape[1]

    for value in hist:
        if value == 0:
            probs.append(0)
        else:
            probs.append(value[0]/N)

    # Find max entropy threshold
    P_K = 0
    max_entropy = None # Will be tuple with (k, H(k))
    for k in progressbar.progressbar(range(len(hist))):
        P_K += probs[k]

        # Calculate entropy up to threshold
        temp = []
        for i in range(k+1): # Up to an including k
            if probs[i] == 0:
                temp.append(0)
            else:
                temp.append((probs[i]/P_K) * math.log(probs[i]/P_K))
        H_A = - np.sum(temp)

        # Calculate entropy for beyond threshold
        temp = []
        for i in range(k+1, len(hist)):
            if probs[i] == 0:
                temp.append(0)
            else:
                temp.append((probs[i]/(1-P_K)) * math.log(probs[i]/(1-P_K)))
        H_B = - np.sum(temp)

        H_K = H_A + H_B # Total entropy
        
        if (max_entropy is None) or (max_entropy[1] < H_K):
            max_entropy = (k, H_K)
        
    return max_entropy

if __name__ == "__main__":

    # Setup arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help="Path to image for processing")

    args = parser.parse_args()

    clear_terminal()

    img = cv.imread(args.image_path) # Attempt to load image

    if img is None:
        print(f"Unable to load image {args.image_path}")
        sys.exit(2)

    image_name = args.image_path[:-4]

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Grayscale conversion
    
    cv.imshow("Gray", gray)
    cv.waitKey(0)

    k, H_K = find_entropic_threshold(gray) # Find threshold based on entropy
    print(f"Found threshold {k} with max entropy {H_K}")

    ret, thresh = cv.threshold(gray, k, 255, cv.THRESH_BINARY)

    cv.imwrite(f"processed_images/binarized_{image_name}.jpg", thresh)


    cv.imshow(f"binarized with thresh={k}",thresh)
    cv.waitKey(0)

    


