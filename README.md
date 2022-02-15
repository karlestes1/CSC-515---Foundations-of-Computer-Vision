# CSC 515 - Foundations of Computer Vision
**Disclaimer:** These projects were built as a requirement for CSC 515: Foundations of Computer Vision at Colorado State University Global under the instruction of Dr. Brian Holbert. Unless otherwise noted, all programs were created to adhere to explicit guidelines as outlined in the assignment requirements I was given. Descriptions of each [programming assignment](#programming-assignments) and the [portfolio project](#portfolio-project) can be found below.

*****This class has been completed, so this repository is archived.*****
___

### Languages and Tools
[<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/python.svg" />](https://www.python.org)
[<img align="left" height="32" width="32" src="https://www.psych.mcgill.ca/labs/mogillab/anaconda2/lib/python2.7/site-packages/anaconda_navigator/static/images/anaconda-icon-512x512.png" />](https://www.anaconda.com/pricing)
[<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/visual-studio-code.svg" />](https://code.visualstudio.com)
[<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/git-icon.svg" />](https://git-scm.com)
[<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/gitkraken.svg" />](https://www.gitkraken.com)
[<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/opencv.svg" />](https://opencv.org)
<br />

### Textbook
The textbooks for this class were [**Computer Vision and Image Processing: Fundamentals and Applications**](https://www.taylorfrancis.com/books/mono/10.1201/9781351248396/computer-vision-image-processing-manas-kamal-bhuyan) by **Manas Kamal Bhuyan** and [**Computer Vision: Principles, Algorithms, Applications, Learning**](https://www.elsevier.com/books/computer-vision/davies/978-0-12-809284-2) by **E. R. Davies**
### VS Code Comment Anchors Extension
I am also using the [Comment Anchors extension](https://marketplace.visualstudio.com/items?itemName=ExodiusStudios.comment-anchors) for Visual Studio Code which places anchors within comments to allow for easy navigation and the ability to track TODO's, code reviews, etc. You may find the following tags intersperesed throughout the code in this repository: ANCHOR, TODO, FIXME, STUB, NOTE, REVIEW, SECTION, LINK, CELL, FUNCTION, CLASS

For anyone using this extension, please note that CELL, FUNCTION, and CLASS are tags I defined myself. 
<br />

___
<!--When doing relative paths, if a file or dir name has a space, use %20 in place of the space-->
## Programming Assignments
### Critical Thinking 3: [Tensorflow ANN Model](CT%203/)
- A basic Tensorflow ANN model trained to model a simple mathematical function.
### Critical Thinking 4: [Laplacian & Gaussian Filters for Different Kernel Windows](CT%204/)
- A simple script to apply and view the effect of Laplacian and Gaussian filters on an image with three different kernel sizes
### Critical Thinking 5: [Morphology Operations for Handwritten Text Enhancement](CT%205/)
- A program which takes image of handwriting on a sticky note, binarizes the iamge, and applyes four morphological operations to it
    - Opening
    - Closing
    - Erosion
    - Dilation
- Each resulting image is displayed for veiwing

### Critical Thinking 6: [Adaptive Thresholding Scheme for Simple Objects](CT%206/)
- This script contains an implementation of adaptive thresholding with an entropy based approach. The entropy for all threshold possibilities is calculated
and the threshold level corresponding to the maximum entropy is chosen for binarization. Edges of objects are subsequently extracted from the binary 
image and drawn onto a copy of the original
- The idea behind entropy based threshholding for image binarization is rooted in the idea that binzarizing an image at the point of greatest entropy will bring the greatest order
    - Total entropy is calculated by taking the entropy of the intensity probability distribution of all pixels up to some value *k*, and adding it to the entropy of the intensity probability distribution of all pixels after *k*
    - Total entropy is computed for all possible pixel values of *k* and the value with maximum entropy becomes the threshold value for image binarization
___
## Portfolio Project: [Face Detection and Privacy](Portfolio%20Project/)
- The purpose of this project was to implement a basic facial recognition algorithm, find the eyes within the face, and blur the detected area. Per the assignment instructions: **To address privacy concerns you may want to use data anonymization.  On images, this can be achieved by hiding features that could lead to a person or personal data identification, such as the person’s facial features or a license plate number.**
- Face and eye detection were done using [OpenCV's haarcascade classifiers](https://github.com/opencv/opencv/tree/master/data/haarcascades)
- The images are first converted to grayscale and then contrast equalized before any detection is done

### Image Examples
#### Starting
![](Portfolio%20Project/Final%20Project/final1.jpg)

#### Grayscale
![](Portfolio%20Project/Final%20Project/grayscale/gray_0.jpg)

#### Equalized
![](Portfolio%20Project/Final%20Project/contrast_equalized/equal_0.jpg)

#### Detected Faces
![](Portfolio%20Project/Final%20Project/detected_faces/1.jpg)
#### Cropped and Aligned Face
![](Portfolio%20Project/Final%20Project/final_faces/1.jpg)

#### Blurred Eyes
![](Portfolio%20Project/Final%20Project/final/face_0.jpg)