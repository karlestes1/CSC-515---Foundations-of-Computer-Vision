"""
Karl Estes
CSC 515 Portfolio Project
Created: November 13th, 2021
Due: December 5th, 2021

Asignment Prompt
----------------
To address privacy concerns you may want to use data anonymization.  On images, this can be achieved by hiding features 
that could lead to a person or personal data identification, such as the person’s facial features or a license plate number.

The goal of this project is to write algorithms for face detection and feature blurring.  Select three color images from 
the web that meet the following requirements:

Two images containing human subjects facing primarily to the front and one image with a non-human subject.
- At least one image of a human subject should contain that person’s entire body.
- At least one image should contain multiple human subjects.
- At least one image should display a person’s face far away.
- All images should vary in light illumination and color intensity. 

First, using the appropriate trained cascade classifier (Links to an external site.), write one algorithm to detect the human 
faces in the gray scaled versions of the original images.  Put a red boundary box around the detected face in the image in order 
to see what region the classifier deemed as a human face. If expected results are not achieved on the unprocessed images, apply 
processing steps before implementing the classifier for optimal results.

After the faces have been successfully detected, you will want to process only the extracted faces before detecting and applying 
blurring to hide the eyes.  Although the eye classifier (Links to an external site.) is fairly accurate, it is important that all 
faces are centered, rotated, and scaled so that the eyes are perfectly aligned. If expected results are not achieved, implement 
more image processing for optimal eye recognition. Now, apply a blurring method to blur the eyes out in the extracted image.

File Description
----------------
TODO - Add file description

Comment Anchors
---------------
I am using the Comment Anchors extension for Visual Studio Code which utilizes specific keywords
to allow for quick navigation around the file by creating sections and anchor points. Any use
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with 
this extension. To trigger these keywords, they must be typed in all caps. 
"""

import cv2 as cv
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import os
import progressbar

class ImageProcessor():
    '''An object which contains processes (implemented with IOpenCV) that can be done to an image'''

    def convert_to_grayscale(self, images: list):
        '''Takes a list of images and returns a list of images converted to grayscale'''
        converted_images = []
        for img in images:
            converted_images.append(cv.cvtColor(img,cv.COLOR_BGR2GRAY))
        
        return converted_images

    def adaptive_gamma_correction(self, images: list) -> list:
        '''
        Takes in a list of images and applies adaptive gamma correction each of them

        The following algorithm is a slight modification from LeoW's Improved Adaptive Gamma Correction algorithm which
        was based on the paper "Contrast enhancement of brightness-distorted images by improved adaptive gamma correction."

        Repo link: https://github.com/leowang7/iagcwd
        Paper link: https://arxiv.org/abs/1709.04427
        '''
        
        corrected_images = []

        for img in images:

            # START OF CODE TAKEN FROM LeoW's REPO
            # Extract intensity component of the image
            YCrCb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
            Y = YCrCb[:,:,0]
            # Determine whether image is bright or dimmed
            threshold = 0.3
            exp_in = 112 # Expected global average intensity 
            M,N = img.shape[:2]
            mean_in = np.sum(Y/(M*N)) 
            t = (mean_in - exp_in)/ exp_in
            
            # Process image for gamma correction
            img_output = None
            if t < -threshold: # Dimmed Image
                result = self._process_dimmed(Y)
                YCrCb[:,:,0] = result
                img_output = cv.cvtColor(YCrCb,cv.COLOR_YCrCb2BGR)
            elif t > threshold:
                result = self._process_bright(Y)
                YCrCb[:,:,0] = result
                img_output = cv.cvtColor(YCrCb,cv.COLOR_YCrCb2BGR)
            else:
                img_output = img
             # END OF CODE TAKEN FROM LeoW's REPO

            corrected_images.append(img_output)

        return corrected_images
            
    def _image_agcwd(self, img, a=0.25, truncated_cdf=False):
        '''This function was taken directly from LeoW's Improved Adaptive Gamma Correction algorithm (https://github.com/leowang7/iagcwd)'''
        h,w = img.shape[:2]
        hist,bins = np.histogram(img.flatten(),256,[0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()
        prob_normalized = hist / hist.sum()

        unique_intensity = np.unique(img)
        intensity_max = unique_intensity.max()
        intensity_min = unique_intensity.min()
        prob_min = prob_normalized.min()
        prob_max = prob_normalized.max()
        
        pn_temp = (prob_normalized - prob_min) / (prob_max - prob_min)
        pn_temp[pn_temp>0] = prob_max * (pn_temp[pn_temp>0]**a)
        pn_temp[pn_temp<0] = prob_max * (-((-pn_temp[pn_temp<0])**a))
        prob_normalized_wd = pn_temp / pn_temp.sum() # normalize to [0,1]
        cdf_prob_normalized_wd = prob_normalized_wd.cumsum()
        
        if truncated_cdf: 
            inverse_cdf = np.maximum(0.5,1 - cdf_prob_normalized_wd)
        else:
            inverse_cdf = 1 - cdf_prob_normalized_wd
        
        img_new = img.copy()
        for i in unique_intensity:
            img_new[img==i] = np.round(255 * (i / 255)**inverse_cdf[i])
    
        return img_new

    def _process_bright(self, img):
        '''This function was taken directly from LeoW's Improved Adaptive Gamma Correction algorithm (https://github.com/leowang7/iagcwd)'''
        img_negative = 255 - img
        agcwd = self._image_agcwd(img_negative, a=0.25, truncated_cdf=False)
        reversed = 255 - agcwd
        return reversed

    def _process_dimmed(self, img):
        '''This function was taken directly from LeoW's Improved Adaptive Gamma Correction algorithm (https://github.com/leowang7/iagcwd)'''
        agcwd = self._image_agcwd(img, a=0.75, truncated_cdf=True)
        return agcwd

    def difference_of_gaussian(self, images: list, k_size1: tuple = (3,3), sigma1: float = 0, k_size2 : tuple = (5,5), sigma2 : float = 0):
        '''
        Applies a DOG processing step to all images in the provided list

        Parameters
        ----------
        images : list
            list of images
        k_size1 : tuple
            Size of the kernel for the first gaussian pass
        sigma1 : float
            Value for the number of stddev's in first gaussian pass
        k_size2 : tuple
            Size of kernel for the second gaussian pass
        sigma2 : float
            Value for the number of stddev's in the second gaussian pass

        Return
        ------
        dog_images : list
            list of images after DOG processing
        '''

        dog_images = []

        for img in images:
            gauss1 = cv.GaussianBlur(img, k_size1, sigma1)
            gauss2 = cv.GaussianBlur(img, k_size2, sigma2)

            dog = gauss1-gauss2

            dog_images.append(dog)

        return dog_images
    
    def contrast_equalization(images: list, clip_limit: float = 40.0, grid_size : tuple = (8,8)):
        '''Applies contrast limited adaptive histogram equalization to all images in the list'''
        
        equalized_images = []

        # Create CLAHE object
        clahe = cv.createCLAHE(clip_limit, grid_size)

        for img in images:
            equalized_images.append(clahe.apply(img))

        return equalized_images

    def align_eyes(self, faces_and_eyes: list):
        '''
        Aligns all faces so that eyes are horizontal

        Parameters
        ----------
        faces_and_eyes : list
            [[face, [l_eye, r_eye]], [face, [l_eye, r_eye]], ...] l_eye/r_eye = (x,y,w,h)

        Return
        ------
        # TODO - Put Return
        '''

        aligned_images = []

        for i, face in enumerate(faces_and_eyes):

            # Get Left and Right Eyes
            if face[1] != None:
                l_eye,r_eye = face[1]

                l_eye_center = (l_eye[0] + l_eye[2]//2, l_eye[1] + l_eye[3]//2)
                r_eye_center = (r_eye[0] + r_eye[2]//2, r_eye[1] + r_eye[3]//2)

                # Calculate angle of rotation
                theta_rad = math.atan2(r_eye_center[1] - l_eye_center[1], r_eye_center[0]-l_eye_center[0])
                theta_deg = math.degrees(theta_rad)

                aligned_images.append(self._rotate_image(face[0], theta_deg))
            else:
                print(f"WARNING: Unable to rotate face {i+1} due to missing eye detection")

        return aligned_images

    def _rotate_image(self, mat, angle):
        '''
        Rotates an image (angle in degrees) and expands image to avoid cropping
        Retrieved from https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
        '''

        height, width = mat.shape[:2] # image shape has 3 dimensions
        image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)
        # print(rotation_mat)

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

    # TODO - Eye Blurring

class FaceFinder():

    def __init__(self, face_cascade_path, eye_cascade_path) -> None:
        '''
        Parameters
        ----------
        face_cascade_path : string
            Path to an xml file with data for a CascadeClassifier associated with detecting faces
        eye_cascade_path : string
            Path to an xml file with data for a CascadeClassifier associated with detecting eyes
        '''
        self.face_cascade = cv.CascadeClassifier()
        self.eye_cascade = cv.CascadeClassifier()

        if not self.face_cascade.load(face_cascade_path):
            print('--(!) ERROR loading face cascade')
            sys.exit(2)
        if not self.eye_cascade.load(eye_cascade_path):
            print('--(!) ERROR loading eye cascade')

    def find_faces(self, images : list, scale = 1.2, min_neighbors = 4, min_size : tuple = None):
        '''
        Find all faces and bounding boxes in each image that is provided
        
        Parameters
        ----------
        scale : float
            Scaling factor of image for CascadeClassifier (1.2 = 120%)
        min_neighbors : int
            How many neighbors need to be found during detection. (Higher number = harder detection)
        min_size : tuple
            Minimum size in pixels of bounding box for detected face to be considered valid 

        Return
        ------
        found_faces : list of lists
            Each sublist is comprised of the original images and a list of all faces found in that image
                Each list of all faces contains a tuple, one is the cropped image and the other are the x,y,w,h coords
        
        [[img1, [[face1, (x,y,w,h)],[face2, (x,y,w,h)]]], [img2, ...]]
        '''

        found_faces = []

        for img in images:

            faces_in_img = []

            # Find faces
            faces = self.face_cascade.detectMultiScale(img, scale, min_neighbors, minSize=min_size)

            for face in faces:
                (x,y,w,h) = face
                faceROI = img[y:y+h, x:x+w]

                faces_in_img.append([faceROI, face])

            found_faces.append([img, faces_in_img])
        
        return found_faces

    def find_eyes(self, faces : list, scale = 1.2, min_neighbors = 4, min_size : tuple = None):
        '''
        Find all eyes and bounding boxes in each image that is provided. Will return none if less or more than 2 eyes are detected in an image
        
        Parameters
        ----------
        scale : float
            Scaling factor of image for CascadeClassifier (1.2 = 120%)
        min_neighbors : int
            How many neighbors need to be found during detection. (Higher number = harder detection)
        min_size : tuple
            Minimum size in pixels of bounding box for detected face to be considered valid 

        Return
        ------
        found_eyes : list of lists
            Each sublist is comprised of the face image and a list of eyes found in that image (None value if no eyes)
                Each eye list contains (x,y,w,h) for the l_eye and r_eye
        
        [[face, [l_eye, r_eye]], [face, [l_eye, r_eye]], ...]
        
        l_eye/r_eye = (x,y,w,h)
        '''
        found_eyes = []

        for face in faces:

            eyes = self.eye_cascade.detectMultiScale(face, scale, min_neighbors, minSize=min_size)

            if len(eyes) == 2:

              # Check if angle is greater than 15 degrees
                l_eye,r_eye = eyes
                
                if r_eye[0] < l_eye[0]:
                    r_eye, l_eye = l_eye, r_eye

                # Calculate angle of rotation
                theta_rad = math.atan2(r_eye[1] - l_eye[1], r_eye[0]-l_eye[0])
                theta_deg = math.degrees(theta_rad)
                if(theta_deg < 15):  
                    found_eyes.append([face, [l_eye,r_eye]])
                else:
                    found_eyes.append([face, None])
            else:
                found_eyes.append([face, None])

        return found_eyes

def load_images(paths):
    '''Attempts to load all images from the provided paths'''

    images = []

    for path in paths:
        try:
            img = cv.imshow(path, 1)
            if img is None:
                print(f"WARNING: Unable to load image at {path}")
            else:
                images.append(img)
        except:
            print(f"WARNING: Unable to load image at {path}")

    return images


def interactive_image_viewer(images : list, window_name : str = 'Default'):
    '''
    Uses OpenCV to create an interactive image viewer

    Keybindings are as follows
    - n : Next Image
    - p : Previous Image
    - e/'esc' : close
    '''

    if (images is None) or (len(images) == 0) :
        print("ERROR: Cannot view images that do not exist!")
        return

    cur = 0
    viewing = True
    window = cv.namedWindow(window_name)

    if args.debug:
        print("Press the following keys to interact with the images:\n\tn - next image\n\tp - previous images\n\te/'esc' - close image viewer\n")

    while viewing:
        cv.imshow(window_name, images[cur])
        k = cv.waitKey(1) & 0xFF # Wait for 1 ms each time

        if k == 27 or k == ord('e'):
            viewing = False
        elif k == ord('n'):
            cur = cur + 1 if cur < (len(images)-1) else 0
        elif k == ord('p'):
            cur = cur - 1 if cur > 0 else (len(images)-1)

    cv.destroyAllWindows() # Destroy all open windows

def save_images(images: list, names: list = None, dir: str = None, type=".jpg"):
    '''
    Saves a list of images with the specified type using OpenCV
    
    Parameters
    ----------
    - images : list
        The list of images that are to be saved
    - names : list[str]
        Names for each of the images (error will be thrown if mistmatch in len)
    - dir : str
        Path to folder to save images (default to local) (will create directory if it doesnt exist)
    - type : str
        Denotes the file type that the image will be saved as
    '''

    
    if images is None or len(images) == 0:
        print("ERROR: Cannot save images that don't exist")
        return

    if (not (names is None)) and len(images) != len(names):
        print("ERROR: Mismatch in length of image list and length of names to be assigned during save")
        return

    if not (dir is None):
        is_dir = os.path.isdir(dir)

        # If folder doesn't exist, then create it.
        if not is_dir:
            os.makedirs(dir)
            print(f"Created dir={dir}")
    
    for i,img in enumerate(images):
        if names is None:
            cv.imwrite(os.path.join(dir, f"{i+1}{type}"), img)
        else:
            cv.imwrite(os.path.join(dir, f"{names[i]}{type}"), img)
        
    
    

if __name__ == "__main__":
     
    parser = argparse.ArgumentParser()

    # Paths for cascade files
    parser.add_argument('--face_cascade_path', type=str, default='TODO - DEFAULT PATH',
                        help="Path to xml file containing information to be loaded into OpenCV CascadeClassifier()")
    parser.add_argument('--eye_cascade_path', type=str, default='TODO - DEFAULT PATH',
                        help="Path to xml file containing information to be loaded into OpenCV CascadeClassifier()")

    # Paths for images
    parser.add_argument('-i', '--images', nargs='+', required=True, help='Individual file paths for each image you would like to load in')

    # Names for images
    parser.add_argument('--names', nargs='+', required=False, default=None, help='List of names to save each image as (NOTE: number of names should equal number of images provided)')

    # ImageProcessor() Args
    parser.add_argument('--k_size1', nargs=2, type=int, default=[3,3], metavar=('WIDTH', 'HEIGHT'), help="Two integers corresponding to size of first kernel for DOG filtering")
    parser.add_argument('--k_size2', nargs=2, type=int, default=[5,5], metavar=('WIDTH', 'HEIGHT'), help="Two integers corresponding to size of second kernel for DOG filtering")
    parser.add_argument('--sigma1', type=float, default=0, help="Sigma value for first Gaussian pass in DOG filtering")
    parser.add_argument('--sigma2', type=float, default=0, help="Sigma value for second Gaussian pass in DOG filtering")
    parser.add_argument('--clip_limit', type=float, default=40.0, help="Clip limit for contrast equalization")
    parser.add_argument('--grid_size', nargs=2, type=int, default=[8,8], metavar=('WIDTH', 'HEIGHT'), help="Two integers corresponding to grid size for contrast equalization")

    # FaceFinder() Args
    parser.add_argument('--scale_factor_face', type=float, default=1.2, help="Scale factor for face cascade classification")
    parser.add_argument('--scale_factor_eyes', type=float, default=1.2, help="Scale factor for eye cascade classification")
    parser.add_argument('--min_neighbors_face', type=int, default=4, help="Min_neighbors for face cascade classification")
    parser.add_argument('--min_neighbors_eyes', type=int, default=4, help="Min_neighbors for eye cascade classification")

    # TODO - Add args for min_size params?
    
    parser.add_argument('-d', '--debug', action='store_true', default=False, help="Add flag to enable thorough printing of program steps and information")

    args = parser.parse_args()

    # Handle conversion of lists to tuples
    args.grid_size = tuple(args.grid_size)
    args.k_size1 = tuple(args.k_size1)
    args.k_size2 = tuple(args.k_size2)

    if args.debug:
        print(f"\nStarting {sys.argv[0]} with arguments: {args}")


    # Attempt to load all images
    images = load_images(args.images)

    if len(images) == 0:
        print("ERROR: Unable to load any images. Terminating script...")
        sys.exit(4)

    if args.debug:
        print("Image viewer for original images")
        interactive_image_viewer(images, "Unmodified Images")

    # Load image processor and face finder
    processor = ImageProcessor()
    face_finder = FaceFinder(args.face_cascade_path, args.eye_cascade_path)

    # TODO - Debug Information
    '''
    Preprocessing Pipeline
    ----------------------
    Gamma Correction -> Grayscale -> DOG -> Contrast Equalization
    '''
    
    gamma_corrected = processor.adaptive_gamma_correction(images)

    grayscale = processor.convert_to_grayscale(gamma_corrected)

    dog = processor.difference_of_gaussian(grayscale, args.k_size1, args.sigma1, args.k_size2, args.sigma2)

    equalized = processor.contrast_equalization(dog, args.clip_limit, args.grid_size)

    # Save images
    print("Saving preprocessed images for later viewing")
    for group,dir in progressbar.progressbar([[gamma_corrected,'gamma_corrected'], [grayscale,'grayscale'], [dog,'difference_of_gaussian'], [equalized,'contrast_equalized']]):
        save_images(gamma_corrected, args.names, dir)

    # TODO - Detection Pipeline
    '''
    Detection Pipeline
    ------------------
    Find Faces -> Rotate -> Redetect -> Scale -> Save
    '''
    # Facial Detections
    # NOTE - Add args.min_size_face if min_size arg added to argparser
    image_face_list = face_finder.find_faces(equalized, args.scale_factor_face, args.min_neighbors_face)

    # TODO - Finish Main image processing code
