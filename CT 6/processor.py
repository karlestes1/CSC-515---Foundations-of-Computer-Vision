import cv2 as cv
import math
import numpy as np

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
            print("t=",t)
            
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
                print("No processing done")
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
    
    def contrast_equalization(self, images: list, clip_limit: float = 40.0, grid_size : tuple = (8,8)):
        '''Applies contrast limited adaptive histogram equalization to all images in the list'''
        
        equalized_images = []

        # Create CLAHE object
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        #clahe = cv.createCLAHE()

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
