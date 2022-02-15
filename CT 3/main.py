"""
Karl Estes
CSC 515 Critical Thinking 3
Created: October 27th, 2021
Due: October 31st, 2021

Asignment Prompt
----------------
A facial recognition system is a technology capable of identifying or verifying a person from a digital image or a video frame from a video source. 
In general, facial recognition systems work by comparing selected facial features from a given image with faces within a database. 
Good facial recognition systems take into account the variation of pose, illumination of the image and facial expression. They also eliminate backgrounds and hairstyles, 
which are image properties that are not useful for computer vision tasks.

Recall that the MORPH-II dataset includes 55,134 mugshots with longitudinal spans taken between 2003 and late 2007. 
Write algorithms to process both images for [subject 1](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQdkf_IPn6VW3jpJ_fTU4IUcwtGPbcvSfxXW4EPXjeVuMjM1Baz) 
and [subject 2](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSulLs5l2Bwr6iFywEqHxQWevj9snjLRrsPxjQWsCIrJmA9cT4q). 
Processed images for each subject should be ideal for facial recognition accuracy. Final images for each subject should be in grayscale, 
adjusted for the effect on illumination, rotated based on the angle between the eye centers, cropped and scaled to create a new bounding box for the face. 
All final image bounding boxes should be the same dimensions.

Be sure to display the final processed images for each subject. Your submission should be an executable Python file.


File Description
----------------
The following script loads images from the associated data/faces folder. The two provided images 'subject1.jpeg' and 'subject2.jpeg', both consist
of two photos of the same individual.

The initial processing step is to load the data, split each image in 2 so each face is isolated, and convert each to grayscale. The grayscale conversion
is to help reduce "noise" in the image and remove unecessary color information. Both histogram equalization and CLAHE techniques were examined for handling
illumination differences, but they led to worse results on the automatic face detection further in the script. As such, simple grayscale conversion was utilized.

After the grayscale conversion, each image is processed for face features via a Haar Cascade Classifier. The accompanying .xml files were downloaded from
the OpenCV github repo. Any images who face and eye bounding boxes cannot be determine are stored in a temporary array and the user is asked if they would
like to manual draw the bounding boxes for each image. If they choose to, the user should draw the face bounding box, press 'c', draw the two eye bounding boxes, 
and press 'c' again. During the process, 'r' can be pressed to reset all boxes, and 'u' can be pressed to undo drawn boxes one-by-one.

Each image is then rotated to align the eyes on the horizontal axis. The same facial detection process is run again w/ the manual option included. Once the
second round of detection is complete, each image is cropped to include just the face region and each image is scaled to 70x80 pixels.

During each step of the process, images are saved to corresponding subfolders in the data folder. These subfolders contain a snapshot of the images
following each step of the process. In the cases where bounding boxes are detected, they are drawn onto a copy of the image that is saved so the boxes are
viewable but not drawn onto the images in processing.

Comment Anchors
---------------
I am using the Comment Anchors extension for Visual Studio Code which utilizes specific keywords
to allow for quick navigation around the file by creating sections and anchor points. Any use
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with 
this extension. To trigger these keywords, they must be typed in all caps. 
"""

import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from os import stat_result, system, name

def clear_terminal():
    """Clears the terminal of all text on Windows/MacOs/Linux"""
    
    # For windows
    if name == 'nt':
        _ = system('cls')
    # Mac and linux 
    else:
        _ = system('clear')

def load_data():
    '''Loads all subject faces from data/faces and loads the feature classifiers'''
    global face_cascade, eye_cascade
    print("Beginning Haar Classifier loading")
    images = []
    face_cascade = cv.CascadeClassifier()
    eye_cascade = cv.CascadeClassifier()

    # Load the classifiers
    if not face_cascade.load('data/haarcascade_fontalface_default.xml'):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eye_cascade.load('data/haarcascade_eye.xml'):
        print('--(!)Error loading eye cascade')
        exit(0)

    print("Loaded feature classifiers")

    print("Beginning image loading")
    for name in os.listdir('data/faces/'):
        path = 'data/faces/' + name
        img = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
        
        half = int(img.shape[1] / 2)
        l_img = img[:,:half]
        r_img = img[:,half:]

        images.append(l_img)
        images.append(r_img)

    print("Loaded images")
    return images

def grayscale_conversion(images: list):
    print("\nBeginning grayscale conversion")
    '''Takes a list of images and converts them to grayscale'''
    # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    for i, img in enumerate(images):
        # images[i] = clahe.apply(cv.cvtColor(img, cv.COLOR_RGB2GRAY))
        # images[i] = cv.equalizeHist(cv.cvtColor(img, cv.COLOR_RGB2GRAY))
        images[i] = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    print("Grayscale conversion complete")

def _detect_face(img):
    '''Attempts to automatically detect the face and eyes in the image and returns bounding box coordinates'''
    global face_cascade, eye_cascade

    # Face detection
    faces = face_cascade.detectMultiScale(img, 1.1, 2)
    if len(faces) == 1:
        (x,y,w,h) = faces[0]
        faceROI = img[y:y+h, x:x+w]

        # Eye detection
        eyes = eye_cascade.detectMultiScale(faceROI, 1.1, 2)

        if len(eyes) == 2:

            # Check if angle is greater than 15 degrees
            l_eye,r_eye = eyes
            
            if r_eye[0] < l_eye[0]:
                r_eye, l_eye = l_eye, r_eye

            # Calculate angle of rotation
            theta_rad = math.atan2(r_eye[1] - l_eye[1], r_eye[0]-l_eye[0])
            theta_deg = math.degrees(theta_rad)
            if(theta_deg < 15):
                return faces,eyes
    
    return None

def draw(event,x,y,flags,param):
    ''' 
    Mouse callback function for drawing manual bounding boxes
    
    When drawing the box, ensure the first click is the upper left corner of the box
    '''
    global ix,iy,drawing,manual_img,manual_img_copy,states,boxes

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv.EVENT_MOUSEMOVE:
        manual_img_copy = manual_img.copy()

        if drawing == True:
                cv.rectangle(manual_img_copy,(ix,iy),(x,y),(255,255,255),2)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.rectangle(manual_img,(ix,iy),(x,y),(255,255,255),2)
        boxes.append((ix,iy,(x-ix),(y-iy)))

        states.append(manual_img.copy())

        if len(boxes) == 1:
            print("Please press c to confirm bounding box for face or r to restart")
        if len(boxes) == 3:
            print("Please press c to confirm bounding boxes for eyes or r to restart")

def _manual_detect_faces(images: list):
    global ix,iy,drawing,manual_img,manual_img_copy,states,boxes

    processed = []
    drawing = False
    states = []
    boxes = []
    
    
    cv.namedWindow('Manual Face Detection')
    cv.setMouseCallback('Manual Face Detection',draw)

    for i, img in enumerate(images):

        print(f"\nManual detection for img {i+1} out of {len(images)}")
        
        states.clear()
        boxes.clear()
        finished = False
        face_box = False
        eye_box = False
        ix = -1
        iy = -1

        manual_img = img.copy()
        image_blank = manual_img.copy()
        manual_img_copy = manual_img.copy()
        states.append(manual_img.copy())

        while not finished:
            if drawing == True:
                cv.imshow('Manual Face Detection',manual_img_copy)
            else:
                cv.imshow('Manual Face Detection',manual_img)
        

            # Wait for keypress for 1ms
            k = cv.waitKey(1) & 0xFF

            if k == ord('c'): # Confirm
                if not face_box and len(boxes) == 1:
                    print("Face bounding box saved")
                    face_box = True
                    x,y,w,h = boxes[0]
                    faceROI = manual_img[y:y+h, x:x+w]
                    manual_img = faceROI.copy()
                    manual_img_copy = manual_img.copy()
                    states.append(manual_img.copy())
                elif not face_box and len(boxes) < 1:
                    print("Please draw bounding box for face")
                elif not face_box and len(boxes) > 1:
                    print("Please press r to restart. Too many face bounding boxes detected")
                elif face_box and not eye_box and len(boxes) == 3:
                    print("Bounding boxes for eyes saved")
                    eye_box = True
                elif face_box and not eye_box and len(boxes) < 3:
                    print("Please draw bounding boxes for both eyes")
                elif face_box and not eye_box and len(boxes) > 3:
                    print("Please press r to restart. Too many eye bounding boxes detected")

                finished = face_box and eye_box

            if k == ord('r'): # Restart
                manual_img = image_blank.copy()
                states.clear()
                states.append(manual_img.copy())
                boxes.clear()

            if k == ord('u'): # Undo
                if len(states) > 1:
                    states.pop()
                    manual_img = np.copy(states[len(states)-1])
                    boxes.pop()

        processed.append([img, [[boxes[0]],[boxes[1],boxes[2]]]])

    cv.destroyAllWindows()
    return processed

def detect_facial_features(images: list):
    print("\nBeginning detection of face and eye bounding boxes")
    manual = []
    ready = []

    # Automatic face detection
    for img in images:
        boxes = _detect_face(img)

        if boxes == None:
            manual.append(img)
        else:
            ready.append([img,boxes])


    print(f"{len(ready)} Images processed automatically.\n{len(manual)} Images need manual detection\n")

    if len(manual) > 0:
        userInput = input("Press m to run manual detection or c to continue with auto processed images only >> ")

        while userInput != 'c' and userInput != 'm':
            userInput = input("Not a valid choice: Please enter either 'c' or 'm' >> ")

        if userInput == 'm':
            # Manual face detection
            processed = _manual_detect_faces(manual)

            for pair in processed:
                ready.append(pair)

    print("Facial feature detection complete")
    return ready

def _rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    Retrieved from https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
    """

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

def _rotate(img, boxes):

    # Find Left and Right Eye Boxes
    l_eye,r_eye = boxes[1]
    
    if r_eye[0] < l_eye[0]:
        r_eye, l_eye = l_eye, r_eye

    l_eye_center = (boxes[0][0][0] + l_eye[0] + l_eye[2]//2, boxes[0][0][1] + l_eye[1] + l_eye[3]//2)
    r_eye_center = (boxes[0][0][0] + r_eye[0] + r_eye[2]//2, boxes[0][0][1] + r_eye[1] + r_eye[3]//2)

    # Calculate angle of rotation
    theta_rad = math.atan2(r_eye_center[1] - l_eye_center[1], r_eye_center[0]-l_eye_center[0])
    theta_deg = math.degrees(theta_rad)
    # print("Theta_deg:",theta_deg)


    # center = ((r_eye[0]+l_eye[0])//2,(r_eye[1]+l_eye[1]))

    # Rotate Image
    return _rotate_image(img, theta_deg)

def rotate_images(images: list):
    '''Takes a list of images paired with bounding boxes and rotates them to align bounding boxes of eyes'''

    print("\nBeginning rotation of images based on eye bounding boxes")
    rotated_images = []

    for pair in images:
        rotated_images.append(_rotate(pair[0],pair[1]))

    print("Image rotation complete")

    return rotated_images

def _crop_and_scale_face(img, boxes):

    # Find Left and Right Eye Boxes
    print(f"Boxes:{boxes}")
    l_eye,r_eye = boxes[1]
    print(f"Left Eye:{l_eye}")
    print(f"Right Eye:{r_eye}")
    
    if r_eye[0] < l_eye[0]:
        r_eye, l_eye = l_eye, r_eye

    l_eye_center = (boxes[0][0][0] + l_eye[0] + l_eye[2]//2, boxes[0][0][1] + l_eye[1] + l_eye[3]//2)
    r_eye_center = (boxes[0][0][0] + r_eye[0] + r_eye[2]//2, boxes[0][0][1] + r_eye[1] + r_eye[3]//2)

    #print(f"Left Eye Center = {l_eye_center}")
    #print(f"Right Eye Center = {r_eye_center}")

    interocular_distance = ((r_eye_center + (r_eye[2]//2,r_eye[3]//2))[0] - (l_eye_center + (-(l_eye[2]//2),l_eye[3]//2))[0])
    #print("Interocular Distance = ",interocular_distance)
    eyemid_x = (r_eye_center[0] + l_eye_center[0]) // 2
    #print("Eyemid X =", eyemid_x)

    x = int(eyemid_x - interocular_distance)
    y = int(l_eye_center[1] - (0.6 * interocular_distance))

    xf = int(eyemid_x + interocular_distance)
    yf = int(l_eye_center[1] + 1.5*interocular_distance)

    #print(f"X = {x}, Y = {y}, XF = {xf}, YF = {yf}")

    cropped_img = img[y:yf,x:xf]
    cropped_img = cv.resize(cropped_img, (70,80))
    return cropped_img

def crop_and_scale_faces(images: list):
    print("\nBeginning the crop and scaling of images")

    processed = []

    for img, boxes in images:
        processed.append(_crop_and_scale_face(img, boxes))

    print("Cropping and scaling complete")
    return processed

def write_images_with_boundingBoxes(images_with_boxes, path, prefix):
    print(f"\nSaving images with bounding boxes to {path}")

    for i,pair in enumerate(images_with_boxes):
        file_path = path + f'{prefix}_{i}.jpg'
        img = pair[0]
        boxes = pair[1]

        cp = img.copy()

        fx,fy,fw,fh = boxes[0][0]

        cv.rectangle(cp,(fx,fy),(fx+fw,fy+fh),(255,255,255),2)

        for i in range(2):
            ex,ey,ew,eh = boxes[1][i]

            ex = ex + fx
            ey = ey + fy

            cv.rectangle(cp,(ex,ey),(ex+ew,ey+eh),(255,255,255),2)

        cv.imwrite(file_path, cp)

    print("Images saved")

def write_images(images, path, prefix):
    ''' Saves all passed images to the /data/processed folder'''

    print(f"\nSaving images to {path}")

    for i,img in enumerate(images):
        file_path = path + f'{prefix}_{i}.jpg'
        cv.imwrite(file_path, img)
    
    print("Images saved")

def main():
    print("* * * * * CSC 515 - Critical Thinkning 3 * * * * *")

    # Load Data
    images = load_data()
    
    # Convert to Grayscale
    grayscale_conversion(images)
    write_images(images, 'data/grayscale/', 'face')

    # Initial Face Detection
    images_and_boxes = detect_facial_features(images)
    write_images_with_boundingBoxes(images_and_boxes, 'data/init_face_detect/', 'face')

    # Rotation
    rotated_images = rotate_images(images_and_boxes)
    write_images(rotated_images, 'data/rotated_faces/', 'face')

    # Redetect faces
    images_and_boxes = detect_facial_features(rotated_images)
    write_images_with_boundingBoxes(images_and_boxes, 'data/final_face_detect/', 'face')

    # Crop and scale
    processed_images = crop_and_scale_faces(images_and_boxes)

    # Save processed images
    write_images(processed_images, 'data/processed/', 'face')


if __name__ == "__main__":
    # Changes the working directory to whatever the parent directory of the script executing the code is
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    main()