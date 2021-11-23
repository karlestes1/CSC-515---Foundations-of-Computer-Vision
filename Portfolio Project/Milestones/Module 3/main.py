"""
Karl Estes
CSC 515 Portfolio Milestone 3
Created: October 25th, 2021
Due: October 31, 2021

Asignment Prompt
----------------
It is time to think more about your upcoming Portfolio Project. In face and object detection, it is often useful to draw on the image. 
Perhaps you would like to put bounding boxes around features or use text to tag objects/people in the image. 

Use a camera to take a picture of yourself facing the frontal.  In OpenCV, draw on the image a red bounding box for your eyes and a green circle around your face.  
Then tag the image with the text “this is me”.

Your submission should be one executable Python file.

File Description
----------------
Python script which loads an image called me.jpeg and allows for basic annotation.
Utilizing OpenCV, red rectangle, green circle, and the text "This is Me" can be added to the image.

The script initially starts in the mode for red rectangle. Click and drag on the screen to place
the shape. Modes can be transitioned between by pressing 'm' on the keyboard. NOTE: The text size
can only be changed via the hardcoded values.

Keypresses are as follows:
    - 'm': cycle mode
    - 's': save image
    - 'r': reload blank image
    - 'u': undo annotations step-by-step
    - 'esc': close program

Comment Anchors
---------------
I am using the Comment Anchors extension for Visual Studio Code which utilizes specific keywords
to allow for quick navigation around the file by creating sections and anchor points. Any use
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with 
this extension. To trigger these keywords, they must be typed in all caps. 
"""

import numpy as np
import cv2 as cv

def main():
    # Globals
    global drawing, mode, img, img_copy, states
    drawing = False # true if mouse is pressed
    mode = 0 # if True, draw rectangle. Press 'm' to toggle to circle
    states = [] # Used for Undo

    img = cv.imread('me.jpeg')
    img_blank = np.copy(img)
    img_copy = np.copy(img)
    states.append(np.copy(img))

    cv.namedWindow('image')
    cv.setMouseCallback('image',draw)
    print("Mode: Rectangle")

    while(1):
        if drawing == True: 
            cv.imshow('image',img_copy)
        else:
            cv.imshow('image',img)
        k = cv.waitKey(1) & 0xFF
        if k == ord('m'): # Change modes
            mode += 1
            if mode == 3:
                mode = 0

            if mode == 0:
                print("Mode: Rectangle")
            elif mode == 1:
                print("Mode: Circle")
            elif mode == 2:
                print("Mode: Text")
        if k == ord('s'): # Save Image
            cv.imwrite('annotated_image.jpeg',img)
            print("Saved image")
        if k == ord('r'): # Reload blank image
            img = img_blank.copy()
            states.clear()
            states.append(np.copy(img))
        if k == ord('u'): # Undo annotations step-by-step
            if len(states) > 1:
                states.pop()
                img = np.copy(states[len(states)-1])

        
        elif k == 27:
            break

    cv.destroyAllWindows()

# mouse callback function
def draw(event,x,y,flags,param):
    global ix, iy, drawing, mode, img, img_copy,states

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        # print(f"Initial Point: ({ix},{iy})")

        if mode == 2:
            cv.putText(img_copy,"This is Me",(x,y),cv.FONT_HERSHEY_TRIPLEX,4,(255,255,255),4,cv.LINE_AA)

    elif event == cv.EVENT_MOUSEMOVE:
        img_copy = np.copy(img)
        if drawing == True:
            if mode == 0:
                cv.rectangle(img_copy,(ix,iy),(x,y),(0,0,255),2)
            elif mode == 1:
                midpoint = int((ix+x)/2), int((iy+y)/2)
                radius = int((max([x,ix]) - min([x,ix]))/2)
                # print(midpoint,radius,ix,iy,x,y)
                cv.circle(img_copy,midpoint,radius,(0,255,0),2)
            elif mode == 2:
                cv.putText(img_copy,"This is Me",(x,y),cv.FONT_HERSHEY_TRIPLEX,4,(255,255,255),4,cv.LINE_AA)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == 0:
            cv.rectangle(img,(ix,iy),(x,y),(0,0,255),2)
        elif mode == 1:
            midpoint = int((ix+x)/2), int((iy+y)/2)
            radius = int((max([x,ix]) - min([x,ix]))/2)
            cv.circle(img,midpoint,radius,(0,255,0),2)
        elif mode == 2:
            cv.putText(img,"This is Me",(x,y),cv.FONT_HERSHEY_TRIPLEX,4,(255,255,255),4,cv.LINE_AA)
            
        states.append(np.copy(img))
        

if __name__ == "__main__":
    main()