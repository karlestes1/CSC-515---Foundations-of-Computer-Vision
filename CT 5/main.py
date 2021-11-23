"""
Karl Estes
CSC 515 Critical Thinking 5 : Morphology Operations for Handwritten Text Enhancement
Created: November 11th, 2021
Due: November 14th, 2021

Asignment Prompt
----------------
Many companies have thousands of documents to process, analyze, and transform in order to carry out day-to-day operations. 
Some of these documents might contain handwritten text. Handwritten text can be found in handwritten notes, memos, whiteboards, 
medical records, historical documents, text input by stylus, etc. Therefore, a complete Optical Character Recognition (OCR) 
solution has to include support for recognizing handwritten text in images.

Handwriting recognition (HWR), also known as Handwritten Text Recognition (HTR), is the ability of a computer to receive and 
interpret intelligible handwritten input. A handwriting recognition system handles formatting, performs correct segmentation into characters, 
and finds the most plausible words. Due to the variety of handwriting styles, orientations, character sizes, and distortions, handwriting 
recognition is a challenging problem. Thus, to reduce the rejection rates during the matching stage, handwritten text has to be enhanced 
prior to matching. Enhancement can be achieved using morphological image processing.

Acquire a scanned input image of cursive handwritten text on a sticky note. In OpenCV, write algorithms to process the image using 
morphological operations (dilation, erosion, opening, and closing).

Comment Anchors
---------------
I am using the Comment Anchors extension for Visual Studio Code which utilizes specific keywords
to allow for quick navigation around the file by creating sections and anchor points. Any use
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with 
this extension. To trigger these keywords, they must be typed in all caps. 
"""
import cv2 as cv
import sys
import getopt
import matplotlib.pyplot as plt

def binarize(img):
    '''Performs Otsu's thresholding after Gaussian filtering'''
    global thorough
    kernel = (5,5)
    blur = cv.GaussianBlur(img,kernel,1)

    if thorough:
        print(f"Image underwent Gaussian Blur with kernel={kernel}")
        cv.imshow("GB Image", blur)
        print("Press Any key to continue...\n")
        k = cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imwrite(f"processed_images/gaussian_blur_{kernel}.jpg", blur)
        print("Saved copy of blurred image to processed_images/")  

    ret,img_binarized = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    if thorough:
        print(f"Image underwent Otsu's Binarization")
        cv.imshow("Binarized Image", img_binarized)
        print("Press Any key to continue...\n")
        k = cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imwrite("processed_images/binarized.jpg", img_binarized)
        print("Saved copy of binarized image to processed_images/")  

    return img_binarized

def apply_morphological_functions(img, k_size):
    '''Applies erosion, dilation, opening, and closing to the provided image with a rectangular kernel'''
    global thorough

    # Get structuring element
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (k_size,k_size))

    if thorough:
        print("Created structuring element")

    erosion = cv.erode(img, kernel, iterations=1)

    if thorough:
        print(f"Image underwent erosion")
        cv.imshow("Eroded Image", erosion)
        print("Press Any key to continue...\n")
        k = cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imwrite(f"processed_images/erosion_{k_size}.jpg", erosion)
        print("Saved copy of eroded image to processed_images/")  

    dilation = cv.dilate(img, kernel, iterations=1)

    if thorough:
        print(f"Image underwent dilation")
        cv.imshow("Dilated Image", dilation)
        print("Press Any key to continue...\n")
        k = cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imwrite(f"processed_images/dilations_{k_size}.jpg", dilation)
        print("Saved copy of dilated image to processed_images/")  

    opened = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    if thorough:
        print(f"Image underwent opening")
        cv.imshow("Opened Image", opened)
        print("Press Any key to continue...\n")
        k = cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imwrite(f"processed_images/opened_{k_size}.jpg", opened)
        print("Saved copy of opened image to processed_images/")

    closed = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

    if thorough:
        print(f"Image underwent closing")
        cv.imshow("Closed Image", closed)
        print("Press Any key to continue...\n")
        k = cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imwrite(f"processed_images/closed_{k_size}.jpg", closed)
        print("Saved copy of closed image to processed_images/")  

    return erosion,dilation,opened,closed

def main():
    global thorough
    thorough = False
    img_path = ""
    k_size = 0

    # Parse command line arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hk:i:", ['image_path=','kernel=','thorough'])
    except getopt.GetoptError:
        print("GETOPT ERROR: main.py -i <path_to_image> -k <kernel_size: int> [--thorough: Optional argument for debugging messages]")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print("main.py -i <path_to_image> -k <kernel_size: int> [--thorough: Optional argument for debugging messages]")
            sys.exit()
        elif opt in ('-i', '--image_path'):
            img_path = arg
        elif opt in ('-k','--kernel'):
            try:
                k_size = int(arg)
            except ValueError:
                print(f"Provided kernel size argument {arg} not convertible to integer")
                sys.exit(3)
        elif opt == '--thorough':
            thorough = True

    if img_path == "" or k_size == 0:
        print("MISSING ARG: main.py -i <path_to_image> -k <kernel_size: int> [--thorough: Optional argument for debugging messages]")
        sys.exit(2)
        
    # Load Image
    if thorough:
        print(f"Loading image from path: {img_path}")
    img = cv.imread(img_path, 1)

    print("Imaged loaded")

    if thorough:
        print("Display original iamge")
        cv.imshow("Original Image", img)
        print("Press Any key to continue...\n")
        k = cv.waitKey(0)
        cv.destroyAllWindows()    


    # Convert to Grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if thorough:
        print("Image converted to grayscale")
        cv.imshow("Grayscale Image", gray)
        print("Press Any key to continue...\n")
        cv.imwrite("processed_images/grayscale.jpg", gray)
        print("Saved copy of grayscale image to processed_images/")
        k = cv.waitKey(0)
        cv.destroyAllWindows()    

    # Binarize
    binarized = binarize(gray)

    # Apply Morphological Transformations
    morphed_images = apply_morphological_functions(binarized, k_size)

    # Graph Resulting Images
    fig, axs = plt.subplots(2,2)
    fig.suptitle(f"Morphological Transformations - Kernel=({k_size},{k_size})")

    for i in range(2):
        for j in range(2):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

    axs[0][0].set_title("Erosion")
    axs[0][0].imshow(morphed_images[0], cmap='gray')
    axs[0][1].set_title("Dilation")
    axs[0][1].imshow(morphed_images[1], cmap='gray')
    axs[1][0].set_title("Opening")
    axs[1][0].imshow(morphed_images[2], cmap='gray')
    axs[1][1].set_title("CLosing")
    axs[1][1].imshow(morphed_images[3], cmap='gray')

    plt.show()



    

if __name__ == "__main__":
    main()