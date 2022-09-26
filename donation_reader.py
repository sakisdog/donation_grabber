import cv2 
import pytesseract
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('video', action='store', type=str, help='The video to process.')
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
count = 0
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # Start coordinate, here (x, y) 
    # x is the horizontal axis and y is the vertical
    start_point = (20, 240)
    
    # Ending coordinate, here (220, 220)
    # represents the bottom right corner of rectangle
    end_point = (260, 480)
    
    # Blue color in BGR
    color = (255, 0, 0)
    # Display the resulting frame
    #frame = cv2.rectangle(frame, start_point, end_point, color, 2)

    crop = frame[320:480,0:280]
    # Load image, convert to HSV format, define lower/upper ranges, and perform
    # color segmentation to create a binary mask
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #lower = np.array([0, 0, 218])
    #upper = np.array([255, 100, 255])
    #crop = cv2.inRange(hsv, lower, upper)

    crop = cv2.resize(crop,None,fx=2.5,fy=2.5)
    #crop = get_grayscale(crop)
    #crop = thresholding(crop)
    #crop1 = opening(crop1)
    #cv2.imwrite("frame.jpg",crop1)
    cv2.imshow('Preview',crop)
    #cv2.imshow('After Threshold',crop1)
    
    custom_config = r'-l grc+eng --psm 6'
    ocr_txt = pytesseract.image_to_string(crop, config=custom_config)
    #print(ocr_txt)
    match = re.search(r'€[0-9]*', ocr_txt)
    if match:
        #print("DING DING DING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(match.group())

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
    count += 180 # i.e. at 30 fps, this advances one second
    cap.set(cv2.CAP_PROP_POS_FRAMES, count)
  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()