import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])# bottom of the image
    y2 = int(y1*3/5)         # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    # add more weight to longer lines
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line  = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([(200, height), (1100, height), (550, 250)])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [triangle], 255)
    masked_image=cv2.bitwise_and(image,mask)#bit operation
    return masked_image

#def display_lines(image,lines):
#  line_image=np.zeros_like(image)
#  if lines is not None:
#   for line in lines:
#      x1,y1,x2,y2=line.reshape(4)
#      cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
#  return line_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


# Load the image
image_path = "/content/center_2023_08_28_16_07_35_438.jpg"
lane_image = cv2.imread(image_path)

# Perform canny edge detection
edges = canny(lane_image)
cropped_image=region_of_interest(edges)
#Applying Hough transformation
lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=4)
averaged_line=average_slope_intercept(lane_image,lines)
line_image=display_lines(lane_image,averaged_line) #creates the Hough line in a blank paper


combo_image=cv2.addWeighted(lane_image,1,line_image,1,1) #Superimpose hough lines of line_image on lane_image)

# Display the resulting image
cv2_imshow(combo_image)  # Display the result using cv2_imshow
