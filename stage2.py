import numpy as np
import cv2 as cv
import math
from scipy import ndimage
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def rotate_img(im):
    img_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    img_edges = cv.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    angles = []
    for x1, y1, x2, y2 in lines[0]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    median_angle = np.median(angles)
    img_rotated = ndimage.rotate(im, median_angle)
    print("Angle is {}".format(median_angle))
    return img_rotated

def pre_process_image(img,skip_dilate=False):
    proc = cv.GaussianBlur(img.copy(), (9, 9), 0)
    proc = cv.adaptiveThreshold(proc, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    proc = cv.bitwise_not(proc, proc)

    if not skip_dilate:
      kernel = np.ones((3, 3), np.uint8)#np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
      proc = cv.dilate(proc, kernel)

    return proc


def invert_img(im,boxes):
    boxes.sort(key=lambda x: x[1])
    ln=len(boxes)
    diff=boxes[0][1]-boxes[6][1]
    if diff<0:
        diff=-diff
    im1=im.copy()
    print("boxes[0][1] "+str(boxes[0][1]))
    y1=boxes[0][1]-50
    y2=boxes[ln-1][1]+boxes[ln-1][3]+50
    if y1<0:
        y1=0
        print("here")
    if y2>im.shape[0]:
        y2=im.shape[0]
    im1=im[y1:y2,0:im.shape[1]]
    y=0
    if diff>50:
        im1=cv.rotate(im1,cv.ROTATE_180)#upside down img
    print("diff "+str(diff))
    #im2=im1[0:y, 0:im1.shape[1]]
    return im1

def show_grid(im,num):
    cnt = 0
    lim = 140
    x = im.shape[0]
    y = im.shape[1]
    #while cnt < 78 and lim>0:
    #lim = lim - 1
    #print("lim is ",lim)
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    #ret, thresh = cv.threshold(imgray, lim, 255, 0)
    mblur = cv.medianBlur(imgray,5)
    th = cv.adaptiveThreshold(mblur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    #print("threshold is "+str(th))
    kernel = np.ones((7, 7), np.uint8)
    mask = cv.erode(th, kernel, iterations=1)#dilate.. lines thicker
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #cv.imshow("thressshhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh",thresh)
    #cv.imwrite("thresh.jpg",mask)

    boxes = []
    output = im.copy()
    thickness = 2
    color = (255, 0, 0)
    cnt = 0
    for contour in contours:
        box = cv.boundingRect(contour)
        h = box[3]
        start_point = (box[0], box[1])
        end_point = (box[0] + box[2], box[1] + box[3])
        if box[2] > 80 and box[3] > 65 and box[2] < 430 and box[3] < 190 and box[2] > box[3]:
            #output = cv.rectangle(output, start_point, end_point, color, thickness)
            boxes.append(box)
            cnt = cnt + 1
        #elif box[2]>box[3] and box[2]>40:
            #print(box)
    #cv.imshow("thressshhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh", output)
    cv.imwrite("thre.jpg", output)
    output=invert_img(output,boxes)
    return output

folder = "C:\\Users\\SONY\\Desktop\\IVP\\mini project\\input_images\\"
num=1
images = load_images_from_folder(folder)
for im1 in images:
    #im1 = cv.imread(r"C:\Users\SONY\Downloads\Answersheets\Answersheets\IMG_20191018_155019.jpg",-1)
    im1=rotate_img(im1)
    y = im1.shape[0]
    x = im1.shape[1]
    #cv.imwrite("rotated.jpg",im1)
    #filename = "C:\\Users\\SONY\\Desktop\\IVP\\mini project\\new_images\\images_s%d.jpg" % (num)
    #cv.imwrite(filename,im1)
    #print(x)
    #print(y)
    #cv.imshow("original-1",crop_img1)
    if y<x:
        im1=cv.transpose(im1)
        im1=cv.flip(im1,flipCode=0)
    cv.imwrite("out1.jpg",im1)
    im1 = im1[450:im1.shape[0]-450, 0:im1.shape[1]]
    out1 = show_grid(im1,1)
    filename = "C:\\Users\\SONY\\Desktop\\IVP\\mini project\\new_images\\images_s%d.jpg" % (num)
    cv.imwrite(filename,out1)
    #print(num)
    num=num+1
#cv.imshow("out-1",out1)

print("Done!!! ")

cv.waitKey(0)
cv.destroyAllWindows()