import numpy as np
import cv2 as cv
import os


def show_grid(im,num):
    cnt = 0
    lim = 155
    x = im.shape[0]
    y = im.shape[1]
    while cnt < 78 and lim>0:
        lim = lim - 1
        #print("lim is ",lim)
        imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        mblur = cv.medianBlur(imgray, 5)
        th = cv.adaptiveThreshold(mblur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        #ret, thresh = cv.threshold(imgray, lim, 255, 0)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv.erode(th, kernel, iterations=1)
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # cv.imshow("thressshhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh",thresh)
        # cv.imwrite("threshssssssssssssssssh.jpg",mask)
        # print(im.shape)

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
            if box[2] > 165 and box[3] > 65 and box[2] < 430 and box[3] < 190 and box[2] > box[3]:#box[2] > 35 and box[3] > 35 and box[2] < 102 and box[3] < 70 and box[2] > box[3]:
                output = cv.rectangle(output, start_point, end_point, color, thickness)
                print(box)
                cnt = cnt + 1
    cv.imwrite("out.jpg",output)
    list_78 = []
    for contour in contours:
        box = cv.boundingRect(contour)
        if box[2] > 165 and box[3] > 65 and box[2] < 430 and box[3] < 190 and box[2] > box[3]:#box[2] > 35 and box[3] > 35 and box[2] < 102 and box[3] < 70 and box[2] > box[3]:
            list_78.append(box)
    #print("78 tuple")
    print(cnt)
    list_78.sort(key = lambda x:x[1])
    i=0
    qno = 0
    while i<77:
        list_tmp = []
        subqno=1
        for j in range(7):
            list_tmp.append(list_78[i])
            i=i+1
        if i>7:
            index=0
            list_tmp.sort(key = lambda x:x[0])
            list_tmp.pop(0)
            for k in range(6):
                crop_img = mask[list_tmp[index][1]:list_tmp[index][1] + list_tmp[index][3], list_tmp[index][0]:list_tmp[index][0] + list_tmp[index][2]]
                crop_img1 = im[list_tmp[index][1]:list_tmp[index][1] + list_tmp[index][3], list_tmp[index][0]:list_tmp[index][0] + list_tmp[index][2]]
                kernel = np.ones((3, 3), np.uint8)
                mask1 = cv.dilate(crop_img, kernel, iterations=1)
                c1, h1 = cv.findContours(mask1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                index=index+1
                if len(c1) > 1:
                    filename = "C:\\Users\\SONY\\Desktop\\IVP\\IVP\\mini project\\new_cropped\\images_s%d\\file_%d_%d.jpg" %(num, qno, subqno)
                    cv.imwrite(filename, crop_img1)
                subqno=subqno+1
            #print(list_tmp)
        qno = qno + 1
    crop_img = mask[list_78[77][1]:list_78[77][1] + list_78[77][3], list_78[77][0]:list_78[77][0] + list_78[77][2]]
    crop_img1 = im[list_78[77][1]:list_78[77][1] + list_78[77][3], list_78[77][0]:list_78[77][0] + list_78[77][2]]
    kernel = np.ones((3, 3), np.uint8)
    mask1 = cv.dilate(crop_img, kernel, iterations=1)
    filename = "C:\\Users\\SONY\\Desktop\\IVP\\IVP\\mini project\\new_cropped\\images_s%d\\file_%d_%d.jpg" % (num, qno, 1)
    cv.imwrite(filename, crop_img1)
    #print(list_78)
    return output


im1 = cv.imread(r"C:\Users\SONY\Desktop\IVP\IVP\mini project\new_images\images_s3.jpg",-1)
med_blur=cv.medianBlur(im1,5)
cv.imwrite("median.jpg",med_blur)
y = im1.shape[0]
x = im1.shape[1]
#crop_img1 = im1[165:y, 0:x]
#cv.imshow("original-1",crop_img1)
out1 = show_grid(med_blur,3)
cv.imwrite("out-1.jpg",out1)

print("Done!!! ")

cv.waitKey(0)
cv.destroyAllWindows()