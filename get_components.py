import stat

from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
import os
import shutil

'''def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)

'''
def empty_folder(pth):
    shutil.rmtree(pth)
    os.makedirs(pth)

def load_images_from_folder(folder):
    images = []
    name = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            path=os.path.basename(filename)
            name.append(path)
    return images,name


#fname=r'C:\Users\SONY\Desktop\IVP\mini project\new_cropped\images_s2\file_11_1.jpg'
ct=0
while ct<8:
    ct=ct+1
    folder = "C:\\Users\\SONY\\Desktop\\IVP\\IVP\\mini project\\new_cropped\\images_s%d\\"%(ct)
    str = "C:/Users/SONY/Desktop/IVP/IVP/mini project/conn_comp/images_s%d/" % (ct)
    # pth=os.listdir("C:/Users/SONY/Desktop/IVP/mini project/conn_comp/images_s%d/")
    empty_folder(str)
    num=1
    images, name = load_images_from_folder(folder)
    glob=0
    for im1 in images:
        name1=name[glob]
        name1=name1[:-4]
        #print(name1)
        blur_radius = 0.4
        threshold = 50
        #img = Image.open(fname).convert('L')
        img=im1.copy()
        #plt.imshow(img)
        #plt.show()
        img=img[10:img.shape[0]-10,10:img.shape[1]-10]
        ori=img.copy()
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #thresh,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        conn=4
        img1=cv2.bitwise_not(thresh)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(img1, kernel, iterations=1)
        labeled, nr_objects = ndimage.label(img1)#connected components
        cnt=1
        num=1
        crop=15
        saved=0
        #print(labeled)
        #plt.imshow(labeled)
        #plt.show()
        print("Number of objects is {}".format(nr_objects))
        max_area=0
        bin_count = [0]*(nr_objects+1)
        #print(bin_count)
        for i in range(labeled.shape[1]):
            for j in range(labeled.shape[0]):
                bin_count[labeled[j][i]]=bin_count[labeled[j][i]]+1
        print(bin_count)
        bin_count.pop(0)
        mx=0
        if nr_objects>0:
            mx=max(bin_count)
        it=0
        while it<nr_objects:
            if bin_count[it]==mx:
                new_it=it#iterator corr to mx label
            it=it+1
        if mx>50:
            pixel=ori[0][0]
            for i in range(labeled.shape[1]):
                for j in range(labeled.shape[0]):
                    if labeled[j][i]==new_it+1:
                        pixel=ori[j][i]
            print("Pixel is ")
            print(pixel)
            cnt=1
            while cnt<=nr_objects:
                blank_img=np.zeros(shape=[img1.shape[0],img1.shape[1],3],dtype=np.uint8)
                flag=0
                area=0
                for i in range(labeled.shape[1]):
                    for j in range(labeled.shape[0]):
                        new_pixel=ori[j][i]
                        if saved!=0 and labeled[j][i]==saved:
                            blank_img[j][i]=255#ori[j][i]
                            #print(ori[j][i])
                            labeled[j][i]=0
                            area=area+1
                            if i>crop and j>crop and i<labeled.shape[1]-crop and j<labeled.shape[0]-crop:
                                flag=1;
                        elif saved==0 and labeled[j][i]!=0:
                            saved=labeled[j][i]
                            blank_img[j][i] = 255#ori[j][i]
                            labeled[j][i] = 0
                            area=area+1
                            if i>crop and j>crop and i<labeled.shape[1]-crop and j<labeled.shape[0]-crop:
                                flag=1
                filename = "C:\\Users\\SONY\\Desktop\\IVP\\IVP\\mini project\\conn_comp\\images_s%d\\%s_%d.jpg" % (ct,name1,num)
                saved=0
                #kernel = np.ones((3, 3), np.uint8)
                #mk = cv2.dilate(blank_img, kernel, iterations=1)
                if area>10 and flag==1:
                    print(filename)
                    cv2.imwrite(filename,blank_img)
                    num = num + 1
                print(area)
                cnt=cnt+1
        glob=glob+1
