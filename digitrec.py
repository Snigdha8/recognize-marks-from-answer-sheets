#make a predictionfor a new image.
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
    img = load_img(filename, grayscale = True, target_size = (28, 28))# convert to array
    cvimg=np.array(img)
    cv2.imwrite("out.jpg",cvimg)
    #print(type(img))
    img = img_to_array(img)
    #plt.imshow(img/255.)
    #plt.show()
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    return img

# load an image and predict the class
w_th=5
def run_example(ct):
    pth = "C:\\Users\\SONY\\Desktop\\IVP\\IVP\\mini project\\conn_comp\\images_s%d\\" % (ct)
    model = load_model(r'C:\Users\SONY\PycharmProjects\mini_project\mnist_keras_cnn_model.h5')  # predict the class
    # i1=img[0:img.shape[0],0:int(img.shape[1]/2)]
    # images=load_images_from_folder(pth)
    q_no=""
    marks=""
    left_dig=0
    point=0
    right_dig=0
    for filename in os.listdir(pth):
        locn=filename[5:-6]
        if locn!=q_no:
            if q_no=="":
                q_no=locn
            else:
                if point==1 and right_dig==0:
                    marks=marks[:-1]
                print(q_no+" -> "+marks)
                left_dig = 0
                point=0
                right_dig=0
                print()
                marks=""
                q_no=locn
        print(filename)
        s="C:\\Users\\SONY\\Desktop\\IVP\\IVP\\mini project\\conn_comp\\images_s%d\\%s" % (ct,filename)
        #print(s)
        img = load_image(s)  # load model
        #print(type(img))
        #print(np.sum(img==1))
        if np.sum(img == 1) > w_th:
            digit = model.predict_classes(img)
            print(digit[0])
            marks=marks+str(digit[0])
            if left_dig==0:
                left_dig=1
            elif point==1:
                right_dig=1
        else:
            if point==0 and left_dig==1:
                point=1
                print(".")
                marks = marks + '.'
    if point == 1 and right_dig == 0:
        marks = marks[:-1]
    print(q_no + " -> " + marks)
    print()
# entry point, run the example
run_example(3)
print("Done!!")
cv2.waitKey(0)
cv2.destroyAllWindows()