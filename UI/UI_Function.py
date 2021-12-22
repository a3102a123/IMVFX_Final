
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QImage, QPixmap
from UI.MainGUI import Ui_MainGUI
import sys
import os
import copy
import cv2
import numpy as np
import math

class GUI(Ui_MainGUI):
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.MainWindow = QtWidgets.QMainWindow()
        self.draing_flag = False
        self.frame = Image()
        self.result = Image()

        self.setupUi(self.MainWindow)

    def run_app(self):
        self.MainWindow.show()
        sys.exit(self.app.exec_())
    
    # display 
    ###########################################
    # refresh/redisplay the GUI
    def display(self):
        self.display_ImageDisplayer(self.ImageDisplayer , self.frame)
        self.display_ImageDisplayer(self.ImageDisplayer_2 , self.result)
        self.app.processEvents()

    def display_ImageDisplayer(self,displayer,img_obj):
        if type(img_obj.image) == type(None):
            return
        Img = img_obj.QImage()
        displayer.setPixmap(QPixmap.fromImage(Img))
        displayer.update()

    # setting
    ###########################################
    # set the input image in cv2 format & display it on displyer
    def set_frame(self,img):
        self.frame.set_image(img)
        self.display()

    def set_result(self,img):
        self.result.set_image(img)
        self.display()

    # value
    ###########################################
    def get_Displayer2Frame_ratio(self):
        width = self.ImageDisplayer.frameGeometry().width()
        height = self.ImageDisplayer.frameGeometry().height()
        img_h , img_w , img_c = self.frame.image.shape
        ratio_w = img_w / width
        ratio_h = img_h / height
        return ratio_w , ratio_h
    def get_alpha(self):
        num = self.Alpha.value()
        interval = self.Alpha.pageStep()
        return num / interval

        
class Image():
    def __init__(self,image_path=None):
        # preserve the original image to recover the modified image 
        self.ori_image = None
        # image will be used to draw or modify
        self.image = None
        # the bounding box range drew on image
        self.boundingBox = Rect(0,0,0,0)
        # the first elemnet is the refence of self.boundingbox, 
        # other element is object bounding box overlap with target
        self.boundingBox_list = [self.boundingBox]
        if(image_path != None):
            img = cv2.imread(image_path)
            self.set_image(img)
            
    def set_image(self,img):
        if(isinstance(img,Image)):
            self.ori_image = copy.deepcopy(img.ori_image)
            self.image = copy.deepcopy(img.ori_image)
            self.boundingBox = copy.deepcopy(img.boundingBox)
            self.boundingBox_list = copy.deepcopy(img.boundingBox_list)
        else:
            self.ori_image = copy.deepcopy(img)
            self.image = copy.deepcopy(img)
    
    # append bounding box of other object to prevent remaining some edge influecing result
    def set_boundingBox_list(self,list,append_pixel = 0):
        # initial boundingBox list 
        self.boundingBox_list.clear()
        self.boundingBox_list.append(self.boundingBox)
        for i,bbox in enumerate(list):
            # assume the first element of input is the target 
            if i == 0:
                self.boundingBox.set(bbox[0] - append_pixel,bbox[1] - append_pixel,bbox[2] + append_pixel,bbox[3] + append_pixel)
            else:
                temp = Rect(bbox[0] - append_pixel,bbox[1] - append_pixel,bbox[2] + append_pixel,bbox[3] + append_pixel)
                self.boundingBox_list.append(temp)

    def print_boundingBox_list(self):
        print("The length of bounding box : ",len(self.boundingBox_list))
        for bbox in self.boundingBox_list:
            bbox.print()

    def QImage(self):
        h, w, c = self.image.shape
        bytesPerline = c * w
        Img = QImage(self.image.data, w, h, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        return Img
    
    def reset(self):
        self.boundingBox.set(0,0,0,0)
        self.image = copy.deepcopy(self.ori_image)
        self.boundingBox_list.clear()
        self.boundingBox_list.append(self.boundingBox)

    def clear_drawing(self):
        self.image = copy.deepcopy(self.ori_image)

    def draw_boundingBox(self,color):
        for bbox in self.boundingBox_list:
            p1 = (int(bbox.x0),int(bbox.y0))
            p2 = (int(bbox.x1),int(bbox.y1))
            cv2.rectangle(self.image,p1,p2,color,thickness=2)
    
    # return the bigger bounding box combined with all bounding box in list
    def get_combined_boundingBox(self):
        x0,y0,x1,y1 = sys.maxsize,sys.maxsize,-1,-1
        for bbox in self.boundingBox_list:
            b_x0,b_y0,b_x1,b_y1 = bbox.get_range()
            x0,y0,x1,y1 = min(b_x0,x0),min(b_y0,y0),max(b_x1,x1),max(b_y1,y1)
        return x0,y0,x1,y1

    # change the content in bounding box to img
    # (the size of img is the bigger bounding box combined with all bounding box in list)
    def set_boundingBox_image(self,img):
        Img = copy.deepcopy(self.ori_image)
        # find the bounding image beginning position in origin image
        img_x0,img_y0,img_x1,img_y1 = self.get_combined_boundingBox()
        for bbox in self.boundingBox_list:
            x0,y0,x1,y1 = bbox.get_range()
            begin_w = x0 - img_x0
            begin_h = y0 - img_y0
            width = x1 - x0
            height = y1 - y0
            Img[y0:y0 + height,x0:x0 + width] = img[begin_h:begin_h + height,begin_w:begin_w + width]
            self.set_image(Img)
        return Img
    
    # combining all bounding box to a big one and return the image of it
    def get_boundingBox_image(self):
        x0,y0,x1,y1 = self.get_combined_boundingBox()
        # append a little of bounding box to prevent resize rounding problem
        Reslut_Img = copy.deepcopy(self.ori_image[y0:(y1 + 5),x0:(x1 + 5)])
        return Reslut_Img

    # apply the alpha blenging with gray scale image in bounding box
    def alpha_blending_boundingBox(self,alpha,gamma = 0):
        Img = self.get_boundingBox_image()
        if len(Img) == 0:
            print("There is no bounding box for blending")
            return
        gray_image = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
        gray_image = cv2.cvtColor(gray_image,cv2.COLOR_GRAY2BGR)
        Blending_Img = cv2.addWeighted(Img,alpha,gray_image,1 - alpha,gamma)
        Reslut_Img = self.set_boundingBox_image(Blending_Img)
        return Reslut_Img

    # return the image which the bounding area is cutted out 
    def cut_boundingBox(self):
        Reslut_Img = copy.deepcopy(self.ori_image)
        for bbox in self.boundingBox_list:
            x0,y0,x1,y1 = bbox.get_range()
            Reslut_Img[y0:y1,x0:x1] = (255,255,255)
        return Reslut_Img

    # return the mask of bounding box in original image scale
    def masking_boundingBox(self):
        h,w,c = self.ori_image.shape
        mask = np.zeros((h,w), np.uint8)
        for bbox in self.boundingBox_list:
            x0,y0,x1,y1 = bbox.get_range()
            mask[y0:y1,x0:x1] = 255
        return mask

    # save the cut out image
    def save_cut(self,path):
        Img = self.cut_boundingBox()
        cv2.imwrite(path,Img)
        print("Creat cutted image saved in : ",path)


    # save the bounding area mask
    def save_mask(self,path):
        mask = self.masking_boundingBox()
        cv2.imwrite(path,mask)
        print("Creat mask saved in : ",path)

    # save the drew iamge result 
    def save(self,path):
        cv2.imwrite(path,self.image)

    # return the resized Image object
    def get_resize_Image(self,width,height):
        h,w,c = self.ori_image.shape
        w_scale = width / w
        h_scale = height / h
        img = copy.deepcopy(self.ori_image)
        img = cv2.resize(img,(width,height),interpolation=cv2.INTER_CUBIC)
        img_obj = Image()
        img_obj.set_image(img)
        for i,bbox in enumerate(self.boundingBox_list):
            x0,y0,x1,y1 = bbox.get_range()
            if i == 0:
                img_obj.boundingBox.set(x0*w_scale,y0*h_scale,x1*w_scale,y1*h_scale)
            else:
                temp = Rect(x0*w_scale,y0*h_scale,x1*w_scale,y1*h_scale)
                img_obj.boundingBox_list.append(temp)
        return img_obj

    # return the copy instance of Image
    def get_Image(self):
        new_Image = Image()
        new_Image.set_image(self)
        return new_Image


# rectangle contain 4 element to represent the range of rectangle
# can be used to draw bounding box 
class Rect():
    def __init__(self):
        self.init()
    def __init__(self,x0,y0,x1,y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
    def init(self):
        self.x0 = 0
        self.x1 = 0
        self.y0 = 0
        self.y1 = 0
    def set_rect(self,rect):
        self.x0 = rect.x0
        self.y0 = rect.y0
        self.x1 = rect.x1
        self.y1 = rect.y1
    def set(self,x0,y0,x1,y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
    def set_p0(self,x0,y0):
        self.x0 = x0
        self.y0 = y0
    def set_p1(self,x1,y1):
        self.x1 = x1
        self.y1 = y1
    def ceil(self):
        self.x0 = math.ceil(self.x0)
        self.y0 = math.ceil(self.y0)
        self.x1 = math.ceil(self.x1)
        self.y1 = math.ceil(self.y1)
    # return the p0 p1 (p0 maybe is the point on left-top or right-top )
    def get(self):
        return self.x0,self.y0,self.x1,self.y1
    # return the list contain [p0,p1]
    def get_list(self):
        x0 = int(min(self.x0,self.x1))
        y0 = int(min(self.y0,self.y1))
        x1 = int(max(self.x0,self.x1))
        y1 = int(max(self.y0,self.y1))
        return [x0,y0,x1,y1]
    # return the bounding box range for cutting image
    def get_range(self):
        x0 = int(min(self.x0,self.x1))
        y0 = int(min(self.y0,self.y1))
        x1 = int(max(self.x0,self.x1))
        y1 = int(max(self.y0,self.y1))
        return x0,y0,x1,y1
    def print(self):
        print("[ ",self.x0," , ",self.y0," , ",self.x1," , ",self.y1," ]")
