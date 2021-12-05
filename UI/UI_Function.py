
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from UI.MainGUI import Ui_MainGUI
import sys
import copy
import cv2

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

    def display_ImageDisplayer(self,displayer,img_obj):
        if type(img_obj.image) == type(None):
            return
        h, w, c = img_obj.image.shape
        Img = img_obj.QImage()
        displayer.resize(w,h)
        displayer.setPixmap(QPixmap.fromImage(Img))
        displayer.show()

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
    def get_alpha(self):
        num = self.Alpha.value()
        interval = self.Alpha.pageStep()
        return num / interval

        
class Image():
    def __init__(self):
        # preserve the original image to recover the modified image 
        self.ori_image = None
        # image will be used to draw or modify
        self.image = None
        # the bouding box range drew on image
        self.boundingBox = Rect(0,0,0,0)
    def set_image(self,img):
        self.ori_image = copy.deepcopy(img)
        self.image = copy.deepcopy(img)
    
    def QImage(self):
        h, w, c = self.image.shape
        bytesPerline = c * w
        Img = QImage(self.image.data, w, h, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        return Img
    
    def reset(self):
        self.boundingBox.set(0,0,0,0)
        self.image = copy.deepcopy(self.ori_image)

    def draw_boundingBox(self,color):
        self.image = copy.deepcopy(self.ori_image)
        p1 = (int(self.boundingBox.x0),int(self.boundingBox.y0))
        p2 = (int(self.boundingBox.x1),int(self.boundingBox.y1))
        cv2.rectangle(self.image,p1,p2,color,thickness=2)
    
    def set_boundingBox_image(self,img):
        x0,y0,x1,y1 = self.boundingBox.get_range()
        Img = copy.deepcopy(self.ori_image)
        Img[y0:y1,x0:x1] = img[0:y1,0:x1]
        return Img
    
    def cut_boundingBox(self):
        x0,y0,x1,y1 = self.boundingBox.get_range()
        Reslut_Img = self.ori_image[y0:y1,x0:x1]
        return Reslut_Img

    def alpha_blending_boundingBox(self,alpha,gamma = 0):
        image = self.cut_boundingBox()
        if len(image) == 0:
            print("There is no bounding box for blending")
            return
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray_image = cv2.cvtColor(gray_image,cv2.COLOR_GRAY2BGR)
        Blending_Img = cv2.addWeighted(image,alpha,gray_image,1 - alpha,gamma)
        Reslut_Img = self.set_boundingBox_image(Blending_Img)
        return Reslut_Img


    def save(self,path):
        cv2.imwrite(path,self.image)

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
    # return the p0 p1 (p0 maybe is the point on left-top or right-top )
    def get(self):
        return self.x0,self.y0,self.x1,self.y1
    # return the bounding box range for cutting image
    def get_range(self):
        x0 = int(min(self.x0,self.x1))
        y0 = int(min(self.y0,self.y1))
        x1 = int(max(self.x0,self.x1))
        y1 = int(max(self.y0,self.y1))
        return x0,y0,x1,y1
    def print(self):
        print("[ ",self.x0," , ",self.y0," , ",self.x1," , ",self.y1," ]")
