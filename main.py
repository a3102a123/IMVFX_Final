from PyQt5 import QtWidgets
from UI.UI_Function import GUI, Rect
import sys
import os
import cv2
import types

# global variable
###########################################
# GUI instance
GUI = GUI()
image_dir = os.path.join(".", "image")
result_dir = os.path.join(".","result")

# cv2 function
###########################################
def im_show(window_name,img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)

# button function
###########################################
def cut_button_fun():
    print("Cut out the image of boundiong box as result")
    Img = GUI.frame.cut_boundingBox()
    GUI.set_result(Img)
    GUI.result.save(os.path.join(result_dir,"cut_result.jpg"))

def bind_buttton_function():
    GUI.cutButton.clicked.connect(cut_button_fun)
# mouse trigger function
###########################################
    # draw rectagle on image1
def img_window_mousePressEvent(self,event):
    GUI.draing_flag = True
    GUI.frame.boundingBox.set_p0(event.x(),event.y())
    GUI.frame.boundingBox.print()

def img_window_mouseReleaseEvent(self,event):
    GUI.draing_flag = False


def img_window_mouseMoveEvent(self,event):
    if GUI.draing_flag:
        GUI.frame.boundingBox.set_p1(event.x(),event.y())
        GUI.frame.draw_boundingBox((0,0,255))
        GUI.display()

def img_window_mouseDoubleClickEvent(self,event):
    GUI.frame.boundingBox.set(0,0,0,0)
    GUI.frame.reset()
    GUI.display()

def bind_img_window_func(obj):
    obj.mousePressEvent = types.MethodType(img_window_mousePressEvent, obj)
    obj.mouseReleaseEvent = types.MethodType(img_window_mouseReleaseEvent, obj)
    obj.mouseMoveEvent = types.MethodType(img_window_mouseMoveEvent, obj)
    obj.mouseDoubleClickEvent = types.MethodType(img_window_mouseDoubleClickEvent, obj)

def check_dir():
    os.makedirs(image_dir,exist_ok=True)
    os.makedirs(result_dir,exist_ok=True)

def init():
    check_dir()
    bind_img_window_func(GUI.ImageDisplayer)
    bind_buttton_function()
    
if __name__ == "__main__":
    init()
    img = cv2.imread('image/sample.png')
    GUI.set_frame(img)
    GUI.run_app()