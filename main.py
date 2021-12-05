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
video = cv2.VideoCapture("video/sample.mp4")
image_dir = os.path.join(".", "image")
result_dir = os.path.join(".","result")

# cv2 function
###########################################
def im_show(window_name,img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)

def get_video_frame(frame_idx,video):
    # the number of frame in video
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # cv2.CAP_PROP_POS_MSEC , 0-based index of the millisecond to be decoded/captured next.
    # cv2.CAP_PROP_POS_FRAMES : 0-based index of the frame to be decoded/captured next.
    video.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)
    # ret is the return value (bool)
    ret, frame = video.read()
    return frame
# button function
###########################################
def cut_button_fun():
    save_path = os.path.join(result_dir,"cut_result.jpg")
    print("Cut out the image of boundiong box as result saved in : \n",save_path)
    Img = GUI.frame.cut_boundingBox()
    if len(Img) == 0:
            print("There is no bounding box for blending")
            return
    GUI.set_result(Img)
    GUI.result.save(save_path)

def alpha_blending_fun():
    # simple alpha blending of gray image in the are of bounding box
    alpha = GUI.get_alpha()
    GUI.Alpha_text.setPlainText(str(alpha))
    Img = GUI.frame.alpha_blending_boundingBox(alpha)
    GUI.set_result(Img)

def bind_buttton_function():
    GUI.cutButton.clicked.connect(cut_button_fun)
    GUI.Alpha.valueChanged.connect(alpha_blending_fun)
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