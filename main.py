import sys
import os
from PyQt5 import QtWidgets
from UI.UI_Function import GUI, Image, Rect
import cv2
import types

# model
###########################################

# path
pwd = os.getcwd()
image_dir ="image/"
mask_dir = "mask/"
result_dir = "result/"
yoloV4_model_path = os.path.join(".","model","yolov4_deepsort")
inpainting_model_path = os.path.join(".","model","generative_inpainting")

### YOLOv4 + deep sort
# change to model folder & add this absolute path of model for import
os.chdir(yoloV4_model_path)
sys.path.append(os.getcwd())
import object_tracker_single as yolo
# after import go back to origin running position
os.chdir(pwd)


# global variable
###########################################
# GUI instance
GUI = GUI()
video = cv2.VideoCapture("video/sample.mp4")
# model
tracker = yolo.init_tracker()

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

# model
###########################################
# parameter input the image file name
def prepare_inpainting_img(img_obj,cut_img_name = "cut.png",mask_img_name = "mask.png"):
    save_path = os.path.join(image_dir,cut_img_name)
    img_obj.save_cut(save_path)
    save_path = os.path.join(mask_dir,mask_img_name)
    img_obj.save_mask(save_path)

# return the Image object of 256*256 inpainting image
def inpainting(img_obj,cut_img_name = "cut.png",mask_img_name = "mask.png",output_img_name = "inpainting_result.png"):
    prepare_inpainting_img(img_obj,cut_img_name,mask_img_name)
    root_path = pwd
    image_path = os.path.join(root_path,image_dir,cut_img_name)
    mask_path = os.path.join(root_path,mask_dir,mask_img_name)
    output_path = os.path.join(root_path,result_dir,output_img_name)
    print(image_path , mask_path, output_path)
    os.chdir(inpainting_model_path)
    os.system("python test.py"\
        " --image " + image_path + \
        " --mask " + mask_path + \
        " --output " + output_path + \
        " --checkpoint_dir model_logs/release_places2_256")
    os.chdir(pwd)
    Img = cv2.imread(output_path)
    return Img

# button function
###########################################
def test_fun():
    save_path = os.path.join(result_dir,"edit_result.jpg")
    result_img_obj = GUI.frame.get_Image()
    result = inpainting(result_img_obj)
    im_show("inpainting result",result)
    result_img_obj.set_image(result)
    result = result_img_obj.get_boundingBox_image()
    GUI.result.set_image(GUI.frame)
    GUI.result.set_boundingBox_image(result)
    GUI.result.save(save_path)
    GUI.result.draw_boundingBox((0,0,255))
    GUI.display()

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
    GUI.set_result(GUI.frame)
    Img = GUI.result.alpha_blending_boundingBox(alpha)
    GUI.display()

def bind_buttton_function():
    GUI.TestButton.clicked.connect(test_fun)
    GUI.cutButton.clicked.connect(cut_button_fun)
    GUI.Alpha.valueChanged.connect(alpha_blending_fun)
# mouse trigger function
###########################################
    # draw rectagle on image1
def img_window_mousePressEvent(self,event):
    GUI.draing_flag = True
    GUI.frame.boundingBox.set_p0(event.x(),event.y())

def img_window_mouseReleaseEvent(self,event):
    GUI.draing_flag = False


def img_window_mouseMoveEvent(self,event):
    if GUI.draing_flag:
        GUI.frame.boundingBox.set_p1(event.x(),event.y())
        GUI.frame.clear_drawing()
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
    # inpainting("1.png","center_mask_256.png")
    init()
    img = cv2.imread('image/sample.png')
    GUI.set_frame(img)
    GUI.run_app()