import sys
import os
from UI.UI_Function import GUI, Image, Rect
import cv2
import numpy as np
import types
import time

# model
###########################################

# path
pwd = os.getcwd()
image_dir ="image/"
mask_dir = "mask/"
result_dir = "result/"
yoloV4_model_path = os.path.join(".","model","yolov4_deepsort")
inpainting_model_path = os.path.join(".","model","generative_inpainting")
mask_RCNN_path = os.path.join(".","model","Mask_RCNN_tf2")
### YOLOv4 + deep sort
# change to model folder & add this absolute path of model for import
try :
    os.chdir(yoloV4_model_path)
    sys.path.append(os.getcwd())
    import object_tracker_single as yolo
    # model
    tracker = yolo.init_tracker()
except:
    print("Import yolo fail! (wrong environment)")
finally:
    # after import go back to origin running position
    os.chdir(pwd)

# global variable
###########################################
# GUI instance
GUI = GUI()
video = cv2.VideoCapture("video/sample.mp4")
# video setting
sample_msec = 100
sample_frame = 10

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

# check image size to fit computer capability
def checkImagesize(img_obj,width,height):
    scale = 1
    h,w,c = GUI.frame.ori_image.shape
    while w > width or h > height:
        h /= 2
        w /= 2
        scale *= 2
    return scale

# model
###########################################
# return the mask calc by mask RCNN
def mask_RCNN(img_name = "image.png",output_mask_name = "RCNN_mask.png"):
    root_path = pwd
    image_path = os.path.join(root_path,image_dir,img_name)
    output_path = os.path.join(root_path,mask_dir,output_mask_name)
    os.chdir(mask_RCNN_path)
    os.system("python test.py"\
        " --image " + image_path + \
        " --output " + output_path + \
        " --checkpoint_dir .\mask_rcnn_coco.h5")
    os.chdir(pwd)
    Img = cv2.imread(output_path,cv2.IMREAD_GRAYSCALE)
    return Img
# parameter input the image file name
# use_RCNN means whether using RCNN mask or bounding box
def prepare_inpainting_img(img_obj,img_name = "image.png",cut_img_name = "cut.png",mask_img_name = "mask.png",use_RCNN = True):
    if use_RCNN:
        # save full frame image for RCNN input
        save_path = os.path.join(image_dir,img_name)
        img_obj.save_ori(save_path)
        # get the part of mask in bounding box range
        RCNN_mask = mask_RCNN(img_name,mask_img_name)
        bbox_mask = img_obj.masking_boundingBox()
        mask = cv2.bitwise_and(RCNN_mask,RCNN_mask,mask=bbox_mask)
        im_show("mask",mask)
        # dilate the mask to fit the person
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations = 10)
        # save result mask and cutted image by this mask
        save_path = os.path.join(image_dir,cut_img_name)
        img_obj.save_mask_cut(save_path,mask)
        save_path = os.path.join(mask_dir,mask_img_name)
        cv2.imwrite(save_path,mask)
    else:
        save_path = os.path.join(image_dir,cut_img_name)
        img_obj.save_cut(save_path)
        save_path = os.path.join(mask_dir,mask_img_name)
        img_obj.save_mask(save_path)


# return the Image object of 256*256 inpainting image
def inpainting(img_obj,img_name = "image.png",cut_img_name = "cut.png",mask_img_name = "mask.png",output_img_name = "inpainting_result.png"):
    prepare_inpainting_img(img_obj,img_name,cut_img_name,mask_img_name)
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

def inpainting_fun(idx = 0):
    # the file name of used image
    image_name = "image_{}.png".format(idx)
    cut_img_name = "cut_{}.png".format(idx)
    mask_img_name = "mask_{}.png".format(idx)
    output_img_name = "inpainting_{}.png".format(idx)
    save_path = os.path.join(result_dir,"edit_result_{}.png".format(idx))
    
    # check Image object to prevent too big size of image to crash the computer
    result_img_obj = GUI.frame.get_Image()
    h,w,c = result_img_obj.image.shape
    scale = checkImagesize(result_img_obj,1280,720)
    if scale > 1:
        result_img_obj = GUI.frame.get_resize_Image(int(w/scale) , int(h/scale))
    # get inpainting result & resize back to original image size
    result = inpainting(result_img_obj,image_name,cut_img_name,mask_img_name,output_img_name)
    result_img_obj.set_image(result)
    if scale > 1 :
        result_img_obj = result_img_obj.get_resize_Image(w,h)
    
    # combine the inpainting result & origin image
    result = result_img_obj.get_boundingBox_image()
    GUI.result.set_image(GUI.frame)
    GUI.result.set_boundingBox_image(result)
    GUI.result.save(save_path)
    GUI.result.draw_boundingBox((0,0,255))
    GUI.display()

def track_fun(ID,next_frame):
    # update frame
    GUI.set_frame(next_frame)
    os.chdir(yoloV4_model_path)
    img = cv2.cvtColor(GUI.frame.ori_image, cv2.COLOR_BGR2RGB)
    
    # tracking , the return list contain all bbox overlap with the target object
    id , bbox_list = yolo.object_track("./",tracker,img,GUI.frame.boundingBox.get_list(),id=ID,is_show = False)
    print("Track Result : ",id," / ",bbox_list)
    
    # update frame bounding box
    GUI.frame.set_boundingBox_list(bbox_list,0)
    os.chdir(pwd)
    return id

def video_inpainting():
    ID = -1
    video_path = os.path.join("video","test.mp4")
    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)
    for i in range(1,5,1):
        if(i != 1):
            vid.set(cv2.CAP_PROP_POS_MSEC,i * sample_msec)
            return_value, frame = vid.read()
            if(not return_value):
                break
            print("Loop num : ",i)
            ID = track_fun(ID,frame)
            print("Frame state : ",ID," / ",GUI.frame.boundingBox.get())
            GUI.frame.draw_boundingBox((0,0,255))
            GUI.display()
        # for debugging tracking skip the first frame
        # else : 
        #     continue
        inpainting_fun(i)
    print("Finsh video inpainting")

def test_fun():
    video_inpainting()
    # prepare_inpainting_img(GUI.frame)

def cut_button_fun():
    save_path = os.path.join(result_dir,"cut_result.png")
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
    GUI.CutButton.clicked.connect(cut_button_fun)
    GUI.InpaintingButton.clicked.connect(inpainting_fun)
    GUI.Alpha.valueChanged.connect(alpha_blending_fun)
# mouse trigger function
###########################################
    # draw rectagle on image1
def img_window_mousePressEvent(self,event):
    ratio_w, ratio_h = GUI.get_Displayer2Frame_ratio()
    GUI.draing_flag = True
    GUI.frame.boundingBox.set_p0(event.x() * ratio_w,event.y() * ratio_h)

def img_window_mouseReleaseEvent(self,event):
    GUI.draing_flag = False


def img_window_mouseMoveEvent(self,event):
    if GUI.draing_flag:
        ratio_w, ratio_h = GUI.get_Displayer2Frame_ratio()
        GUI.frame.boundingBox.set_p1(event.x() * ratio_w,event.y() * ratio_h)
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
    img = cv2.imread('image/test.png')
    GUI.set_frame(img)
    GUI.run_app()