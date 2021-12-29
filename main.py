from math import floor
import sys
import os
from UI.UI_Function import GUI, Image, Rect
import cv2
import numpy as np
import types
import atexit
import copy

# model
###########################################

# path
pwd = os.getcwd()
image_dir ="image/"
mask_dir = "mask/"
result_dir = "result/"
video_dir = "video/"
yoloV4_model_path = os.path.join(".","model","yolov4_deepsort")
inpainting_model_path = os.path.join(".","model","generative_inpainting")
mask_RCNN_path = os.path.join(".","model","Mask_RCNN_tf2")
video_path = os.path.join(video_dir,"test.mp4")
output_video_path = os.path.join(video_dir,"result.avi")
output_video_low_path = os.path.join(video_dir,"result_low.avi")
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
use_RCNN = True
# video capture to read video
try:
    vid = cv2.VideoCapture(int(video_path))
except:
    vid = cv2.VideoCapture(video_path)
# output video writer
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'XVID')
out_vid = cv2.VideoWriter(output_video_path, codec, fps, (width, height))
out_vid_low = None

# video setting
begin_frame = 3
stop_msec = 2000
sample_msec = 100
sample_frame = 10

# cv2 function
###########################################
def im_show(window_name,img):
    if(isinstance(img,list)):
        for i,im in enumerate(img):
            cv2.imshow(window_name + "_" + str(i), im)
    else:
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

def print_video_frame_num():
    print(int(vid.get(cv2.CAP_PROP_FRAME_COUNT)))

def print_video_current_frame():
    print("Read frame number : ",vid.get(cv2.CAP_PROP_POS_FRAMES))
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
# global variable use_RCNN means whether using RCNN mask or bounding box (default is true)
def prepare_inpainting_img(img_obj,img_name = "image.png",cut_img_name = "cut.png",mask_img_name = "mask.png"):
    global use_RCNN
    print("Inpainting related image saved as : ",img_name,cut_img_name,mask_img_name)
    if use_RCNN:
        # save full frame image for RCNN input
        save_path = os.path.join(image_dir,img_name)
        img_obj.save_ori(save_path)
        # get the part of mask in bounding box range
        RCNN_mask = mask_RCNN(img_name,mask_img_name)
        bbox_mask = img_obj.masking_boundingBox()
        mask = cv2.bitwise_and(RCNN_mask,RCNN_mask,mask=bbox_mask)
        # dilate once to concate target near by mask to single big mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations = 1)
        contours , hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_NONE)
        print("num of contours : ",len(contours))
        # find the biggest mask
        max_area = 0
        biggest_contr_idx = -1
        for i,contr in enumerate(contours):
            area = cv2.contourArea(contr)
            print("contour area {}: ".format(i),area)
            if area > max_area:
                max_area = area
                biggest_contr_idx = i
        x0,y0,x1,y1 = img_obj.get_combined_boundingBox()
        bbox_h_offset = floor((y1 - y0) / 3)
        print("offset : ",bbox_h_offset)
        append_pixel = int((x1 - x0) / 10)

        # remove other contour only remain target contour
        for i,contr in enumerate(contours):
            if i != biggest_contr_idx:
                # fill contours need array of array as input
                mask = cv2.drawContours(mask,[contr],-1,0,-1)
        # fulfill the inner of contour
        mask = cv2.drawContours(mask,[contours[biggest_contr_idx]],-1,255,-1)
        
        # bgr mask to draw for debug
        temp = mask.copy()
        temp = cv2.cvtColor(temp,cv2.COLOR_GRAY2BGR)
        temp = cv2.drawContours(temp,contours[biggest_contr_idx],-1,(255,0,0),append_pixel*3)
        temp = cv2.drawContours(temp,contours[biggest_contr_idx],-1,(255,255,0),append_pixel)
        # im_show("temp",temp)

        # append target around its contour
        mask_top = cv2.drawContours(mask.copy(),contours[biggest_contr_idx],-1,255,append_pixel)
        mask_mid = cv2.drawContours(mask.copy(),contours[biggest_contr_idx],-1,255,append_pixel*2)
        mask_bottom = cv2.drawContours(mask.copy(),contours[biggest_contr_idx],-1,255,append_pixel*4)
        # let bottom part append(for shadow) more than top part of mask
        mask[:y0 + bbox_h_offset,:] = mask_top[:y0 + bbox_h_offset,:]
        mask[(y0 + bbox_h_offset):(y0 + bbox_h_offset * 2),:] = mask_mid[(y0 + bbox_h_offset):(y0 + bbox_h_offset * 2),:]
        mask[(y0 + bbox_h_offset * 2):,:] = mask_bottom[(y0 + bbox_h_offset * 2):,:]

        # full fill the final mask
        contours , hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_NONE)
        mask = cv2.drawContours(mask,contours,-1,255,-1)

        # temp2 = cv2.cvtColor(mask_cont,cv2.COLOR_GRAY2BGR)
        # temp3 = cv2.cvtColor(mask_dila,cv2.COLOR_GRAY2BGR)
        # cv2.rectangle(temp2,())
        # im_show("mask",[temp,mask_cont,mask_dila,mask])
        
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

def inpainting_fun(idx = None):
    global out_vid_low
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
    # resize image to lower resolution
    if scale > 1:
        result_img_obj = GUI.frame.get_resize_Image(int(w/scale) , int(h/scale))
    # get inpainting result & resize back to original image size
    result = inpainting(result_img_obj,image_name,cut_img_name,mask_img_name,output_img_name)
    result_img_obj.set_image(result)
    if scale > 1 :
        result_img_obj = result_img_obj.get_resize_Image(w,h)
    
    # set up low resolution video writer
    if scale > 1 and out_vid_low == None:
        r_h,r_w,r_c = result.shape
        out_vid_low = cv2.VideoWriter(output_video_low_path, codec, fps, (r_w , r_h))

    # combine the inpainting result & origin image
    # prepare mask to combine two image
    mask_path = os.path.join(mask_dir,mask_img_name)
    mask_obj = Image(mask_path)
    mask_obj = mask_obj.get_resize_Image(w,h)
    mask = cv2.cvtColor(mask_obj.ori_image,cv2.COLOR_BGR2GRAY)
    # using bounding box to combine two image (the size of this is smaller than original image)
    # edit_result = result_img_obj.get_boundingBox_image()
    # using mask to combine two image (the size of this is as same as original image)
    edit_result = result_img_obj.ori_image
    GUI.result.set_image(GUI.frame)
    GUI.result.set_boundingBox_image(edit_result,mask=mask)
    GUI.result.save(save_path)
    GUI.result.draw_boundingBox((0,0,255))
    GUI.display()
    # return origin inpainting result
    return result

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
    global out_vid_low
    global stop_msec
    global begin_frame
    current_msec = 0
    i = 0
    while(current_msec <= stop_msec):
        # for debugging stop the iteration after few frame
        # if (i >= 4 ):
        #     break
        if(i != 0):
            print("Video progress rate : ",current_msec," / ",stop_msec)
            vid.set(cv2.CAP_PROP_POS_FRAMES,begin_frame + i * sample_frame)
            print_video_current_frame()
            return_value, frame = vid.read()
            if(not return_value):
                break
            print("Loop num : ",i)
            ID = track_fun(ID,frame)
            print("Frame state : ",ID," / ",GUI.frame.boundingBox.get())
            GUI.frame.draw_boundingBox((0,0,255))
            GUI.display()
        result = inpainting_fun(begin_frame + i)
        # write edit result into output video
        out_vid.write(GUI.result.ori_image)
        # write original result into low resolution video 
        if(out_vid_low != None):
            out_vid_low.write(result)
        i += 1
        current_msec = vid.get(cv2.CAP_PROP_POS_MSEC)
    print("Finsh video inpainting")

def RCNN_video_inpainting_fun():
    global use_RCNN
    use_RCNN = True
    video_inpainting()

def bbox_video_inpainting_fun():
    global use_RCNN
    use_RCNN = False
    video_inpainting()

def RCNN_inpainting_fun():
    global use_RCNN
    use_RCNN = True
    inpainting_fun()

def bbox_inpainting_fun():
    global use_RCNN
    use_RCNN = False
    inpainting_fun()

def test_fun():
    print("test function")
    print(GUI.get_sampleFrequency())
    # video_inpainting()
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

def videoSampleFrequency_fun():
    global sample_frame
    freq = GUI.get_sampleFrequency()
    sample_frame = freq
    GUI.VideoSampleFrequency_text.setPlainText(str(freq))

def bind_buttton_function():
    GUI.TestButton.clicked.connect(test_fun)
    GUI.CutButton.clicked.connect(cut_button_fun)
    GUI.RCNN_VideoInpaintingButton.clicked.connect(RCNN_video_inpainting_fun)
    GUI.RCNN_InpaintingButton.clicked.connect(RCNN_inpainting_fun)
    GUI.VideoInpaintingButton.clicked.connect(bbox_video_inpainting_fun)
    GUI.InpaintingButton.clicked.connect(bbox_inpainting_fun)
    GUI.Alpha.valueChanged.connect(alpha_blending_fun)
    GUI.VideoSampleFrequency.valueChanged.connect(videoSampleFrequency_fun)
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

# release video & memory when program end
def closeEvent():
    print("Close GUI & release video!")
    out_vid.release()
    out_vid_low.release()
    vid.release()

def init():
    check_dir()
    bind_img_window_func(GUI.ImageDisplayer)
    bind_buttton_function()
    videoSampleFrequency_fun()
    atexit.register(closeEvent)
    vid.set(cv2.CAP_PROP_POS_FRAMES,begin_frame)
    
if __name__ == "__main__":
    # inpainting("1.png","center_mask_256.png")
    init()
    return_value, img = vid.read()
    # img = cv2.imread('image/test.png')
    GUI.set_frame(img)
    GUI.run_app()