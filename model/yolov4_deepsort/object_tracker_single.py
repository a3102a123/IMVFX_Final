import os
import sys
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
# original author paramete for app
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

class flag():
    def __init__(self,video_path = './data/video/test.mp4',output_path = None,is_show=True):
        self.size = int(416)
        self.tiny = False
        self.model = 'yolov4'
        self.video = video_path
        self.output = output_path
        self.output_format = 'XVID'
        self.iou = 0.45
        self.score = 0.50
        self.dont_show = not is_show
        self.info = True
        self.count = False

def init_tracker():
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)
    return tracker

def object_track(root_path,tracker,frame,rect_list,id = -1,is_show=False):
    FLAGS = flag(is_show = is_show)
    # Definition of the parameters
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = os.path.join(root_path,'model_data/mars-small128.pb')
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load standard tensorflow saved model
    weights_path = os.path.join(root_path,'./checkpoints/yolov4-416')
    saved_model_loaded = tf.saved_model.load(weights_path, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output and False:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    # calc single frame
    for i in range(1):
        image = Image.fromarray(frame)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on yolo
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        #allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        is_find = False
        result_rect_list = []
        result_id = -1
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            if int(track.track_id) == id:
                is_find == True
                result_id = int(track.track_id)
                bbox = track.to_tlbr()
                class_name = track.get_class()
                result_rect_list = bbox
                break
        
        min_dist = sys.maxsize
        if not is_find:
            center = np.zeros(2)
            center = np.array((int(rect_list[2] + rect_list[0]) / 2 , int(rect_list[3] + rect_list[1]) / 2),dtype=float)
            for track in tracker.tracks:
                bbox = track.to_tlbr()
                track_center = np.zeros(2)
                track_center = np.array((int(bbox[2] + bbox[0]) / 2 , int(bbox[3] + bbox[1]) / 2),dtype=float)
                dist = np.linalg.norm(track_center - center)
                if dist < min_dist:
                    min_dist = dist
                    class_name = track.get_class()
                    result_rect_list = bbox
                    result_id = int(track.track_id)

        # draw bbox on screen
        color = colors[int(result_id) % len(colors)]
        color = [i * 255 for i in color]
        cv2.rectangle(frame, (int(result_rect_list[0]), int(result_rect_list[1])), (int(result_rect_list[2]), int(result_rect_list[3])), color, 2)
        cv2.rectangle(frame, (int(result_rect_list[0]), int(result_rect_list[1]-30)), (int(result_rect_list[0])+(len(class_name)+len(str(result_id)))*17, int(result_rect_list[1])), color, -1)
        cv2.putText(frame, class_name + "-" + str(result_id),(int(result_rect_list[0]), int(result_rect_list[1]-10)),0, 0.75, (255,255,255),2)
        
        # if enable info flag then print details about each track
        if FLAGS.info:
            print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(result_id), class_name, (int(result_rect_list[0]), int(result_rect_list[1]), int(result_rect_list[2]), int(result_rect_list[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output and False:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    session.close()
    return result_id,result_rect_list

if __name__ == '__main__':
    try:
        video_path = './data/video/test.mp4'
        tracker = init_tracker()
        rect_list = [865, 386, 989, 590]
        # begin video capture
        try:
            vid = cv2.VideoCapture(int(video_path))
        except:
            vid = cv2.VideoCapture(video_path)
        
        for i in range(3):
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                id , rect = object_track('./',tracker,frame,rect_list,id=6,is_show=True)
                print(id," : ",rect)
            else:
                print('Video has ended or failed, try a different video format!')
                break
    except SystemExit:
        pass