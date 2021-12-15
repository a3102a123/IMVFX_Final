import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from models.model import Model
from utils.util import *
from tqdm import tqdm
from PIL import Image, ImageDraw
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')
parser.add_argument('--pretrained_model_dir', type=str, default='pretrained_models', 
                    help='pretrained model folder')
parser.add_argument ('--checkpoint_prefix', type=str, default='')

def load_image(image_file) :
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize (image, [256, 256])
    #print(image.shape)
    image = image / 255.0

    return image

def load_mask(mask_file) :
    mask = tf.io.read_file(mask_file)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask = tf.image.resize(mask, [256, 256])
    mask = mask / 255.0
    return mask

def test(config) :
    gt_image = load_image(config.image)
    gt_image = np.expand_dims(gt_image, axis=0)
    mask = load_mask(config.mask)
    mask = np.where(np.array(mask) > 0.5, 1.0, 0.0).astype(np.float32)
    mask = np.expand_dims(mask, axis=0)

    input_image = np.where(mask==1, 1, gt_image)

    prediction_coarse, prediction_refine = generator([input_image, mask], training=False)
    prediction_refine = prediction_refine * mask + gt_image * (1  - mask)
    save_image(prediction_refine[0,...], config.output)

if __name__ == '__main__' :
    config, unknown = parser.parse_known_args()

    model = Model()
    generator = model.build_generator(256, 256)
    checkpoint = tf.train.Checkpoint (generator=generator)
    checkpoint.restore (os.path.join (config.pretrained_model_dir, config.checkpoint_prefix))
    
    test(config)