from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import cv2
import numpy as np

from keras.datasets import cifar10
from keras import backend as K
from keras.utils import np_utils

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

FLAGS = None
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

def create_image_lists():
    image_dir='/home/rick/derma/dataset'
    testing_percentage=20

    """
    if not gfile.Exists(image_dir): #check if image_dir exists
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        #return None
    """

    result = {}
    counter_for_result_label=0

    sub_dirs = [x[0] for x in gfile.Walk(image_dir)] #create sub_dirs


    # The root directory comes first, so skip it.

    dir_name=[]

    #ignore first element in sub_dir
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue


        dir_name = os.path.basename(sub_dir)

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        #dir_name = os.path.basename(image_dir)
        #if dir_name == image_dir:
          #continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
          #for image_dir in sub_dir
          file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
          file_list.extend(gfile.Glob(file_glob))      #create a list of all files


        """
        #exception cases for not file_list  
        if not file_list:
          tf.logging.warning('No files found')
          continue
        if len(file_list) < 20:
          tf.logging.warning(
              'WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
          tf.logging.warning(
              'WARNING: Folder {} has more than {} images. Some images will '
              'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        """
        #using regex to set label name
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

        #dividing
        training_images = []
        testing_images = []
        for file_name in file_list:
          base_name = os.path.basename(file_name) #just take name of image (eg: 5547758_ed54_n)
          # We want to ignore anything after '_nohash_' in the file name when
          # deciding which set to put an image in, the data set creator has a way of
          # grouping photos that are close variations of each other. For example
          # this is used in the plant disease data set to group multiple pictures of
          # the same leaf.
          hash_name = re.sub(r'_nohash_.*$', '', file_name)
          # This looks a bit magical, but we need to decide whether this file should
          # go into the training, testing, or validation sets, and we want to keep
          # existing files in the same set even if more files are subsequently
          # added.
          # To do that, we need a stable way of deciding based on just the file name
          # itself, so we do a hash of that and then use that to generate a
          # probability value that we use to assign it.
          hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
          percentage_hash = ((int(hash_name_hashed, 16) %
                              (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                             (100.0 / MAX_NUM_IMAGES_PER_CLASS))
          if percentage_hash < testing_percentage:
            #testing_images.append(file_name)
            testing_images.append(cv2.imread(file_name))
            #testing_images.append(base_name)           
          else:
            #training_images.append(file_name)
            training_images.append(cv2.imread(file_name))
            #training_images.append(base_name)


        result[counter_for_result_label] = {
        'training_label': [counter_for_result_label]*(len(training_images)),
        'testing_label': [counter_for_result_label]*(len(testing_images)),
        'training': training_images,
        'testing': testing_images,
        }

        counter_for_result_label=counter_for_result_label+1

        """    
        result[label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        }
        """
    return result



"""
#return result

def add_evaluation_step(result_tensor, ground_truth_tensor):
    with tf.name_scope('accuracy'):
      with tf.name_scope('correct_prediction'):
      prediction = tf.argmax(result_tensor, 1)
      correct_prediction = tf.equal(
          prediction, tf.argmax(ground_truth_tensor, 1))
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)
  return evaluation_step, prediction


Inserts the operations we need to evaluate the accuracy of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

  Returns:
    Tuple of (evaluation step, prediction).


Builds a list of training images from the file system.

  Analyzes the sub folders in the image directory, splits them into stable
  training, testing, and validation sets, and returns a data structure
  describing the lists of images for each label and their paths.

  Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.

  Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
"""

#evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)