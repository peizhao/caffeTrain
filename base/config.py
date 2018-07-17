#!/usr/bin/env python

"""
config.py is to detect the system configuration for the caffe trainning process
export all the global setting to other module
"""
import os

class TrainConfig:

    def __init__(self):
        self.caffe_root= os.environ['CAFFE_HOME']
        self.output_root=""
        self.image_format='RGB'
        self.image_width = 0
        self.image_height = 0
        self.origin_pos_image_root=""
        self.origin_neg_image_root=""
        self.train_pos_filter_list =[]
        self.train_neg_filter_list = []
        self.label_shape =[]
        self.train_solver =""

    """ the caffe root folder """
    def get_caffe_root(self):
        return self.caffe_root

    """ the output folder for the tranning process"""
    def get_output_root_folder(self):
        return self.output_root

    def set_output_root_folder(self, output):
        self.output_root =output

    """ the target training image format """
    def get_image_format(self):
        return self.image_format

    def set_image_format(self, image_format):
        self.image_format = image_format

    """ the target training image size"""
    def get_image_width(self):
        return self.image_width

    def set_image_width(self, width):
        self.image_width = width

    def get_image_height(self):
        return self.image_height

    def set_image_height(self, height):
        self.image_height = height

    """ the origin image folder """
    def get_origin_pos_image_root(self):
        return self.origin_pos_image_root

    def set_origin_pos_image_root(self, folder):
        self.origin_pos_image_root = folder

    def get_origin_neg_image_root(self):
        return self.origin_neg_image_root

    def set_origin_neg_image_root(self, folder):
        self.origin_neg_image_root = folder



