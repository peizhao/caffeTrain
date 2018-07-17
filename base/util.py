#!/usr/bin/env python

""" helper function for the training """

import os
from PIL import Image

""" get the file list under path
    suffix to describe which type file you want to get
"""
def getFileList(path, suffix = None):
    if(suffix == None):
        return os.listdir(path)
    else:
        file_list = [fn for fn in os.listdir(path) if any(fn.endswith(ext) for ext in suffix)]
    return file_list

def getImageFileList(path):
    suffix= [ '.jpg', '.png']
    return getFileList(path, suffix)

def create_dir(dir_name):
    if (os.path.exists(dir_name)):
        return
    else:
        os.mkdir(dir_name)

def create_file(file_name):
        if(os.path.exists(file_name)):
            return
        else:
            os.mknod(file_name)

""" normalize the gtfile:
    root: the root path for positive root folder
    sub_folder: one subfoler under the root path
"""
def normalize_with_gtfile(root,sub_folder):
    fileList = getFileList("{}/{}".format(root, sub_folder),".txt")
    gtfile = fileList[0]
    gtfile_path = "{}/{}/{}".format(root,sub_folder,gtfile)
    gt_handler = open(gtfile_path,'r')
    gtlines = gt_handler.readlines()
    save_lines = []
    for gtline in gtlines:
        line = gtline.strip().split()
        x1 = line[2]
        y1 = line[3]
        width = line[4]
        height = line[5]
        image_file = "{}/{}/{}".format(root,sub_folder,line[0])
        im = Image.open(image_file)
        im_width,im_height = im.size
        y2 = (float(height) + float(y1)) / im_height
        x2 = (float(width) + float(x1)) / im_width
        y1 = float(y1) / im_height
        x1 = float(x1) / im_width
        save_lines.append("{}/{} {:.3f} {:.3f} {:.3f} {:.3f} 1\n".format(sub_folder, line[0], x1, y1, x2, y2))
    return save_lines

""" normlize without the gtfile which use for the negative case
    root: the root path for negative root folder
    sub_folder: one subfoler under the root path
"""
def normlize(root,sub_folder):
    save_lines = []
    #image_list = os.listdir("{}/{}".format(root,sub_folder))
    image_path = "{}/{}".format(root,sub_folder)
    image_list = getImageFileList(image_path)
    for item in image_list:
        save_lines.append("{}/{} 0.0 0.0 0.0 0.0 0\n".format(sub_folder, item))
    return save_lines




