#!/usr/bin/env python

from util import *
import logging
import sys
import cv2
import numpy
from hisceneLmdb import *

logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    # filename='logger.log', filemode='w',
    format="[%(asctime)s] %(name)s:%(levelname)s: %(message)s"
)

class hisceneTrain:

    def __init__(self, config):
        self.trainConfig = config
        self.__create_output_folder()
        self.train_file = "{}/{}".format(self.output_root, "train.txt")
        self.test_file = "{}/{}".format(self.output_root, "test.txt")
        self.train_pos_filter = self.trainConfig.train_pos_filter_list
        self.train_neg_filter = self.trainConfig.train_neg_filter_list
        self.resize_folder=""
        self.Lmdb = hisceneLmdb(self.output_root, self.trainConfig.label_shape)
        self.__create_resize_folder()

    def generate_train_file(self):
        if os.path.exists(self.train_file):
            logging.info("train file already exist, please delete to Regenerate")
            return

        logging.info("generate train file now ...")
        create_file(self.train_file)
        train_pos_folder_list = []
        train_neg_folder_list = []
        train_pos_root = self.trainConfig.get_origin_pos_image_root()
        train_neg_root = self.trainConfig.get_origin_neg_image_root()

        """clear the train file first"""
        open(self.train_file,'w').close()

        for item in os.listdir(self.trainConfig.get_origin_pos_image_root()):
            if item not in self.train_pos_filter:
                train_pos_folder_list.append(item)
        logging.info("training pos folders: ")
        logging.info(train_pos_folder_list)

        for item in os.listdir(self.trainConfig.get_origin_neg_image_root()):
            if item not in self.train_neg_filter:
                train_neg_folder_list.append(item)
        logging.info("training neg folders: ")
        logging.info(train_neg_folder_list)

        self.__generate_pos_normalize_file(train_pos_root,train_pos_folder_list,self.train_file)
        self.__generate_neg_normalize_file(train_neg_root,train_neg_folder_list,self.train_file)

    def generate_test_file(self):
        if os.path.exists(self.test_file):
            logging.info("test file already exist, please delete to Regenerate")
            return
        logging.info("generate test file now ...")

        create_file(self.test_file)
        test_pos_folder_list = []
        test_neg_folder_list = []
        test_pos_root = self.trainConfig.get_origin_pos_image_root()
        test_neg_root = self.trainConfig.get_origin_neg_image_root()

        """clear the test file first"""
        open(self.test_file,'w').close()

        for item in os.listdir(self.trainConfig.get_origin_pos_image_root()):
            if item in self.train_pos_filter:
                test_pos_folder_list.append(item)
        logging.info("testing pos folders: ")
        logging.info(test_pos_folder_list)

        for item in os.listdir(self.trainConfig.get_origin_neg_image_root()):
            if item in self.train_neg_filter:
                test_neg_folder_list.append(item)
        logging.info("testing neg folders: ")
        logging.info(test_neg_folder_list)

        self.__generate_pos_normalize_file(test_pos_root,test_pos_folder_list,self.test_file)
        self.__generate_neg_normalize_file(test_neg_root,test_neg_folder_list,self.test_file)
        return

    def resize(self):
        logging.info("begin to resize the image ...")
        self.__create_resize_folder()
        dst_pos_dir_list = getFileList(self.trainConfig.get_origin_pos_image_root())
        dst_neg_dir_list = getFileList(self.trainConfig.get_origin_neg_image_root())
        height = self.trainConfig.get_image_height()
        width = self.trainConfig.get_image_width()
        format = self.trainConfig.get_image_format()
        for item in dst_neg_dir_list:
            dst_folder = "{}/{}".format(self.resize_folder,item)
            src_folder = "{}/{}".format(self.trainConfig.get_origin_neg_image_root(),item)
            logging.info("resize folder:"+item)
            if(self.trainConfig.get_image_format() == 'BGR'):
                self.__generate_resize_image_BGR(src_folder,dst_folder,height,width)
            else:
                self.__generate_resize_image(src_folder,dst_folder,height,width,format)

        for item in dst_pos_dir_list:
            dst_folder = "{}/{}".format(self.resize_folder, item)
            src_folder = "{}/{}".format(self.trainConfig.get_origin_pos_image_root(), item)
            logging.info("resize folder:" + item)
            if (self.trainConfig.get_image_format() == 'BGR'):
                self.__generate_resize_image_BGR(src_folder, dst_folder, height, width)
            else:
                self.__generate_resize_image(src_folder, dst_folder, height, width, format)
        return

    def create_lmdb(self):
        logging.info("create lmdb now ...")
        train_images=[]
        train_labels=[]
        test_images=[]
        test_labels=[]
        with open(self.train_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.strip().split()
                a = array([tmp[1], tmp[2], tmp[3], tmp[4], tmp[5]], float32)
                train_images.append("{}/{}".format(self.resize_folder,tmp[0]))
                train_labels.append(a)
        self.Lmdb.create_train_data_lmdb(train_images)
        self.Lmdb.create_train_label_lmdb(train_labels)
        logging.info("create train data and label done")

        with open(self.test_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.strip().split()
                a = array([tmp[1], tmp[2], tmp[3], tmp[4], tmp[5]], float32)
                test_images.append("{}/{}".format(self.resize_folder,tmp[0]))
                test_labels.append(a)
        self.Lmdb.create_test_data_lmdb(test_images)
        self.Lmdb.create_test_label_lmdb(test_labels)
        logging.info("create test data and label done")
        return

    def create_mean(self):
        mean_command = self.trainConfig.get_caffe_root()+"/build/tools/compute_image_mean"
        train_db = self.Lmdb.train_data_path
        binaryproto = "{}/{}".format(self.output_root,"mean.binaryproto")
        cmd = mean_command+' '+train_db+' '+binaryproto
        os.system(cmd)
        return

    def train(self):
        train_command = self.trainConfig.get_caffe_root()+"/build/tools/caffe train -solver "
        cmd = train_command + self.trainConfig.train_solver
        os.system(cmd)
        return

    def train_gpu(self,gpu_Options):
        train_command = self.trainConfig.get_caffe_root()+"/build/tools/caffe train -solver "
        cmd = train_command + self.trainConfig.train_solver+" "+ gpu_Options
        os.system(cmd)
        return

    def __create_output_folder(self):
        output = self.trainConfig.get_output_root_folder()
        if(output == ""):
            logging.info("ouput folder is not setting")
            sys.exit(-1)
        self.output_root = output
        create_dir(self.output_root)

    def __create_resize_folder(self):
        resize_name = "resize_{}_{}".format(self.trainConfig.get_image_height(),self.trainConfig.get_image_width())
        self.resize_folder = "{}/{}".format(self.output_root,resize_name)
        create_dir(self.resize_folder)

    def __generate_pos_normalize_file(self, root_path, folder_list, output_file):
        fHandler = open(output_file,'a')
        for item in folder_list:
            fHandler.writelines(normalize_with_gtfile(root_path,item))
        fHandler.flush()
        fHandler.close()

    def __generate_neg_normalize_file(self, root_path, folder_list, output_file):
        fHandler = open(output_file, 'a')
        for item in folder_list:
            fHandler.writelines(normlize(root_path,item))
        fHandler.flush()
        fHandler.close()

    def __generate_resize_image(self, src_folder, dst_folder, resize_height, resize_width, format):
        if not os.path.exists(src_folder):
            logging.error("src_folder not exist")
            return
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        img_list = getImageFileList(src_folder)
        for img_file in img_list:
            logging.debug("resize image file:"+ img_file)
            try:
                img = Image.open(src_folder + '/' + img_file)
                new_img = img.resize((resize_width, resize_height))
                new_img.convert(format).save(dst_folder+ '/' + img_file)
            except KeyError, e:
                logging.info(e)
            except IOError, e:
                logging.info(e)

    def __generate_resize_image_BGR(self, src_folder, dst_folder, resize_height, resize_width):
        if not os.path.exists(src_folder):
            logging.error("src_folder not exist")
            return
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        img_list = getImageFileList(src_folder)
        for img_file in img_list:
            logging.debug("resize image file:"+ img_file)
            try:
                img = Image.open(src_folder + '/' + img_file)
                new_img = img.resize((resize_width, resize_height))
                opencvImage = cv2.cvtColor(numpy.array(new_img), cv2.COLOR_RGB2BGR)
                cv2.imwrite(dst_folder+ '/' + img_file, opencvImage)
                """new_img.convert(format).save(dst_folder+ '/' + img_file)"""
            except KeyError, e:
                logging.info(e)
            except IOError, e:
                logging.info(e)