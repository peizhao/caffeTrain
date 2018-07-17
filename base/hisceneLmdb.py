import sys
import numpy as np
import lmdb
from numpy import *
import os
import shutil
import random
import caffe

import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s:%(levelname)s: %(message)s"
)

class hisceneLmdb:

    def __init__(self, lmdb_root, shape):
        self.lmdb_root = lmdb_root
        self.train_data_path = "{}/{}".format(lmdb_root, "train_data_lmdb")
        self.train_label_path = "{}/{}".format(lmdb_root, "train_label_lmdb")
        self.test_data_path = "{}/{}".format(lmdb_root, "test_data_lmdb")
        self.test_label_path = "{}/{}".format(lmdb_root, "test_label_lmdb")
        self.lable_shape = shape

    def create_train_label_lmdb(self, labels):
        if os.path.exists(self.train_label_path):
            shutil.rmtree(self.train_label_path)
        self.__create_label(self.train_label_path, labels)

    def create_train_data_lmdb(self, images):
        if os.path.exists(self.train_data_path):
            shutil.rmtree(self.train_data_path)
        self.__create_data(self.train_data_path,images)

    def create_test_label_lmdb(self, labels):
        if os.path.exists(self.test_label_path):
            shutil.rmtree(self.test_label_path)
        self.__create_label(self.test_label_path,labels)

    def create_test_data_lmdb(self, images):
        if os.path.exists(self.test_data_path):
            shutil.rmtree(self.test_data_path)
        self.__create_data(self.test_data_path,images)

    def __create_label(self, lmdb_label_name, labels):
        for idx in range(int(math.ceil(len(labels) / 1000.0))):
            in_db_label = lmdb.open(lmdb_label_name, map_size=int(1e12))
            with in_db_label.begin(write=True) as in_txn:
                for label_idx, label_ in enumerate(labels[(1000 * idx):(1000 * (idx + 1))]):
                    im_dat = caffe.io.array_to_datum(np.array(label_).astype(float).reshape(self.lable_shape))
                    in_txn.put('{:0>10d}'.format(1000 * idx + label_idx), im_dat.SerializeToString())
            in_db_label.close()

    def __create_data(self, lmdb_data_name, images):
        for idx in range(int(math.ceil(len(images) / 1000.0))):
            in_db_data = lmdb.open(lmdb_data_name, map_size=int(1e12))
            with in_db_data.begin(write=True) as in_txn:
                for in_idx, in_ in enumerate(images[(1000 * idx):(1000 * (idx + 1))]):
                    im = caffe.io.load_image(in_)
                    im_dat = caffe.io.array_to_datum(im.astype(float).transpose((2, 0, 1)))
                    in_txn.put('{:0>10d}'.format(1000 * idx + in_idx), im_dat.SerializeToString())
            in_db_data.close()