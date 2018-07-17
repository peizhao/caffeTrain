#!/usr/bin/env python
from base.config import *
from base.hisceneTrain import *

config = TrainConfig()
config.set_output_root_folder("/home/leon/temp/data/headcount/output")
config.set_origin_pos_image_root("/home/leon/temp/data/headcount/origin_data/pos")
config.set_origin_neg_image_root("/home/leon/temp/data/headcount/origin_data/neg")
config.train_pos_filter_list.append('fzwkllwy0001')
config.train_neg_filter_list.append('potting')
config.set_image_width(32)
config.set_image_height(64)
config.label_shape = (5, 1, 1)
config.train_solver = "/home/leon/temp/data/models/solver.prototxt"
Train = hisceneTrain(config)

#Train.generate_train_file()
#Train.generate_test_file()
#Train.resize()
#Train.create_lmdb()
#Train.create_mean()
Train.train()
#Train.train_gpu("--gpu 0,1,2,3")