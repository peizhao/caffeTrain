#!/usr/bin/env python
import unittest
from base.config import *
from base.hisceneTrain import *

class TestHisceneTrainFunction(unittest.TestCase):
    def setUp(self):
        self.config = TrainConfig()
        self.config.set_output_root_folder("/home/leon/temp/data/output")
        self.config.set_origin_pos_image_root("/home/leon/temp/data/original_data/head_backup")
        self.config.set_origin_neg_image_root("/home/leon/temp/data/original_data/negative")
        self.config.train_pos_filter_list.append('fzwkllwy0001')
        self.config.train_neg_filter_list.append('17_neg')
        self.config.set_image_width(32)
        self.config.set_image_height(64)
        self.config.label_shape = (5,1,1)
        self.config.train_solver="/home/leon/temp/data/models/solver.prototxt"
        self.config.image_format='BGR'
        self.Train = hisceneTrain(self.config)

    def testGenerateTrainFile(self):
        self.Train.generate_train_file()

    def testGenerateTestFile(self):
        self.Train.generate_test_file()

    def testResize(self):
        self.Train.resize()
        self.Train.create_lmdb()
        self.Train.create_mean()
        self.Train.train()

if __name__ == '__main__':
    unittest.main()