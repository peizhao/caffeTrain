#!/usr/bin/env python
import unittest
from base.config import TrainConfig

class TestConfigFunction(unittest.TestCase):
    def setUp(self):
        self.train = TrainConfig()

    def test_get_caffe_root(self):
        test_root = self.train.get_caffe_root();
        self.assertNotEqual("", test_root)

    def test_get_image_formate(self):
        self.assertEqual(self.train.get_image_format(),'RGB')

    def test_output_root(self):
        self.train.set_output_root_folder('test root')
        self.assertEqual('test root',self.train.get_output_root_folder())

    def test_image_size(self):
        self.train.set_image_width(500)
        self.train.set_image_height(700)
        self.assertEqual(500, self.train.get_image_width())
        self.assertEqual(700, self.train.get_image_height())

if __name__ == '__main__':
    unittest.main()
