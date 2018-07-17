#!/usr/bin/env python
import unittest

from base.util import *

class TestUtilFunction(unittest.TestCase):
    def setUp(self):
        return

    def test_get_file_list(self):
        files = getFileList("/home/leon/temp/xinchu/Caffe_3CNN_TrainTest/original_data/head_backup/fzwkllwy0001")
        print(files.__len__())


if __name__ == '__main__':
    unittest.main()