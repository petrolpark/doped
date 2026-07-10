import os
import unittest

from test_utils import data_dir

aims_data_dir = os.path.join(data_dir, "aims")


class AimsTest(unittest.TestCase):

    def setUp(self):
        self.CdTe_data_dir = os.path.join(aims_data_dir, "CdTe")

    def tearDown(self):
        pass