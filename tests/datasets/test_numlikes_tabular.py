import unittest
import shift28m.msgs as M
from shift28m.datasets import load_numlikes_tabular


class TestNumlikesTabular(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def test_dataset_corrupted(self):
        with self.assertRaises(RuntimeError, msg=M.DATASET_NOT_FOUND):
            load_numlikes_tabular("./dummy_dir")

    def tearDown(self):
        return super().tearDown()
