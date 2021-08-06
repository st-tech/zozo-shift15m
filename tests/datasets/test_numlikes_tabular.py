import unittest
import shift15m.msgs as M
from shift15m.datasets import NumLikesRegression


class TestNumlikesTabular(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def test_dataset_corrupted(self):
        with self.assertRaises(RuntimeError, msg=M.DATASET_NOT_FOUND):
            NumLikesRegression(root="./dummy_dir")

    def tearDown(self):
        return super().tearDown()
