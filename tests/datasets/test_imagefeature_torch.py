import pathlib
import unittest

import shift15m.constants as C
from shift15m.datasets import ImageFeatureDataset


class TestNumlikesTabular(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def test_dataset_corrupted(self):
        items = [("1000025", "10")]
        dataset = ImageFeatureDataset(items, pathlib.Path(C.FEATURE_ROOT))

        vec, label = dataset[0]
        self.assertEqual(len(vec), 4096)
        self.assertEqual(label, 0)
        self.assertEqual(dataset.category_size, 1)
        self.assertEqual(dataset.category_count[0], 1)

    def tearDown(self):
        return super().tearDown()
