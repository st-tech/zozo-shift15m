import gzip
import pathlib
import shutil
import unittest

try:
    from shift15m.datasets import ImageFeatureDataset

    SKIP = False
except ImportError:
    SKIP = True


class TestNumlikesTabular(unittest.TestCase):
    def setUp(self):
        pathlib.Path("test_features").mkdir(exist_ok=True)
        with gzip.open("test_features/xxxxx.json.gz", mode="wt") as fp:
            fp.write("[1, 2, 3]")
        return super().setUp()

    @unittest.skipIf(SKIP, "not supported")
    def test_dataset_corrupted(self):
        items = [("xxxxx", "10")]
        dataset = ImageFeatureDataset(items, pathlib.Path("test_features"))

        vec, label = dataset[0]
        self.assertEqual(len(vec), 3)
        self.assertEqual(label, 0)
        self.assertEqual(dataset.category_size, 1)
        self.assertEqual(dataset.category_count[0], 1)

    def tearDown(self):
        shutil.rmtree("test_features")
        return super().tearDown()
