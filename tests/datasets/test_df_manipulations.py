import unittest
import numpy as np
from shift15m.datasets import df_manipulations


class TestDFManipulations(unittest.TestCase):
    def setUp(self):
        # test cases for user_id
        self.extract_user_id_testcase1 = {"user_id": "10"}
        self.extract_user_id_testcase2 = {"item_id": "10"}

        # test cases for price
        self.price_testcase1 = [{"price": "1000"}, {"price": "5000"}]
        self.price_testcase2 = [{"price": "0"}]
        self.price_testcase3 = []

        # test cases for category
        self.category_testcase1 = [{"category_id1": 10}, {"category_id1": 2}]
        self.category_testcase2 = [{"category_id1": 1}]
        self.category_testcase3 = [{"category_id1": 1}, {"category_id1": 1}]
        self.category_testcase4 = []

    def test_extract_user_id(self):
        self.assertEqual(
            df_manipulations.extract_user_id(self.extract_user_id_testcase1), 10
        )
        self.assertEqual(
            df_manipulations.extract_user_id(self.extract_user_id_testcase2), -1
        )

    def test_price_sum(self):
        self.assertEqual(df_manipulations.price_sum(self.price_testcase1), 6000)
        self.assertEqual(df_manipulations.price_sum(self.price_testcase2), 0)
        self.assertEqual(df_manipulations.price_sum(self.price_testcase3), 0)

    def test_price_mean(self):
        self.assertEqual(df_manipulations.price_mean(self.price_testcase1), 3000)
        self.assertEqual(df_manipulations.price_mean(self.price_testcase2), 0)
        self.assertEqual(df_manipulations.price_mean(self.price_testcase3), 0)

    def test_price_max(self):
        self.assertEqual(df_manipulations.price_max(self.price_testcase1), 5000)
        self.assertEqual(df_manipulations.price_max(self.price_testcase2), 0)
        self.assertEqual(df_manipulations.price_max(self.price_testcase3), 0)

    def test_price_min(self):
        self.assertEqual(df_manipulations.price_min(self.price_testcase1), 1000)
        self.assertEqual(df_manipulations.price_min(self.price_testcase2), 0)
        self.assertEqual(df_manipulations.price_min(self.price_testcase3), 0)

    def test_categories_count_embedding_id1(self):
        self.assertTrue(
            np.array_equal(
                df_manipulations.categories_count_embedding_id1(
                    self.category_testcase1
                ),
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
        )
        self.assertTrue(
            np.array_equal(
                df_manipulations.categories_count_embedding_id1(
                    self.category_testcase2
                ),
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
        )
        self.assertTrue(
            np.array_equal(
                df_manipulations.categories_count_embedding_id1(
                    self.category_testcase3
                ),
                [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
        )
        self.assertTrue(
            np.array_equal(
                df_manipulations.categories_count_embedding_id1(
                    self.category_testcase4
                ),
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
        )
