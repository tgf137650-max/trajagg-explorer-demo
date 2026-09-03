import unittest

from backend.trajagg_runtime import TEST_OFFSET, TEST_SIZE, parse_query_id


class QueryIdTests(unittest.TestCase):
    def test_accepts_query_and_trajectory_prefixes(self):
        self.assertEqual(parse_query_id("Q-03000"), (0, 3000))
        self.assertEqual(parse_query_id("trj-09999"), (6999, 9999))

    def test_accepts_bare_global_index(self):
        self.assertEqual(parse_query_id("4887"), (1887, 4887))

    def test_rejects_indices_outside_test_split(self):
        for query_id in ("Q-02999", "Q-10000", "not-an-id"):
            with self.subTest(query_id=query_id), self.assertRaises(ValueError):
                parse_query_id(query_id)

    def test_test_split_constants_cover_7000_rows(self):
        self.assertEqual(TEST_OFFSET, 3000)
        self.assertEqual(TEST_SIZE, 7000)


if __name__ == "__main__":
    unittest.main()
