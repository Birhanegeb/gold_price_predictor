import unittest

import pandas as pd

from tasks.merge_sentiment import add_next_day_target
from tasks.train_model import chronological_train_test_split


class TrainingDataTests(unittest.TestCase):
    def test_chronological_split_keeps_latest_rows_out_of_training(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10)[::-1],
                "close": range(10),
                "target": [0, 1] * 5,
            }
        )

        train_df, test_df = chronological_train_test_split(df)

        self.assertEqual(len(train_df), 8)
        self.assertEqual(len(test_df), 2)
        self.assertLess(train_df["date"].max(), test_df["date"].min())
        self.assertEqual(
            list(test_df["date"]),
            list(pd.date_range("2024-01-09", periods=2)),
        )

    def test_next_day_target_drops_row_without_a_future_close(self):
        df = pd.DataFrame({"close": [10.0, 12.0, 11.0]})

        result = add_next_day_target(df)

        self.assertEqual(result["target"].tolist(), [1, 0])
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
