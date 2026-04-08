import unittest
import pandas as pd
import os
from src.data.data_collection import load_data   # replace with your actual module name


class TestLoadData(unittest.TestCase):

    def setUp(self):
        # Create a temporary CSV file
        self.test_file = "test_data.csv"
        self.df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        self.df.to_csv(self.test_file, index=False)

    def tearDown(self):
        # Remove the temporary file after test
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_load_data_success(self):
        loaded_df = load_data(self.test_file)
        pd.testing.assert_frame_equal(loaded_df, self.df)

    def test_load_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_data("non_existent.csv")


if __name__ == "__main__":
    unittest.main()