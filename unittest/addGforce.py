"""
Unit Test for the `add_gforce` function in the `src.enrich` module.

Dependencies:
    - Install HTMLTestRunner: `pip install htmltestrunner-rv`
    - Assumes the `config.yml` file is present and contains the path for saving the test report.

@Lead Analyst: Ashkan Dashgtban    
"""

import unittest
import pandas as pd
import numpy as np
import yaml
from src.enrich import add_gforce
from HTMLTestRunner import HTMLTestRunner

# Load configuration from YAML file
with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)


class TestAddGForce(unittest.TestCase):
    """Test case for the add_gforce function in the enrich module."""

    def setUp(self):
        """
        Set up a sample DataFrame for testing.
        The DataFrame contains XYZ averages of accelerometer readings.
        """
        self.df = pd.DataFrame({
            'ID': [1, 2, 3],
            'averageX': [1.0, 2.0, 3.0],
            'averageY': [2.0, 3.0, 4.0],
            'averageZ': [3.0, 4.0, 5.0]
        })

    def test_add_gforce(self):
        """
        Test the add_gforce function to ensure it adds the 'gforce' column correctly
        and computes the values as expected.
        """
        # Call the function with the sample DataFrame
        df_with_gforce = add_gforce(self.df)

        # Check if the 'gforce' column is added
        self.assertIn('gforce', df_with_gforce.columns, "'gforce' column not found")

        # Check if the 'gforce' values are calculated correctly
        expected_gforce_values = [
            np.sqrt(1**2 + 2**2 + 3**2) / 9.81,
            np.sqrt(2**2 + 3**2 + 4**2) / 9.81,
            np.sqrt(3**2 + 4**2 + 5**2) / 9.81
        ]

        for idx, expected_value in enumerate(expected_gforce_values):
            actual_value = df_with_gforce.at[idx, 'gforce']
            self.assertAlmostEqual(actual_value, expected_value, places=6,
                                   msg=f"Mismatch in 'gforce' at index {idx}: expected {expected_value}, got {actual_value}")

    def tearDown(self):
        """
        Generate HTML report for the test results.
        """
        with open('test_report.html', 'w') as report_file:
            runner = HTMLTestRunner(stream=report_file, title='Unit Test Report', verbosity=2)
            runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(TestAddGForce))


if __name__ == '__main__':
    unittest.main(testRunner=HTMLTestRunner.HTMLTestRunner(output=config['path'] + './unittest/TestAddGForce.html'))
