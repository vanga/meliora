import meliora.meliora.val_tests as vt
import unittest
import pandas as pd


# create test case for Jeffreys test
class TestCases(unittest.TestCase):
    """ Create a unit test for all functions of the tests library """

    def load_pd_data(self):
        """ Load data for testing """
        return pd.read_csv("C:\\projects\\meliora\\data\\pd_test_data.csv")

    def test_jeffreys(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""

        # Real results
        data = self.load_pd_data()
        result = vt.jeffreys_test(data, "ratings", "default_flag", "predicted_pd")

        # Expected results (see R notebook for values)
        expected = [0.01995857, 0.84955196, 0.59864873]

        # Tests
        result = result.set_index(result["Rating class"])

        self.assertAlmostEqual(result.loc["A", "p_value"], expected[0])
        self.assertAlmostEqual(result.loc["B", "p_value"], expected[1])
        self.assertAlmostEqual(result.loc["C", "p_value"], expected[2])

    def test_binomial(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""

        # Real results
        data = self.load_pd_data()
        result = vt.binomial_test(data, "ratings", "default_flag", "predicted_pd")

        # Expected results (see R notebook for values)
        expected = [0.02389227, 0.86744061, 0.66055279]

        # Tests
        result = result.set_index(result["Rating class"])

        self.assertAlmostEqual(result.loc["A", "p_value"], expected[0])
        self.assertAlmostEqual(result.loc["B", "p_value"], expected[1])
        self.assertAlmostEqual(result.loc["C", "p_value"], expected[2])

    def test_brier(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""
        data = self.load_pd_data()
        result = vt.brier_score(data, "ratings", "default_flag", "predicted_pd")
        expected = 0.00128950849979173
        self.assertAlmostEqual(result, expected)

    def test_spiegelhalter(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""
        data = self.load_pd_data()
        result = vt.spiegelhalter_test(data, "ratings", "default_flag", "predicted_pd")
        # NOT EXTERNALLY VALIDATED
        expected = 0.867346545132708
        self.assertAlmostEqual(result.loc["total", "p_value"], expected)

    def test_hosmer(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""
        data = self.load_pd_data()
        result = vt.hosmer_test(data, "ratings", "default_flag", "predicted_pd")
        expected = 0.13025
        self.assertAlmostEqual(result[0], expected)

    def test_herfindahl(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""
        data = self.load_pd_data()
        result = vt.herfindahl_test(data, "ratings")
        # ONLY TESTING HERFINDAHL INDEX FOR NOW
        # -- p_value not externally validated
        expected = 0.408232
        self.assertAlmostEqual(result[1], expected)
