import meliora.meliora.core as vt
import unittest
import pandas as pd


# create test case for Jeffreys test
class TestCases(unittest.TestCase):
    """Create a unit test for all functions of the tests library"""

    def load_pd_data(self):
        """Load data for testing"""
        return pd.read_csv("C:\\projects\\meliora\\data\\pd_test_data.csv")

    def load_lgd_data(self):
        """Load data for testing"""
        return pd.read_csv("C:\\projects\\meliora\\data\\lgd_dataset.csv")

    def load_german_data(self):
        """Load data for testing"""
        return pd.read_csv("C:\\projects\\meliora\\data\\german_data.csv")

    def test_jeffreys(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""

        data = self.load_pd_data()
        result = vt.jeffreys_test(data, "ratings", "default_flag", "predicted_pd")

        # Expected results (see R notebook for values)
        expected = [0.01995857, 0.84955196, 0.59864873]

        result = result.set_index(result["Rating class"])
        self.assertAlmostEqual(result.loc["A", "p_value"], expected[0])
        self.assertAlmostEqual(result.loc["B", "p_value"], expected[1])
        self.assertAlmostEqual(result.loc["C", "p_value"], expected[2])

    def test_binomial(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""

        data = self.load_pd_data()
        result = vt.binomial_test(data, "ratings", "default_flag", "predicted_pd")

        # Expected results (see R notebook for values)
        expected = [0.02389227, 0.86744061, 0.66055279]

        result = result.set_index(result["Rating class"])
        self.assertAlmostEqual(result.loc["A", "p_value"], expected[0])
        self.assertAlmostEqual(result.loc["B", "p_value"], expected[1])
        self.assertAlmostEqual(result.loc["C", "p_value"], expected[2])

    def test_brier(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""
        data = self.load_pd_data()
        result = vt.brier_score(data, "ratings", "default_flag", "predicted_pd")

        # Expected results (see R notebook for values)
        expected = 0.00128950849979173

        self.assertAlmostEqual(result, expected)

    def test_spiegelhalter(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""
        data = self.load_pd_data()
        result = vt.spiegelhalter_test(data, "ratings", "default_flag", "predicted_pd")
        # TODO: check if this is correct
        expected = -0.6637590511485174
        self.assertAlmostEqual(result[0], expected)

    def test_hosmer(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""
        data = self.load_pd_data()
        result = vt.hosmer_test(data, "ratings", "default_flag", "predicted_pd")

        # Expected results (see R notebook for values)
        expected = 0.13025

        self.assertAlmostEqual(result[0], expected)

    def test_herfindahl(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""
        data = self.load_pd_data()
        result = vt.herfindahl_test(data, "ratings")

        # Expected results (see R notebook for values)
        expected = 0.408232

        self.assertAlmostEqual(result[1], expected)

    def test_roc_auc(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""
        data = self.load_pd_data()
        result = vt.roc_auc(data, "default_flag", "predicted_pd")

        # Expected results (see R notebook for values)
        expected = 0.500854754970242

        self.assertAlmostEqual(result, expected)

    def test_clar(self):
        data = self.load_lgd_data()
        result = vt.clar(data, "realised_outcome", "predicted_outcome")

        # Expected results (see R notebook for values)
        expected = 0.84

        self.assertAlmostEqual(result, expected)
    
    def test_loss_capture_ratio(self):
        pass

    def test_bayesian_error_rate(self):
        pass

    def test_calc_iv(self):
        """Information calculation is described in the r_test_cases.ipynb"""
        data = self.load_german_data()
        result = vt.calc_iv(data, "checkingstatus", "GoodCredit")

        # Expected results (see R notebook for values)
        expected = 0.308738892211175

        self.assertAlmostEqual(result[1], expected)

    def test_lgd_t_test(self):
        pass

    def test_migration_matrix_stability(self):
        pass
