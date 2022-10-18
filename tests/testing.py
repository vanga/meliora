import unittest
import jeffreys_test


# create test case for Jeffreys test
class TestCases(unittest.TestCase):
    def load_pd_data(self):
        return pd.read_csv("pd_test_data.csv")

    def test_jeffreys(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""
        data = self.load_pd_data()
        result = jeffreys_test(data, "ratings", "default_flag", "predicted_pd")
        expected = [0.811206650001989, 0.317745098799562, 0.45107113090191414]
        self.assertAlmostEqual(result["A", "p_value"], expected[0])
        self.assertAlmostEqual(result["B", "p_value"], expected[1])
        self.assertAlmostEqual(result["C", "p_value"], expected[2])

    def test_binomial(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""
        data = self.load_pd_data()
        result = binomial_test(data, "ratings", "default_flag", "predicted_pd")
        actual = list(result["p_value"])
        expected = [0.8343581343848849, 0.33977518714781485, 0.4999238306728917]
        self.assertAlmostEqual(result["A", "p_value"], expected[0])
        self.assertAlmostEqual(result["B", "p_value"], expected[1])
        self.assertAlmostEqual(result["C", "p_value"], expected[2])

    def test_brier(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""
        data = self.load_pd_data()
        result = brier_score(data, "ratings", "default_flag", "predicted_pd")
        expected = 0.1152275
        self.assertAlmostEqual(result.loc["total", "brier_score"], expected)

    def test_spiegelhalter(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""
        data = self.load_pd_data()
        result = spiegelhalter_test(data, "ratings", "default_flag", "predicted_pd")
        # NOT EXTERNALLY VALIDATED
        expected = 0.867346545132708
        self.assertAlmostEqual(result.loc["total", "p_value"], expected)

    def test_hosmer(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""
        data = self.load_pd_data()
        result = hosmer_test(data, "ratings", "default_flag", "predicted_pd")
        expected = 0.3148128127815959
        self.assertAlmostEqual(result.loc["total", "p_value"], expected)

    def test_herfindahl(self):
        """Expected value calculation is described in the r_test_cases.ipynb"""
        data = self.load_pd_data()
        result = herfindahl_test(data, data, "ratings")
        # ONLY TESTING HERFINDAHL INDEX FOR NOW
        # -- p_value not externally validated
        expected = 0.421758
        self.assertAlmostEqual(result.loc["total", "h_initial"], expected)
