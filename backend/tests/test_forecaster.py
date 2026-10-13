import unittest
from backend.app.ml.forecaster import FinancialForecaster


class TestForecaster(unittest.TestCase):
    def test_forecast_projection_bands(self):
        history = [10000.0, 12000.0, 14000.0, 16000.0, 18000.0]
        results = FinancialForecaster.forecast_cashflow(history, forecast_months=3)

        self.assertEqual(len(results["base_case"]), 3)
        self.assertEqual(len(results["best_case"]), 3)
        self.assertEqual(len(results["worst_case"]), 3)

        # Upper bound > base > lower bound
        for i in range(3):
            self.assertGreater(results["best_case"][i], results["base_case"][i])
            self.assertLess(results["worst_case"][i], results["base_case"][i])

    def test_volatility_score(self):
        stable = [100.0, 100.0, 100.0]
        self.assertEqual(FinancialForecaster.compute_volatility_score(stable), 0.0)


if __name__ == "__main__":
    unittest.main()
