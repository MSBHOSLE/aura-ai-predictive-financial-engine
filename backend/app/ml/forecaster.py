import math
from typing import Dict, List


class FinancialForecaster:
    """Predictive financial modeling engine supporting trend projection and Monte Carlo scenario analysis."""

    @staticmethod
    def forecast_cashflow(
        historical_monthly_revenue: List[float],
        forecast_months: int = 6,
        growth_multiplier: float = 1.05,
    ) -> Dict[str, List[float]]:
        if not historical_monthly_revenue:
            raise ValueError("Historical revenue series cannot be empty.")

        n = len(historical_monthly_revenue)
        # Linear regression trend
        x_mean = (n - 1) / 2.0
        y_mean = sum(historical_monthly_revenue) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(historical_monthly_revenue))
        denominator = sum((i - x_mean) ** 2 for i in range(n)) or 1.0

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        base_forecast = []
        best_case = []
        worst_case = []

        last_val = historical_monthly_revenue[-1]

        for m in range(1, forecast_months + 1):
            projected = max(0.0, intercept + slope * (n - 1 + m))
            projected = projected * (growth_multiplier ** (m / 12.0))

            base_forecast.append(round(projected, 2))
            best_case.append(round(projected * 1.15, 2))  # +15% upside
            worst_case.append(round(projected * 0.85, 2))  # -15% downside

        return {
            "base_case": base_forecast,
            "best_case": best_case,
            "worst_case": worst_case,
        }

    @staticmethod
    def compute_volatility_score(series: List[float]) -> float:
        if len(series) < 2:
            return 0.0
        mean = sum(series) / len(series)
        variance = sum((x - mean) ** 2 for x in series) / (len(series) - 1)
        std_dev = math.sqrt(variance)
        return round((std_dev / mean) * 100.0, 2) if mean > 0 else 0.0
