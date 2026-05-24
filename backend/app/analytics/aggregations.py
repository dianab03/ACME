from decimal import Decimal
from typing import Iterable
from app.models.time_series import TimeSeriesPoint

class AggregationResult:
    def __init__(self):
        self.count: int = 0
        self.min_close: Decimal | None = None
        self.max_close: Decimal | None = None
        self.avg_close: Decimal | None = None
        self.total_volume: int = 0

class TrendSummary:
    def __init__(self):
        self.direction: str = "flat"
        self.start_close: Decimal | None = None
        self.end_close: Decimal | None = None
        self.change: Decimal | None = None
        self.change_pct: Decimal | None = None

class ForecastResult:
    def __init__(self):
        self.method: str = "naive_last_close"
        self.predicted_next_close: Decimal | None = None

class RiskSignalResult:
    def __init__(self):
        self.signal: str = "unknown"
        self.volatility_pct: Decimal | None = None
        self.max_drawdown_pct: Decimal | None = None
        self.summary: str = ""


def _valid_close_prices(points: Iterable[TimeSeriesPoint]) -> list[Decimal]:
    ordered = sorted(points, key=lambda p: p.record_date)
    return [p.close_price for p in ordered if p.close_price is not None]


def compute_aggregations(points: Iterable[TimeSeriesPoint]) -> AggregationResult:
    result = AggregationResult()
    close_values = _valid_close_prices(points)
    total_volume = 0

    for point in points:
        result.count += 1
        if point.volume is not None:
            total_volume += point.volume
    
    if close_values:
        result.min_close = min(close_values)
        result.max_close = max(close_values)
        result.avg_close = sum(close_values) / len(close_values)
    
    result.total_volume = total_volume
    return result


def summarize_trend(points: Iterable[TimeSeriesPoint]) -> TrendSummary:
    result = TrendSummary()
    closes = _valid_close_prices(points)
    if len(closes) < 2:
        return result

    result.start_close = closes[0]
    result.end_close = closes[-1]
    result.change = result.end_close - result.start_close
    if result.start_close != 0:
        result.change_pct = (result.change / result.start_close) * Decimal("100")

    if result.change > 0:
        result.direction = "up"
    elif result.change < 0:
        result.direction = "down"
    else:
        result.direction = "flat"
    return result


def forecast_next_close(points: Iterable[TimeSeriesPoint]) -> ForecastResult:
    result = ForecastResult()
    closes = _valid_close_prices(points)
    if closes:
        result.predicted_next_close = closes[-1]
    return result


def compare_assets(points_a: Iterable[TimeSeriesPoint], points_b: Iterable[TimeSeriesPoint]) -> dict:
    agg_a = compute_aggregations(points_a)
    agg_b = compute_aggregations(points_b)
    a_avg = agg_a.avg_close
    b_avg = agg_b.avg_close
    avg_close_diff = None
    if a_avg is not None and b_avg is not None:
        avg_close_diff = a_avg - b_avg
    return {
        "asset_a": agg_a,
        "asset_b": agg_b,
        "avg_close_diff": avg_close_diff,
    }


def compute_risk_signal(points: Iterable[TimeSeriesPoint]) -> RiskSignalResult:
    result = RiskSignalResult()
    closes = _valid_close_prices(points)
    if len(closes) < 2:
        result.signal = "insufficient_data"
        result.summary = "Need at least two close values to estimate risk."
        return result

    returns_pct: list[Decimal] = []
    peak = closes[0]
    max_drawdown = Decimal("0")
    for idx in range(1, len(closes)):
        prev = closes[idx - 1]
        curr = closes[idx]
        if prev != 0:
            returns_pct.append(((curr - prev) / prev) * Decimal("100"))
        if curr > peak:
            peak = curr
        if peak != 0:
            drawdown = ((peak - curr) / peak) * Decimal("100")
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    if returns_pct:
        abs_returns = [abs(v) for v in returns_pct]
        result.volatility_pct = sum(abs_returns) / Decimal(len(abs_returns))
    result.max_drawdown_pct = max_drawdown

    volatility = result.volatility_pct or Decimal("0")
    drawdown = result.max_drawdown_pct or Decimal("0")
    if volatility >= Decimal("3") or drawdown >= Decimal("15"):
        result.signal = "high"
    elif volatility >= Decimal("1") or drawdown >= Decimal("5"):
        result.signal = "medium"
    else:
        result.signal = "low"
    result.summary = (
        f"Risk is {result.signal} "
        f"(volatility={volatility:.2f}%, max_drawdown={drawdown:.2f}%)."
    )
    return result
