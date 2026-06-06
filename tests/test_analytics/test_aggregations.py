from decimal import Decimal
from datetime import date, datetime, timezone
from uuid import uuid4

from app.analytics.aggregations import (
    compare_assets,
    compute_aggregations,
    compute_risk_signal,
    forecast_next_close,
    summarize_trend,
)
from app.models.time_series import TimeSeriesPoint


def _make_point(**kwargs) -> TimeSeriesPoint:
    now = datetime.now(timezone.utc)
    defaults = dict(
        instrument_id=uuid4(),
        source_id=uuid4(),
        record_year=2024,
        record_date=date(2024, 1, 15),
        system_date=now,
        ingested_at=now,
    )
    return TimeSeriesPoint(**(defaults | kwargs))


def test_empty_input_returns_zero_count():
    result = compute_aggregations([])
    assert result.count == 0
    assert result.min_close is None
    assert result.max_close is None
    assert result.avg_close is None
    assert result.total_volume == 0


def test_single_point_computes_correct_stats():
    point = _make_point(close_price=Decimal("100.0"), volume=1000)
    result = compute_aggregations([point])
    assert result.count == 1
    assert result.min_close == Decimal("100.0")
    assert result.max_close == Decimal("100.0")
    assert result.avg_close == Decimal("100.0")
    assert result.total_volume == 1000


def test_multiple_points_computes_min_max_avg():
    points = [
        _make_point(close_price=Decimal("100.0"), volume=1000),
        _make_point(close_price=Decimal("200.0"), volume=2000),
        _make_point(close_price=Decimal("150.0"), volume=500),
    ]
    result = compute_aggregations(points)
    assert result.count == 3
    assert result.min_close == Decimal("100.0")
    assert result.max_close == Decimal("200.0")
    assert result.avg_close == Decimal("150.0")
    assert result.total_volume == 3500


def test_null_close_price_excluded_from_stats():
    points = [
        _make_point(close_price=None, volume=500),
        _make_point(close_price=Decimal("100.0"), volume=500),
    ]
    result = compute_aggregations(points)
    assert result.count == 2
    assert result.min_close == Decimal("100.0")
    assert result.max_close == Decimal("100.0")
    assert result.total_volume == 1000


def test_summarize_trend_detects_upward_direction():
    points = [
        _make_point(close_price=Decimal("100.0")),
        _make_point(close_price=Decimal("120.0")),
    ]
    result = summarize_trend(points)
    assert result.direction == "up"
    assert result.change == Decimal("20.0")


def test_forecast_uses_last_close():
    points = [
        _make_point(close_price=Decimal("101.0")),
        _make_point(close_price=Decimal("103.5")),
    ]
    result = forecast_next_close(points)
    assert result.predicted_next_close == Decimal("103.5")


def test_compare_assets_reports_avg_difference():
    a_points = [
        _make_point(close_price=Decimal("10.0")),
        _make_point(close_price=Decimal("14.0")),
    ]
    b_points = [
        _make_point(close_price=Decimal("8.0")),
        _make_point(close_price=Decimal("10.0")),
    ]
    result = compare_assets(a_points, b_points)
    assert result["asset_a"].avg_close == Decimal("12.0")
    assert result["asset_b"].avg_close == Decimal("9.0")
    assert result["avg_close_diff"] == Decimal("3.0")


def test_compute_risk_signal_classifies_high_when_drawdown_large():
    points = [
        _make_point(close_price=Decimal("100.0")),
        _make_point(close_price=Decimal("80.0")),
    ]
    result = compute_risk_signal(points)
    assert result.signal == "high"
    assert result.max_drawdown_pct == Decimal("20.0")
