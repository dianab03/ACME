from datetime import date, datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock
from uuid import uuid4

from fastapi.testclient import TestClient

from app.api.analytics import get_ts_repo
from app.main import app
from app.models.time_series import TimeSeriesPoint

client = TestClient(app)


def _make_point(instrument_id, source_id, record_date, close_price, volume=1000) -> TimeSeriesPoint:
    now = datetime.now(timezone.utc)
    return TimeSeriesPoint(
        instrument_id=instrument_id,
        source_id=source_id,
        record_year=record_date.year,
        record_date=record_date,
        system_date=now,
        open_price=close_price,
        close_price=close_price,
        high_price=close_price,
        low_price=close_price,
        adj_close=close_price,
        volume=volume,
        ex_dividend=Decimal("0"),
        split_ratio=Decimal("1"),
        extra_indicators={},
        ingested_at=now,
    )


def test_get_trend_endpoint_returns_direction():
    instrument_id = uuid4()
    source_id = uuid4()
    repo = MagicMock()
    repo.find_range.return_value = [
        _make_point(instrument_id, source_id, date(2024, 1, 1), Decimal("100")),
        _make_point(instrument_id, source_id, date(2024, 1, 2), Decimal("110")),
    ]
    app.dependency_overrides[get_ts_repo] = lambda: repo
    try:
        response = client.get(f"/analytics/{instrument_id}/{source_id}/trend")
        assert response.status_code == 200
        assert response.json()["direction"] == "up"
    finally:
        app.dependency_overrides.pop(get_ts_repo, None)


def test_get_forecast_endpoint_returns_last_close():
    instrument_id = uuid4()
    source_id = uuid4()
    repo = MagicMock()
    repo.find_range.return_value = [
        _make_point(instrument_id, source_id, date(2024, 1, 1), Decimal("100")),
        _make_point(instrument_id, source_id, date(2024, 1, 2), Decimal("111")),
    ]
    app.dependency_overrides[get_ts_repo] = lambda: repo
    try:
        response = client.get(f"/analytics/{instrument_id}/{source_id}/forecast")
        assert response.status_code == 200
        assert response.json()["predicted_next_close"] == "111"
    finally:
        app.dependency_overrides.pop(get_ts_repo, None)


def test_get_risk_signal_endpoint_returns_signal():
    instrument_id = uuid4()
    source_id = uuid4()
    repo = MagicMock()
    repo.find_range.return_value = [
        _make_point(instrument_id, source_id, date(2024, 1, 1), Decimal("100")),
        _make_point(instrument_id, source_id, date(2024, 1, 2), Decimal("80")),
    ]
    app.dependency_overrides[get_ts_repo] = lambda: repo
    try:
        response = client.get(f"/analytics/{instrument_id}/{source_id}/risk-signal")
        assert response.status_code == 200
        assert response.json()["signal"] == "high"
    finally:
        app.dependency_overrides.pop(get_ts_repo, None)


def test_get_compare_endpoint_returns_difference():
    source_id = uuid4()
    instrument_a_id = uuid4()
    instrument_b_id = uuid4()
    repo = MagicMock()
    repo.find_range.side_effect = [
        [
            _make_point(instrument_a_id, source_id, date(2024, 1, 1), Decimal("10")),
            _make_point(instrument_a_id, source_id, date(2024, 1, 2), Decimal("14")),
        ],
        [
            _make_point(instrument_b_id, source_id, date(2024, 1, 1), Decimal("8")),
            _make_point(instrument_b_id, source_id, date(2024, 1, 2), Decimal("10")),
        ],
    ]
    app.dependency_overrides[get_ts_repo] = lambda: repo
    try:
        response = client.get(
            f"/analytics/compare/{source_id}",
            params={"instrument_a_id": str(instrument_a_id), "instrument_b_id": str(instrument_b_id)},
        )
        assert response.status_code == 200
        assert response.json()["avg_close_diff"] == "3"
    finally:
        app.dependency_overrides.pop(get_ts_repo, None)
