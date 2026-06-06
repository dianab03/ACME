from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from uuid import UUID
from datetime import date
from decimal import Decimal

from app.db.connection import get_session
from app.db.repositories.time_series import TimeSeriesRepository
from app.analytics.aggregations import (
    compare_assets,
    compute_aggregations,
    compute_risk_signal,
    forecast_next_close,
    summarize_trend,
)

router = APIRouter()


class AnalyticsResponse(BaseModel):
    instrument_id: UUID
    source_id: UUID
    start_date: date
    end_date: date
    count: int
    min_close: Decimal | None = None
    max_close: Decimal | None = None
    avg_close: Decimal | None = None
    total_volume: int


class TrendResponse(BaseModel):
    instrument_id: UUID
    source_id: UUID
    start_date: date
    end_date: date
    direction: str
    start_close: Decimal | None = None
    end_close: Decimal | None = None
    change: Decimal | None = None
    change_pct: Decimal | None = None


class ForecastResponse(BaseModel):
    instrument_id: UUID
    source_id: UUID
    start_date: date
    end_date: date
    method: str
    predicted_next_close: Decimal | None = None


class RiskSignalResponse(BaseModel):
    instrument_id: UUID
    source_id: UUID
    start_date: date
    end_date: date
    signal: str
    volatility_pct: Decimal | None = None
    max_drawdown_pct: Decimal | None = None
    summary: str


class CompareResponse(BaseModel):
    source_id: UUID
    start_date: date
    end_date: date
    instrument_a_id: UUID
    instrument_b_id: UUID
    instrument_a_count: int
    instrument_b_count: int
    instrument_a_avg_close: Decimal | None = None
    instrument_b_avg_close: Decimal | None = None
    avg_close_diff: Decimal | None = None


def get_ts_repo() -> TimeSeriesRepository:
    return TimeSeriesRepository(get_session())


def _normalize_dates(start_date: date | None, end_date: date | None) -> tuple[date, date]:
    return start_date or date(2000, 1, 1), end_date or date.today()


@router.get("/{instrument_id}/{source_id}", response_model=AnalyticsResponse)
def get_analytics(
    instrument_id: UUID,
    source_id: UUID,
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
    repo: TimeSeriesRepository = Depends(get_ts_repo),
):
    """Return min/max/avg close price and total volume for an asset over a date range."""
    start_date, end_date = _normalize_dates(start_date, end_date)
    points = repo.find_range(instrument_id, source_id, start_date, end_date)
    agg = compute_aggregations(points)

    return AnalyticsResponse(
        instrument_id=instrument_id,
        source_id=source_id,
        start_date=start_date,
        end_date=end_date,
        count=agg.count,
        min_close=agg.min_close,
        max_close=agg.max_close,
        avg_close=agg.avg_close,
        total_volume=agg.total_volume,
    )


@router.get("/{instrument_id}/{source_id}/trend", response_model=TrendResponse)
def get_trend(
    instrument_id: UUID,
    source_id: UUID,
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
):
    repo = get_ts_repo()
    start_date, end_date = _normalize_dates(start_date, end_date)
    trend = summarize_trend(repo.find_range(instrument_id, source_id, start_date, end_date))
    return TrendResponse(
        instrument_id=instrument_id,
        source_id=source_id,
        start_date=start_date,
        end_date=end_date,
        direction=trend.direction,
        start_close=trend.start_close,
        end_close=trend.end_close,
        change=trend.change,
        change_pct=trend.change_pct,
    )


@router.get("/{instrument_id}/{source_id}/forecast", response_model=ForecastResponse)
def get_forecast(
    instrument_id: UUID,
    source_id: UUID,
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
):
    repo = get_ts_repo()
    start_date, end_date = _normalize_dates(start_date, end_date)
    forecast = forecast_next_close(repo.find_range(instrument_id, source_id, start_date, end_date))
    return ForecastResponse(
        instrument_id=instrument_id,
        source_id=source_id,
        start_date=start_date,
        end_date=end_date,
        method=forecast.method,
        predicted_next_close=forecast.predicted_next_close,
    )


@router.get("/{instrument_id}/{source_id}/risk-signal", response_model=RiskSignalResponse)
def get_risk_signal(
    instrument_id: UUID,
    source_id: UUID,
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
):
    repo = get_ts_repo()
    start_date, end_date = _normalize_dates(start_date, end_date)
    risk = compute_risk_signal(repo.find_range(instrument_id, source_id, start_date, end_date))
    return RiskSignalResponse(
        instrument_id=instrument_id,
        source_id=source_id,
        start_date=start_date,
        end_date=end_date,
        signal=risk.signal,
        volatility_pct=risk.volatility_pct,
        max_drawdown_pct=risk.max_drawdown_pct,
        summary=risk.summary,
    )


@router.get("/by-source/{source_id}/compare", response_model=CompareResponse)
def get_compare(
    source_id: UUID,
    instrument_a_id: UUID = Query(...),
    instrument_b_id: UUID = Query(...),
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
):
    repo = get_ts_repo()
    start_date, end_date = _normalize_dates(start_date, end_date)
    result = compare_assets(
        repo.find_range(instrument_a_id, source_id, start_date, end_date),
        repo.find_range(instrument_b_id, source_id, start_date, end_date),
    )
    agg_a = result["asset_a"]
    agg_b = result["asset_b"]
    return CompareResponse(
        source_id=source_id,
        start_date=start_date,
        end_date=end_date,
        instrument_a_id=instrument_a_id,
        instrument_b_id=instrument_b_id,
        instrument_a_count=agg_a.count,
        instrument_b_count=agg_b.count,
        instrument_a_avg_close=agg_a.avg_close,
        instrument_b_avg_close=agg_b.avg_close,
        avg_close_diff=result["avg_close_diff"],
    )
