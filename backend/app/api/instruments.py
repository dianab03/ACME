from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from uuid import UUID
import uuid

from app.db.connection import get_session
from app.db.repositories.instrument import InstrumentRepository
from app.models.instrument import FinancialInstrument, InstrumentSummary

router = APIRouter()


def get_instrument_repo() -> InstrumentRepository:
    return InstrumentRepository(get_session())


class CreateInstrumentRequest(BaseModel):
    symbol: str
    instrument_class: str
    name: str
    region: str
    currency: str
    description: str | None = None
    exchange_id: UUID | None = None


class UpdateInstrumentRequest(BaseModel):
    symbol: str | None = None
    instrument_class: str | None = None
    name: str | None = None
    region: str | None = None
    currency: str | None = None
    description: str | None = None
    exchange_id: UUID | None = None


@router.get("", response_model=list[InstrumentSummary])
def list_instruments(
    page_size: int = Query(default=1000, ge=1, le=10000),
    repo: InstrumentRepository = Depends(get_instrument_repo),
):
    """Q1: Return limited info about all financial assets."""
    instruments = list(repo.find_all())[:page_size]
    return [
        InstrumentSummary(
            instrument_id=i.instrument_id,
            symbol=i.symbol,
            instrument_class=i.instrument_class,
            name=i.name,
            region=i.region,
        )
        for i in instruments
    ]


@router.get("/{instrument_id}", response_model=FinancialInstrument)
def get_instrument(
    instrument_id: UUID,
    repo: InstrumentRepository = Depends(get_instrument_repo),
):
    """Q2: Return all details of an asset by identifier."""
    instrument = repo.find_latest(instrument_id)
    if instrument is None:
        raise HTTPException(status_code=404, detail="Instrument not found")
    return instrument


@router.post("", response_model=FinancialInstrument)
def create_instrument(
    request: CreateInstrumentRequest,
    repo: InstrumentRepository = Depends(get_instrument_repo),
):
    now = datetime.now(timezone.utc)
    instrument = FinancialInstrument(
        instrument_id=uuid.uuid4(),
        symbol=request.symbol,
        instrument_class=request.instrument_class,
        name=request.name,
        region=request.region,
        currency=request.currency,
        description=request.description,
        exchange_id=request.exchange_id,
        created_at=now,
    )
    return repo.save(instrument)


@router.put("/{instrument_id}", response_model=FinancialInstrument)
def update_instrument(
    instrument_id: UUID,
    request: UpdateInstrumentRequest,
    repo: InstrumentRepository = Depends(get_instrument_repo),
):
    current = repo.find_latest(instrument_id)
    if current is None:
        raise HTTPException(status_code=404, detail="Instrument not found")

    updated = current.model_copy(
        update={
            "symbol": request.symbol or current.symbol,
            "instrument_class": request.instrument_class or current.instrument_class,
            "name": request.name or current.name,
            "region": request.region or current.region,
            "currency": request.currency or current.currency,
            "description": request.description if request.description is not None else current.description,
            "exchange_id": request.exchange_id if request.exchange_id is not None else current.exchange_id,
            "created_at": datetime.now(timezone.utc),
        }
    )
    return repo.save(updated)


@router.delete("/{instrument_id}", status_code=204)
def mark_instrument_deleted(
    instrument_id: UUID,
    valid_from: datetime | None = Query(default=None),
    repo: InstrumentRepository = Depends(get_instrument_repo),
):
    current = repo.find_latest(instrument_id)
    if current is None:
        raise HTTPException(status_code=404, detail="Instrument not found")
    repo.mark_deleted(instrument_id, valid_from=valid_from)
