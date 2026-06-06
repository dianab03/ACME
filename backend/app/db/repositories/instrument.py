import json
import uuid
from datetime import datetime, timezone
from typing import Iterable
from uuid import UUID

from app.db.repositories.base import CassandraRepository, WarehouseRepository
from app.models.instrument import FinancialInstrument, InstrumentSummary


class InstrumentRepository(CassandraRepository, WarehouseRepository[FinancialInstrument, UUID]):
    def __init__(self, session):
        super().__init__(session)
        self._insert = session.prepare(
            """
            INSERT INTO financial_instruments
              (instrument_id, symbol, instrument_class, name, region, currency,
               exchange_id, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        )
        self._select_by_id = session.prepare(
            "SELECT * FROM financial_instruments WHERE instrument_id = ?"
        )
        self._delete_by_id = session.prepare(
            "DELETE FROM financial_instruments WHERE instrument_id = ?"
        )
        self._select_all = session.prepare(
            "SELECT * FROM financial_instruments"
        )
        self._insert_version = session.prepare(
            """
            INSERT INTO instrument_versions
              (instrument_id, valid_from, version_id, change_type,
               is_delete_marker, snapshot, valid_to, changed_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
        )
        self._select_latest_version = session.prepare(
            "SELECT * FROM instrument_versions WHERE instrument_id = ? LIMIT 1"
        )

    def save(self, entity: FinancialInstrument) -> FinancialInstrument:
        existing = self._fetch_one(self._select_by_id, [entity.instrument_id])
        if existing is None:
            self._execute(self._insert, [
                entity.instrument_id, entity.symbol, entity.instrument_class,
                entity.name, entity.region, entity.currency,
                entity.exchange_id, entity.description, entity.created_at,
            ])
            change_type = "create"
        else:
            # Temporal behavior: append a new version instead of mutating base row.
            change_type = "update"

        snapshot = json.dumps(entity.model_dump(mode="json"))
        self._execute(self._insert_version, [
            entity.instrument_id,
            entity.created_at or datetime.now(timezone.utc),
            uuid.uuid4(),
            change_type,
            False,
            snapshot,
            None,
            "system",
        ])
        return entity

    def mark_deleted(self, key: UUID, valid_from: datetime | None = None, changed_by: str = "system") -> None:
        # Temporal delete: append marker version instead of deleting data.
        row = self._fetch_one(self._select_by_id, [key])
        if row is None:
            return
        self._execute(self._insert_version, [
            key,
            valid_from or datetime.now(timezone.utc),
            uuid.uuid4(),
            "delete_marker",
            True,
            None,
            None,
            changed_by,
        ])

    def delete(self, key: UUID) -> None:
        self.mark_deleted(key)

    def delete_all(self, partition_key: UUID) -> None:
        self.delete(partition_key)

    def find_latest(self, partition_key: UUID) -> FinancialInstrument | None:
        row = self._fetch_one(self._select_by_id, [partition_key])
        if row is None:
            return None
        latest_version = self._fetch_one(self._select_latest_version, [partition_key])
        if latest_version is None:
            return self._row_to_model(row)
        if getattr(latest_version, "is_delete_marker", False):
            return None
        snapshot = getattr(latest_version, "snapshot", None)
        if snapshot:
            return FinancialInstrument.model_validate(json.loads(snapshot))
        return self._row_to_model(row)

    def find_all(self, partition_key=None) -> Iterable[FinancialInstrument]:
        rows = self._fetch_all(self._select_all)
        instruments: list[FinancialInstrument] = []
        for row in rows:
            instrument = self.find_latest(row.instrument_id)
            if instrument is not None:
                instruments.append(instrument)
        return instruments

    def find_all_by_class(self, instrument_class: str) -> Iterable[InstrumentSummary]:
        rows = self._fetch_all(self._select_all)
        return [
            InstrumentSummary(
                instrument_id=r.instrument_id,
                symbol=r.symbol,
                instrument_class=r.instrument_class,
                name=r.name,
                region=r.region,
            )
            for r in rows
            if r.instrument_class == instrument_class
        ]

    @staticmethod
    def _row_to_model(row) -> FinancialInstrument:
        return FinancialInstrument(
            instrument_id=row.instrument_id,
            symbol=row.symbol,
            instrument_class=row.instrument_class,
            name=row.name,
            region=row.region,
            currency=row.currency,
            exchange_id=getattr(row, "exchange_id", None),
            description=getattr(row, "description", None),
            created_at=row.created_at,
        )
