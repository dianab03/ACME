"""
Worker process: consumes ingestion jobs from Redis/RQ and runs the pipeline.
Run with: python -m app.worker.worker
"""
import logging
import time
import uuid
from datetime import datetime, timezone

from redis import Redis
from rq import Connection, Queue, SimpleWorker

from app.config import settings
from app.db.connection import get_session
from app.db.repositories.data_source import DataSourceRepository
from app.db.repositories.ingest_job import IngestJobRepository
from app.db.repositories.ingest_log import IngestLogRepository
from app.db.repositories.instrument import InstrumentRepository
from app.db.repositories.time_series import TimeSeriesRepository
from app.db.schema import apply_schema
from app.ingestion.pipeline import IngestionPipeline
from app.models.data_source import DataSource
from app.models.ingest_job import IngestJob
from app.models.instrument import FinancialInstrument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NASDAQ_SOURCE_ID = uuid.uuid5(uuid.NAMESPACE_DNS, "acme.nasdaq_wiki_source")


def _build_pipeline(session) -> tuple[IngestionPipeline, IngestJobRepository]:
    pipeline = IngestionPipeline(
        instrument_repo=InstrumentRepository(session),
        source_repo=DataSourceRepository(session),
        ts_repo=TimeSeriesRepository(session),
        log_repo=IngestLogRepository(session),
    )
    job_repo = IngestJobRepository(session)
    return pipeline, job_repo


def process_job_message(message: dict) -> None:
    session = get_session()
    job_id = uuid.UUID(message["job_id"])
    symbol = message["symbol"]
    datatable_code = message.get("datatable_code", "WIKI/PRICES")
    instrument_class = message.get("instrument_class", "stock")
    region = message.get("region", "US")
    currency = message.get("currency", "USD")

    pipeline, job_repo = _build_pipeline(session)
    now = datetime.now(timezone.utc)

    job_repo.save(
        IngestJob(
            job_id=job_id,
            symbol=symbol,
            datatable_code=datatable_code,
            status="running",
            queued_at=now,
            started_at=now,
        )
    )

    source = DataSource(
        source_id=NASDAQ_SOURCE_ID,
        source_name="NASDAQ_WIKI",
        source_type="rest",
        base_url="https://data.nasdaq.com/api/v3/datatables",
        api_key_required=True,
        description="Nasdaq Data Link WIKI prices",
        created_at=now,
    )
    instrument = FinancialInstrument(
        instrument_id=uuid.uuid4(),
        symbol=symbol,
        instrument_class=instrument_class,
        name=symbol,
        region=region,
        currency=currency,
        created_at=now,
    )

    try:
        result = pipeline.ingest(
            instrument=instrument,
            source=source,
            datatable_code=datatable_code,
            filters={"ticker": symbol},
        )
        status = "completed" if not result.errors else "failed"
        error_msg = "; ".join(result.errors) if result.errors else None
        record_count = result.stored
    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
        record_count = 0

    completed_at = datetime.now(timezone.utc)
    job_repo.save(
        IngestJob(
            job_id=job_id,
            symbol=symbol,
            datatable_code=datatable_code,
            status=status,
            queued_at=now,
            started_at=now,
            completed_at=completed_at,
            record_count=record_count,
            error_message=error_msg,
        )
    )
    logger.info("Job %s for %s: %s (%d records)", job_id, symbol, status, record_count or 0)


def run_worker() -> None:
    session = None
    for attempt in range(1, 21):
        try:
            session = get_session()
            break
        except Exception as exc:
            if attempt == 20:
                raise
            logger.warning("Cassandra not ready (attempt %d/20): %s", attempt, exc)
            time.sleep(3)

    assert session is not None
    apply_schema(session, settings.cassandra_keyspace)

    redis_conn = Redis.from_url(settings.redis_url)
    with Connection(redis_conn):
        # Use non-forking worker mode for better stability with DB clients in this setup.
        worker = SimpleWorker([Queue("ingestion_jobs")])
        logger.info("RQ worker started. Waiting for jobs...")
        worker.work()


def main() -> None:
    run_worker()


if __name__ == "__main__":
    main()
