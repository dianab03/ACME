import logging
import re
from cassandra.cluster import Session

from app.config import settings

_log = logging.getLogger(__name__)

def _replication_cql() -> str:
    strategy = settings.cassandra_replication_strategy.strip()
    if strategy == "NetworkTopologyStrategy":
        dcs = [dc.strip() for dc in settings.cassandra_replication_dcs.split(",") if dc.strip()]
        if not dcs:
            dcs = [settings.cassandra_dc]
        replication = {"class": strategy, **{dc: settings.cassandra_replication_factor for dc in dcs}}
    else:
        replication = {"class": "SimpleStrategy", "replication_factor": settings.cassandra_replication_factor}
    return str(replication)


KEYSPACE_CQL = """
CREATE KEYSPACE IF NOT EXISTS {keyspace}
WITH replication = {replication};
"""

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS financial_instruments (
        instrument_id UUID PRIMARY KEY,
        symbol TEXT,
        instrument_class TEXT,
        name TEXT,
        region TEXT,
        currency TEXT,
        exchange_id UUID,
        description TEXT,
        created_at TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS instrument_versions (
        instrument_id UUID,
        valid_from TIMESTAMP,
        version_id UUID,
        change_type TEXT,
        is_delete_marker BOOLEAN,
        snapshot TEXT,
        valid_to TIMESTAMP,
        changed_by TEXT,
        PRIMARY KEY (instrument_id, valid_from)
    ) WITH CLUSTERING ORDER BY (valid_from DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS data_sources (
        source_id UUID PRIMARY KEY,
        source_name TEXT,
        source_type TEXT,
        base_url TEXT,
        api_key_required BOOLEAN,
        description TEXT,
        attributes SET<TEXT>,
        created_at TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS time_series_by_instrument (
        instrument_id UUID,
        source_id UUID,
        record_year INT,
        record_date DATE,
        system_date TIMESTAMP,
        open_price DECIMAL,
        close_price DECIMAL,
        high_price DECIMAL,
        low_price DECIMAL,
        adj_close DECIMAL,
        volume BIGINT,
        ex_dividend DECIMAL,
        split_ratio DECIMAL,
        extra_indicators MAP<TEXT, TEXT>,
        ingested_at TIMESTAMP,
        PRIMARY KEY ((instrument_id, source_id, record_year), record_date, system_date)
    ) WITH CLUSTERING ORDER BY (record_date DESC, system_date DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_close_rolling_avg_by_instrument (
        instrument_id UUID,
        source_id UUID,
        window_days INT,
        record_date DATE,
        close_price DOUBLE,
        rolling_avg_close DOUBLE,
        observation_count INT,
        computed_at TIMESTAMP,
        PRIMARY KEY ((instrument_id, source_id, window_days), record_date)
    ) WITH CLUSTERING ORDER BY (record_date DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS close_price_predictions_by_instrument (
        instrument_id UUID,
        source_id UUID,
        prediction_generated_at TIMESTAMP,
        model_run_id TEXT,
        last_record_date DATE,
        predicted_next_close DOUBLE,
        model_path TEXT,
        training_rows INT,
        training_rmse DOUBLE,
        PRIMARY KEY ((instrument_id, source_id), prediction_generated_at, model_run_id)
    ) WITH CLUSTERING ORDER BY (prediction_generated_at DESC, model_run_id ASC)
    """,
    """
    CREATE TABLE IF NOT EXISTS ingest_jobs (
        job_id UUID PRIMARY KEY,
        symbol TEXT,
        datatable_code TEXT,
        status TEXT,
        queued_at TIMESTAMP,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        record_count INT,
        error_message TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingest_log (
        source_id UUID,
        log_year INT,
        ingested_at TIMESTAMP,
        log_id UUID,
        instrument_id UUID,
        status TEXT,
        record_count INT,
        error_message TEXT,
        duration_ms BIGINT,
        PRIMARY KEY ((source_id, log_year), ingested_at)
    ) WITH CLUSTERING ORDER BY (ingested_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS llm_query_log (
        user_id UUID,
        session_id UUID,
        asked_at TIMESTAMP,
        query_id UUID,
        user_prompt TEXT,
        tools_invoked TEXT,
        llm_response TEXT,
        duration_ms BIGINT,
        PRIMARY KEY ((user_id, session_id), asked_at)
    ) WITH CLUSTERING ORDER BY (asked_at DESC)
    """,
]


def apply_schema(session: Session, keyspace: str) -> None:
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]{0,47}", keyspace):
        raise ValueError(f"Invalid keyspace name: {keyspace!r}")
    session.execute(KEYSPACE_CQL.format(keyspace=keyspace, replication=_replication_cql()))
    session.set_keyspace(keyspace)
    for i, stmt in enumerate(SCHEMA_STATEMENTS):
        try:
            session.execute(stmt)
        except Exception as exc:
            raise RuntimeError(
                f"Schema statement {i} failed: {stmt.strip()[:80]!r}"
            ) from exc
    _log.info("Schema applied to keyspace %r (%d tables)", keyspace, len(SCHEMA_STATEMENTS))
