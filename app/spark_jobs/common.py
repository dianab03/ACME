from __future__ import annotations

import os
from datetime import date
from typing import Any

from app.config import settings

DEFAULT_CONNECTOR_PACKAGE = "com.datastax.spark:spark-cassandra-connector_2.12:3.5.1"


def parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def build_spark_session(app_name: str, *, include_connector_package: bool = True) -> Any:
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise RuntimeError(
            "PySpark is required for Spark jobs. Install backend/requirements-spark.txt "
            "or run the job with a Spark distribution that provides pyspark."
        ) from exc

    # Allow explicit environment overrides for connectivity (useful when Cassandra is
    # port-forwarded or running in Docker with non-standard ports).
    cassandra_host_env = os.getenv("CASSANDRA_HOST") or os.getenv("CASSANDRA_HOSTS")
    cassandra_port_env = os.getenv("CASSANDRA_PORT")

    cassandra_host_value = cassandra_host_env if cassandra_host_env is not None else ",".join(
        settings.cassandra_hosts_list
    )
    cassandra_port_value = cassandra_port_env if cassandra_port_env is not None else str(settings.cassandra_port)

    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.cassandra.connection.host", cassandra_host_value)
        .config("spark.cassandra.connection.port", str(cassandra_port_value))
    )

    connector_package = os.getenv("SPARK_CASSANDRA_CONNECTOR_PACKAGE", DEFAULT_CONNECTOR_PACKAGE)
    if include_connector_package and connector_package:
        builder = builder.config("spark.jars.packages", connector_package)

    return builder.getOrCreate()


def cassandra_options(table: str, keyspace: str | None = None) -> dict[str, str]:
    return {
        "keyspace": keyspace or settings.cassandra_keyspace,
        "table": table,
    }


def read_cassandra_table(spark: Any, table: str, keyspace: str | None = None) -> Any:
    return (
        spark.read.format("org.apache.spark.sql.cassandra")
        .options(**cassandra_options(table, keyspace))
        .load()
    )


def write_cassandra_table(df: Any, table: str, keyspace: str | None = None, mode: str = "append") -> None:
    (
        df.write.format("org.apache.spark.sql.cassandra")
        .options(**cassandra_options(table, keyspace))
        .mode(mode)
        .save()
    )


def latest_daily_prices(df: Any) -> Any:
    from pyspark.sql import Window
    from pyspark.sql import functions as functions

    latest_window = Window.partitionBy(
        "instrument_id", "source_id", "record_date"
    ).orderBy(functions.col("system_date").desc())

    return (
        df.select(
            "instrument_id",
            "source_id",
            functions.col("record_date").cast("date").alias("record_date"),
            "system_date",
            functions.col("close_price").cast("double").alias("close_price"),
            functions.col("volume").cast("double").alias("volume"),
        )
        .where(functions.col("close_price").isNotNull())
        .withColumn("row_number", functions.row_number().over(latest_window))
        .where(functions.col("row_number") == 1)
        .drop("row_number", "system_date")
    )


def filter_price_history(
    df: Any,
    *,
    instrument_id: str | None = None,
    source_id: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> Any:
    from pyspark.sql import functions as functions

    if instrument_id:
        df = df.where(functions.col("instrument_id").cast("string") == instrument_id)
    if source_id:
        df = df.where(functions.col("source_id").cast("string") == source_id)
    if start_date:
        df = df.where(functions.col("record_date") >= functions.lit(start_date.isoformat()).cast("date"))
    if end_date:
        df = df.where(functions.col("record_date") <= functions.lit(end_date.isoformat()).cast("date"))
    return df

