from __future__ import annotations

import argparse
from typing import Sequence

from app.spark_jobs.common import (
    build_spark_session,
    filter_price_history,
    latest_daily_prices,
    parse_iso_date,
    read_cassandra_table,
    write_cassandra_table,
)

DEFAULT_INPUT_TABLE = "time_series_by_instrument"
DEFAULT_OUTPUT_TABLE = "daily_close_rolling_avg_by_instrument"


def compute_rolling_average(prices_df, window_days: int):
    from pyspark.sql import Window
    from pyspark.sql import functions as functions

    if window_days < 1:
        raise ValueError("window_days must be at least 1")

    rolling_window = (
        Window.partitionBy("instrument_id", "source_id")
        .orderBy("record_date")
        .rowsBetween(-(window_days - 1), 0)
    )

    return (
        prices_df.withColumn("rolling_avg_close", functions.avg("close_price").over(rolling_window))
        .withColumn("observation_count", functions.count("close_price").over(rolling_window).cast("int"))
        .withColumn("window_days", functions.lit(window_days).cast("int"))
        .withColumn("computed_at", functions.current_timestamp())
        .select(
            "instrument_id",
            "source_id",
            "window_days",
            "record_date",
            "close_price",
            "rolling_avg_close",
            "observation_count",
            "computed_at",
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute rolling daily close-price averages and persist them to Cassandra."
    )
    parser.add_argument("--instrument-id", help="Optional UUID filter for a single instrument.")
    parser.add_argument("--source-id", help="Optional UUID filter for a single data source.")
    parser.add_argument("--start-date", help="Optional inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="Optional inclusive end date in YYYY-MM-DD format.")
    parser.add_argument("--window-days", type=int, default=7, help="Rolling window size in observed trading days.")
    parser.add_argument("--input-table", default=DEFAULT_INPUT_TABLE)
    parser.add_argument("--output-table", default=DEFAULT_OUTPUT_TABLE)
    return parser


def run(args: argparse.Namespace) -> int:
    spark = build_spark_session("daily-close-rolling-average")
    try:
        raw_prices = read_cassandra_table(spark, args.input_table)
        prices = filter_price_history(
            latest_daily_prices(raw_prices),
            instrument_id=args.instrument_id,
            source_id=args.source_id,
            start_date=parse_iso_date(args.start_date),
            end_date=parse_iso_date(args.end_date),
        )
        result = compute_rolling_average(prices, args.window_days)
        row_count = result.count()
        write_cassandra_table(result, args.output_table)
        print(f"Wrote {row_count} rolling-average rows to {args.output_table}.")
        return 0
    finally:
        spark.stop()


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

