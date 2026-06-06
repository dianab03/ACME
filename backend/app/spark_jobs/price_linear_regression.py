from __future__ import annotations

import argparse
from typing import Sequence
from uuid import uuid4

from app.spark_jobs.common import (
    build_spark_session,
    filter_price_history,
    latest_daily_prices,
    parse_iso_date,
    read_cassandra_table,
    write_cassandra_table,
)

DEFAULT_INPUT_TABLE = "time_series_by_instrument"
DEFAULT_PREDICTION_TABLE = "close_price_predictions_by_instrument"
FEATURE_COLUMNS = [
    "day_index",
    "close_price",
    "lag_1_close",
    "lag_2_close",
    "lag_3_close",
    "volume",
]


def build_supervised_price_history(prices_df):
    from pyspark.sql import Window
    from pyspark.sql import functions as functions

    ordered_window = Window.partitionBy("instrument_id", "source_id").orderBy("record_date")

    return (
        prices_df.withColumn("day_index", functions.row_number().over(ordered_window).cast("double"))
        .withColumn("lag_1_close", functions.lag("close_price", 1).over(ordered_window))
        .withColumn("lag_2_close", functions.lag("close_price", 2).over(ordered_window))
        .withColumn("lag_3_close", functions.lag("close_price", 3).over(ordered_window))
        .withColumn("label", functions.lead("close_price", 1).over(ordered_window))
        .withColumn("volume", functions.coalesce(functions.col("volume"), functions.lit(0.0)))
    )


def training_rows(supervised_df):
    from pyspark.sql import functions as functions

    required_columns = [functions.col(column).isNotNull() for column in FEATURE_COLUMNS + ["label"]]
    condition = required_columns[0]
    for column_condition in required_columns[1:]:
        condition = condition & column_condition
    return supervised_df.where(condition)


def latest_feature_row(supervised_df):
    from pyspark.sql import functions as functions

    required_columns = [functions.col(column).isNotNull() for column in FEATURE_COLUMNS]
    condition = required_columns[0]
    for column_condition in required_columns[1:]:
        condition = condition & column_condition
    return supervised_df.where(condition).orderBy(functions.col("record_date").desc()).limit(1)


def train_and_predict(supervised_df, *, min_training_rows: int):
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import LinearRegression
    from pyspark.sql import functions as functions

    assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features")
    model_input = training_rows(supervised_df)
    row_count = model_input.count()
    if row_count < min_training_rows:
        raise ValueError(
            f"Need at least {min_training_rows} training rows after lag/label preparation; found {row_count}."
        )

    training = assembler.transform(model_input).select(
        "features", functions.col("label").cast("double").alias("label")
    )
    model = LinearRegression(featuresCol="features", labelCol="label", predictionCol="prediction").fit(training)
    fitted = model.transform(training)
    rmse = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse"
    ).evaluate(fitted)

    next_features = assembler.transform(latest_feature_row(supervised_df))
    prediction = model.transform(next_features)
    return model, prediction, row_count, rmse


def prediction_output(prediction_df, *, model_run_id: str, model_path: str, training_rows_count: int, rmse: float):
    from pyspark.sql import functions as functions

    return prediction_df.withColumn("prediction_generated_at", functions.current_timestamp()).select(
        "instrument_id",
        "source_id",
        "prediction_generated_at",
        functions.lit(model_run_id).alias("model_run_id"),
        functions.col("record_date").alias("last_record_date"),
        functions.col("prediction").cast("double").alias("predicted_next_close"),
        functions.lit(model_path).alias("model_path"),
        functions.lit(training_rows_count).cast("int").alias("training_rows"),
        functions.lit(float(rmse)).cast("double").alias("training_rmse"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train linear regression on close-price history, predict next close, and save the model."
    )
    parser.add_argument("--instrument-id", required=True, help="UUID of the instrument to train on.")
    parser.add_argument("--source-id", required=True, help="UUID of the data source to train on.")
    parser.add_argument("--start-date", help="Optional inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="Optional inclusive end date in YYYY-MM-DD format.")
    parser.add_argument("--model-path", required=True, help="Filesystem or distributed path where Spark saves the model.")
    parser.add_argument("--min-training-rows", type=int, default=10)
    parser.add_argument("--input-table", default=DEFAULT_INPUT_TABLE)
    parser.add_argument("--prediction-output-table", default=DEFAULT_PREDICTION_TABLE)
    parser.add_argument("--skip-prediction-write", action="store_true")
    parser.add_argument("--no-overwrite-model", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    spark = build_spark_session("close-price-linear-regression")
    try:
        raw_prices = read_cassandra_table(spark, args.input_table)
        prices = filter_price_history(
            latest_daily_prices(raw_prices),
            instrument_id=args.instrument_id,
            source_id=args.source_id,
            start_date=parse_iso_date(args.start_date),
            end_date=parse_iso_date(args.end_date),
        )
        supervised = build_supervised_price_history(prices)
        model, prediction, row_count, rmse = train_and_predict(
            supervised, min_training_rows=args.min_training_rows
        )

        writer = model.write()
        if not args.no_overwrite_model:
            writer = writer.overwrite()
        writer.save(args.model_path)

        prediction_row = prediction.select("record_date", "prediction").first()
        if prediction_row is None:
            raise ValueError("No row was available for next-close prediction.")

        model_run_id = str(uuid4())
        if not args.skip_prediction_write:
            output = prediction_output(
                prediction,
                model_run_id=model_run_id,
                model_path=args.model_path,
                training_rows_count=row_count,
                rmse=rmse,
            )
            write_cassandra_table(output, args.prediction_output_table)

        print(
            "Saved model to "
            f"{args.model_path}; predicted next close after {prediction_row.record_date}: "
            f"{prediction_row.prediction:.6f}; training RMSE={rmse:.6f}; run={model_run_id}."
        )
        return 0
    finally:
        spark.stop()


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

