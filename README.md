# DW_Acme-Ltd
Project for a fictional company (Acme Ltd) that wants to collect financial market data, keep it trustworthy over time, and use it to generate insights, risk views, and recommendations for customers.

## Simplified UC-Complete Scope

This project was simplified to the minimum architecture that still covers all required use cases:

1. UC1 Ingestion and provenance:
- Ingestion API: `POST /ingestion`
- Pipeline: extractor -> transformer -> Cassandra load
- Provenance kept through `source_id` in all time-series rows + `ingest_log`

2. UC2 REST API for warehouse data:
- `GET /instruments` (Q1)
- `GET /instruments/{instrument_id}` (Q2)
- `GET /data-sources` (Q3)
- `GET /data-sources/{source_id}` (Q4)
- `GET /time-series/{instrument_id}/{source_id}` (Q5)

3. UC3 Analytics:
- `GET /analytics/{instrument_id}/{source_id}` for count/min/max/avg/volume aggregations
- `GET /analytics/{instrument_id}/{source_id}/trend` for trend direction and percentage change
- `GET /analytics/{instrument_id}/{source_id}/forecast` for simple next-close forecast
- `GET /analytics/{instrument_id}/{source_id}/risk-signal` for risk classification (low/medium/high)
- `GET /analytics/by-source/{source_id}/compare?instrument_a_id=...&instrument_b_id=...` for asset comparison
- Spark DataFrame job computes rolling close-price averages and writes them to Cassandra
- Spark ML job trains a linear regression model on price history, predicts the next close, and saves the model

4. UC4 LLM + MCP-style tools:
- Chat API: `POST /chat`
- Tool layer: list assets, fetch details, fetch time series, aggregate analytics
- Responses are grounded in warehouse data via tool execution

## Cassandra Schema

Core tables only:
- `financial_instruments`
- `instrument_versions` (temporal history + delete markers)
- `data_sources`
- `time_series_by_instrument`
- `daily_close_rolling_avg_by_instrument`
- `close_price_predictions_by_instrument`
- `ingest_jobs`
- `ingest_log`
- `llm_query_log`

This removes portfolio/user/recommendation/risk-domain tables that are not mandatory for the assignment's required UCs.

## Temporal Instrument Behavior

The platform now applies temporal rules for instruments:
- No physical update of an existing instrument row; updates append a new row in `instrument_versions`.
- No physical delete; delete operations append a `delete_marker` version.
- Read APIs return the latest non-deleted version snapshot.

Relevant endpoints:
- `POST /instruments` create instrument + initial version
- `PUT /instruments/{instrument_id}` append updated version
- `DELETE /instruments/{instrument_id}` append delete marker

## Spark Jobs

Optional Spark dependencies are isolated in `backend/requirements-spark.txt` so the API image can stay lightweight. The jobs read Cassandra using the DataStax Spark Cassandra connector configured by `SPARK_CASSANDRA_CONNECTOR_PACKAGE` (default `com.datastax.spark:spark-cassandra-connector_2.12:3.5.1`).

From `backend`, install optional dependencies when running locally:

```powershell
pip install -r requirements-spark.txt
```

Compute a 7-observation rolling close average and persist rows to `daily_close_rolling_avg_by_instrument`:

```powershell
python -m app.spark_jobs.rolling_close_average --window-days 7
```

Limit the DataFrame job to one instrument/source and date range:

```powershell
python -m app.spark_jobs.rolling_close_average --instrument-id <instrument_uuid> --source-id <source_uuid> --start-date 2024-01-01 --end-date 2024-12-31 --window-days 30
```

Train a linear regression model for one instrument/source, predict the next close, save the Spark ML model, and persist the prediction to `close_price_predictions_by_instrument`:

```powershell
python -m app.spark_jobs.price_linear_regression --instrument-id <instrument_uuid> --source-id <source_uuid> --model-path ./models/<instrument_uuid>-linear-regression
```

Set `CASSANDRA_HOSTS`, `CASSANDRA_PORT`, and `CASSANDRA_KEYSPACE` when running outside Docker. With the provided compose file, Cassandra is exposed on host port `9043`.
