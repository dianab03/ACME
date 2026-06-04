# Acme-Ltd
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
- Time series rows can be exported directly to Spark or other ML/analytics tools

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
