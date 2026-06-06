from fastapi import FastAPI
from app.api import instruments, data_sources, time_series, analytics, ingestion, logs, chat

app = FastAPI(
    title="Acme Ltd Data Warehouse",
    description="Acme Ltd - Financial data warehouse API",
    version="1.0.0",
)


app.include_router(instruments.router, prefix="/instruments", tags=["instruments"])
app.include_router(data_sources.router, prefix="/data-sources", tags=["data-sources"])
app.include_router(time_series.router, prefix="/time-series", tags=["time-series"])
app.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
app.include_router(ingestion.router, prefix="/ingestion", tags=["ingestion"])
app.include_router(logs.router, prefix="/ingestion/logs", tags=["ingestion-logs"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])


@app.get("/health")
def health():
    return {"status": "ok"}