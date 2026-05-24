from pydantic import BaseModel, model_validator
from uuid import UUID
from datetime import datetime

class IngestLog(BaseModel): 
    source_id: UUID
    instrument_id: UUID
    log_id: UUID
    ingested_at: datetime
    log_year: int = 0
    status: str
    record_count: int = 0
    error_message: str | None = None
    duration_ms: int | None = None

    @model_validator(mode="after")
    def set_log_year(self):
        if self.log_year == 0 and self.ingested_at is not None:
            self.log_year = self.ingested_at.year
        return self
