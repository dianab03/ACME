from redis import Redis
from rq import Queue
from app.config import settings
from app.worker.worker import process_job_message

def publish_ingest_job(message: dict) -> None:
    conn = Redis.from_url(settings.redis_url)
    q = Queue("ingestion_jobs", connection=conn)
    q.enqueue(process_job_message, message)