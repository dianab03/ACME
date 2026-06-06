import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    cassandra_hosts: str = "localhost"
    cassandra_keyspace: str = "financial_dw"
    cassandra_port: int = 9042
    cassandra_dc: str = "datacenter1"
    cassandra_replication_strategy: str = "SimpleStrategy"
    cassandra_replication_factor: int = 1
    cassandra_replication_dcs: str = ""
    nasdaq_base_url: str = "https://data.nasdaq.com/api/v3/datatables"
    nasdaq_api_key: str = ""
    redis_url: str = "redis://redis:6379/0"
    ollama_base_url: str = "http://0.0.0.0:11434"
    ollama_model: str = "llama3.2"

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def cassandra_hosts_list(self) -> list[str]:
        return [h.strip() for h in self.cassandra_hosts.split(",")]
    
settings = Settings()

# Backwards-compat: allow older env var `OLLAMA_HOST` to specify the ollama base URL.
# If set (e.g. "0.0.0.0:11434" or "http://0.0.0.0:11434"), use it when `ollama_base_url`
# was not explicitly set via the expected env name.
if not settings.ollama_base_url:
    ollama_host_env = os.getenv("OLLAMA_HOST")
    if ollama_host_env:
        if ollama_host_env.startswith("http"):
            settings.ollama_base_url = ollama_host_env
        else:
            settings.ollama_base_url = f"http://{ollama_host_env}"
