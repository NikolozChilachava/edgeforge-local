from datetime import datetime

from pydantic import BaseModel, ConfigDict


class BenchmarkCreate(BaseModel):
    model_id: str
    runtime_id: str
    batch_size: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    throughput_items_per_second: float


class BenchmarkResponse(BenchmarkCreate):
    id: int
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )
