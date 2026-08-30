from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from edgeforge_api.database import Base, engine, get_db
from edgeforge_api.models import BenchmarkRecord
from edgeforge_api.schemas import BenchmarkCreate, BenchmarkResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="EdgeForge API",
    version="0.1.0",
    lifespan=lifespan,
)


DatabaseSession = Annotated[
    Session,
    Depends(get_db),
]


@app.get("/health")
def health(
    db: DatabaseSession,
) -> dict[str, str]:
    db.execute(text("SELECT 1"))

    return {
        "status": "ok",
        "database": "connected",
    }


@app.post(
    "/benchmarks",
    response_model=BenchmarkResponse,
    status_code=201,
)
def create_benchmark(
    payload: BenchmarkCreate,
    db: DatabaseSession,
) -> BenchmarkRecord:
    record = BenchmarkRecord(
        model_id=payload.model_id,
        runtime_id=payload.runtime_id,
        batch_size=payload.batch_size,
        mean_ms=payload.mean_ms,
        median_ms=payload.median_ms,
        min_ms=payload.min_ms,
        max_ms=payload.max_ms,
        throughput_items_per_second=(payload.throughput_items_per_second),
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return record


@app.get(
    "/benchmarks",
    response_model=list[BenchmarkResponse],
)
def list_benchmarks(
    db: DatabaseSession,
) -> list[BenchmarkRecord]:
    statement = select(BenchmarkRecord).order_by(BenchmarkRecord.created_at.desc())

    return list(db.scalars(statement).all())


@app.get(
    "/benchmarks/best/{model_id}",
    response_model=BenchmarkResponse,
)
def best_benchmark(
    model_id: str,
    db: DatabaseSession,
) -> BenchmarkRecord:
    statement = (
        select(BenchmarkRecord)
        .where(BenchmarkRecord.model_id == model_id)
        .order_by(BenchmarkRecord.mean_ms.asc())
        .limit(1)
    )

    record = db.scalar(statement)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="No benchmark results found for model.",
        )

    return record
