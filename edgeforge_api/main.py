import json
import platform
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import psutil
import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.get("/system")
def system_information() -> dict[str, object]:
    memory = psutil.virtual_memory()

    gpu_name: str | None = None

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "operating_system": platform.system(),
        "os_version": platform.release(),
        "processor": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "memory_gb": round(memory.total / (1024**3), 1),
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_name,
    }


@app.get("/optimization/resnet18")
def resnet18_optimization() -> dict[str, object]:
    result_path = Path("artifacts/results/resnet18_int8_optimization.json")

    if not result_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Optimization report not found.",
        )

    return json.loads(result_path.read_text(encoding="utf-8"))


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
