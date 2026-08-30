from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from edgeforge_api.database import Base


class BenchmarkRecord(Base):
    __tablename__ = "benchmark_results"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    model_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    runtime_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    batch_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    mean_ms: Mapped[float] = mapped_column(Float, nullable=False)
    median_ms: Mapped[float] = mapped_column(Float, nullable=False)
    min_ms: Mapped[float] = mapped_column(Float, nullable=False)
    max_ms: Mapped[float] = mapped_column(Float, nullable=False)

    throughput_items_per_second: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
