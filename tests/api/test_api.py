from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from edgeforge_api.database import Base, get_db
from edgeforge_api.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    test_engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    testing_session = sessionmaker(
        bind=test_engine,
        autoflush=False,
        autocommit=False,
    )

    Base.metadata.create_all(bind=test_engine)

    def override_get_db() -> Generator[Session, None, None]:
        db = testing_session()

        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    yield TestClient(app)

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)


def benchmark_payload(
    runtime_id: str = "openvino_int8_cpu",
    mean_ms: float = 3.389,
) -> dict[str, object]:
    return {
        "model_id": "resnet18_imagenet",
        "runtime_id": runtime_id,
        "batch_size": 1,
        "mean_ms": mean_ms,
        "median_ms": mean_ms,
        "min_ms": mean_ms - 0.2,
        "max_ms": mean_ms + 0.3,
        "throughput_items_per_second": 1000 / mean_ms,
    }


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "connected",
    }


def test_create_benchmark(client: TestClient) -> None:
    response = client.post(
        "/benchmarks",
        json=benchmark_payload(),
    )

    assert response.status_code == 201

    data = response.json()

    assert data["id"] == 1
    assert data["model_id"] == "resnet18_imagenet"
    assert data["runtime_id"] == "openvino_int8_cpu"


def test_list_benchmarks(client: TestClient) -> None:
    client.post(
        "/benchmarks",
        json=benchmark_payload(),
    )

    response = client.get("/benchmarks")

    assert response.status_code == 200
    assert len(response.json()) == 1


def test_best_benchmark_returns_fastest_runtime(
    client: TestClient,
) -> None:
    client.post(
        "/benchmarks",
        json=benchmark_payload(
            runtime_id="pytorch_cpu",
            mean_ms=38.0,
        ),
    )

    client.post(
        "/benchmarks",
        json=benchmark_payload(
            runtime_id="openvino_int8_cpu",
            mean_ms=3.4,
        ),
    )

    response = client.get("/benchmarks/best/resnet18_imagenet")

    assert response.status_code == 200
    assert response.json()["runtime_id"] == "openvino_int8_cpu"


def test_best_benchmark_returns_404_when_missing(
    client: TestClient,
) -> None:
    response = client.get("/benchmarks/best/missing_model")

    assert response.status_code == 404
