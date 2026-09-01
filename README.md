# EdgeForge

[![CI](https://github.com/NikolozChilachava/edgeforge-local/actions/workflows/ci.yml/badge.svg)](https://github.com/NikolozChilachava/edgeforge-local/actions/workflows/ci.yml)

EdgeForge is a local machine-learning deployment lab. It runs the same model
with PyTorch, ONNX Runtime, and OpenVINO, measures performance, stores results,
and displays them in a React dashboard.

This guide starts with the easiest local setup. No PostgreSQL server or GPU is
required to open the API and dashboard.

## Quick start on Windows

### 1. Install the prerequisites

You need:

- [Git](https://git-scm.com/download/win)
- [Python 3.12](https://www.python.org/downloads/)
- [Node.js 24](https://nodejs.org/en/download)

Confirm that PowerShell can find them:

```powershell
git --version
py -3.12 --version
node --version
npm --version
```

### 2. Download EdgeForge

```powershell
git clone https://github.com/NikolozChilachava/edgeforge-local.git
cd edgeforge-local
```

If you already downloaded the project, open PowerShell in its folder and skip
the clone command.

### 3. Install the Python application

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The first installation can take several minutes because the machine-learning
packages are large.

### 4. Install the dashboard

```powershell
cd dashboard
npm ci
cd ..
```

### 5. Start the API

Keep this PowerShell window open:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn edgeforge_api.main:app --reload --host 127.0.0.1 --port 8000
```

EdgeForge uses a local SQLite file for this quick start. The database and tables
are created automatically.

### 6. Check the API

Open a second PowerShell window in the EdgeForge folder and run:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

The result should show:

```text
status    : ok
database  : connected
```

You can also open the interactive API documentation at
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 7. Start the dashboard

In the second PowerShell window:

```powershell
cd dashboard
npm run dev
```

Open the local address printed by Vite, normally
[http://localhost:5173](http://localhost:5173).

The first screen will show the computer information and empty benchmark cards.
That is expected until you generate and publish benchmark results.

Use `Ctrl+C` in each PowerShell window when you want to stop the servers.

## Add benchmark data

Keep the API running. In another PowerShell window, activate the Python
environment from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
```

Export the pretrained ResNet-18 model, run every runtime available on your
computer, and publish the results:

```powershell
python -m apps.export_resnet18_onnx
python -m apps.benchmark_resnet18
python -m worker.publish_results
```

Refresh the dashboard after publishing. CPU runtimes work without a GPU. CUDA
results are added automatically when compatible NVIDIA hardware and drivers are
available. The first export downloads the pretrained model weights.

To create the OpenVINO INT8 comparison shown in the optimization cards:

```powershell
python -m apps.quantize_resnet18_int8
python -m apps.compare_openvino_int8
```

The quantization step downloads CIFAR-10 calibration data. Refresh the dashboard
after the comparison finishes.

Generated models, datasets, and benchmark results stay under `artifacts/`. Git
ignores them because they are large and specific to one computer.

## Use PostgreSQL instead of SQLite

SQLite is convenient for a first run. Use PostgreSQL to run the complete
production-style architecture.

Install PostgreSQL 16 or later, then open its SQL Shell (`psql`) and run:

```sql
CREATE USER edgeforge WITH PASSWORD 'edgeforge_local';
CREATE DATABASE edgeforge OWNER edgeforge;
```

From the EdgeForge folder, create the local environment file:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set its contents to:

```dotenv
DATABASE_URL=postgresql+psycopg://edgeforge:edgeforge_local@localhost:5432/edgeforge
```

Save the file and restart the API. EdgeForge creates its tables automatically.
Run the health check again to confirm the connection. The real `.env` file is
ignored by Git and must never be committed.

## macOS and Linux notes

The application commands are the same, but create and activate the Python
environment with:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Check the API with:

```bash
curl http://127.0.0.1:8000/health
```

CUDA and OpenVINO device availability depends on the operating system, hardware,
and installed drivers.

## Troubleshooting

### PowerShell blocks environment activation

Allow scripts only for the current PowerShell window, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### The API does not start

- Make sure the command is running from the repository root.
- Confirm the virtual environment is active.
- If you configured PostgreSQL, make sure its service is running and the values
  in `.env` match the database user, password, port, and database name.
- For the simple SQLite setup, do not create `.env`.

### The dashboard says it cannot reach the API

- Keep the API command running in its own PowerShell window.
- Run the `/health` check above before opening the dashboard.
- Use the exact Vite address printed by `npm run dev`.
- EdgeForge allows the usual Vite ports `5173` and `5174`. Stop older Vite
  processes if it chooses a higher port.

### The dashboard has no benchmark results

This is normal on a fresh installation. Run the three commands in **Add
benchmark data**, keep the API running while publishing, and refresh the page.

### A CUDA runtime is missing

GPU results are optional. EdgeForge skips CUDA benchmarks when PyTorch cannot
detect a compatible NVIDIA GPU. CPU results remain available.

### `npm ci` reports an unsupported Node version

Install Node.js 24, open a new PowerShell window, and confirm `node --version`
before trying again.

## Run the quality checks

From the repository root:

```powershell
.\scripts\check.ps1
```

This runs Python linting, formatting checks, type checking, tests, the local
performance regression gate, dashboard linting, and a production build.

If this computer has not generated benchmark artifacts yet, run:

```powershell
.\scripts\check.ps1 -SkipPerformance
```

GitHub Actions runs the deterministic checks automatically on every push and
pull request. Hardware performance remains a local check because hosted runners
cannot reproduce the reference machine.

## Architecture

```mermaid
flowchart LR
    Model[PyTorch model] --> Export[ONNX export]
    Model --> PT[PyTorch runtime]
    Export --> ORT[ONNX Runtime]
    Export --> OV[OpenVINO]
    OV --> INT8[INT8 quantization]
    PT --> Bench[Benchmark engine]
    ORT --> Bench
    OV --> Bench
    INT8 --> Bench
    Bench --> Files[JSON results]
    Files --> Worker[Result publisher]
    Worker --> API[FastAPI]
    API --> DB[(SQLite or PostgreSQL)]
    DB --> API
    API --> UI[React dashboard]
```

## Project layout

```text
apps/              Runnable export, benchmark, and optimization workflows
dashboard/         React and TypeScript dashboard
edgeforge_api/     FastAPI and database application
edgeforge_core/    Models, runtimes, benchmarking, and validation
worker/            Benchmark result publisher
tests/             Unit and API tests
configs/           Tracked performance baseline
scripts/           Local quality and regression commands
artifacts/         Generated models, reports, and benchmark output
```

## API routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Verify API and database connectivity |
| `GET` | `/system` | Report host hardware information |
| `GET` | `/benchmarks` | List stored benchmark results |
| `POST` | `/benchmarks` | Store a benchmark result |
| `GET` | `/benchmarks/best/{model_id}` | Return the lowest-latency result |
| `GET` | `/optimization/resnet18` | Return the latest INT8 comparison |

## Reference results

These batch-size-one measurements were recorded on Windows 11 with 12 physical
CPU cores, 31.7 GB of memory, and an NVIDIA GeForce RTX 4060 Laptop GPU. Results
will vary with hardware, drivers, power settings, and background activity.

| Runtime | Mean latency | Throughput |
| --- | ---: | ---: |
| ONNX Runtime CUDA | 1.88 ms | 532.0 items/s |
| PyTorch CUDA | 2.55 ms | 392.3 items/s |
| ONNX Runtime CPU | 10.26 ms | 97.5 items/s |
| OpenVINO CPU | 15.18 ms | 65.9 items/s |
| OpenVINO GPU | 33.24 ms | 30.1 items/s |
| PyTorch CPU | 38.28 ms | 26.1 items/s |

OpenVINO INT8 quantization reduced the model from 44.71 MB to 11.40 MB, a 74.5%
reduction. The recorded comparison improved mean CPU latency from 16.04 ms to
3.41 ms, a 4.71x speedup.

## Current scope

EdgeForge currently focuses on ResNet-18 with batch size one and local execution.
Future work could add more model families, configurable batch sizes, experiment
history, containerized deployment, and remote workers.
