import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import "./App.css";

type Benchmark = {
  id: number;
  model_id: string;
  runtime_id: string;
  batch_size: number;
  mean_ms: number;
  median_ms: number;
  min_ms: number;
  max_ms: number;
  throughput_items_per_second: number;
  created_at: string;
};
type Optimization = {
  model_id: string;
  fp32_latency_ms: number;
  int8_latency_ms: number;
  fp32_throughput: number;
  int8_throughput: number;
  fp32_size_mb: number;
  int8_size_mb: number;
  speedup: number;
  size_reduction_percent: number;
};

type SystemInfo = {
  operating_system: string;
  os_version: string;
  processor: string;
  cpu_cores: number;
  logical_cpus: number;
  memory_gb: number;
  cuda_available: boolean;
  gpu: string | null;
};
const API_URL = (
  import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

async function fetchJson<T>(path: string, label: string): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_URL}${path}`);
  } catch {
    throw new Error(`Cannot reach the EdgeForge API at ${API_URL}.`);
  }

  if (!response.ok) {
    throw new Error(`${label} request failed (${response.status}).`);
  }

  return response.json() as Promise<T>;
}

async function fetchOptionalJson<T>(
  path: string,
  label: string,
): Promise<T | null> {
  let response: Response;

  try {
    response = await fetch(`${API_URL}${path}`);
  } catch {
    throw new Error(`Cannot reach the EdgeForge API at ${API_URL}.`);
  }

  if (response.status === 404) {
    return null;
  }

  if (!response.ok) {
    throw new Error(`${label} request failed (${response.status}).`);
  }

  return response.json() as Promise<T>;
}

function formatRuntime(runtime: string) {
  return runtime
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function App() {
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([]);
  const [best, setBest] = useState<Benchmark | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [optimization, setOptimization] =
    useState<Optimization | null>(null);
  const [systemInfo, setSystemInfo] =
    useState<SystemInfo | null>(null);

  useEffect(() => {
    async function loadDashboard() {
      try {
        const [
          benchmarkData,
          bestData,
          optimizationData,
          systemData,
        ] = await Promise.all([
          fetchJson<Benchmark[]>("/benchmarks", "Benchmarks"),
          fetchOptionalJson<Benchmark>(
            "/benchmarks/best/resnet18_imagenet",
            "Best benchmark",
          ),
          fetchOptionalJson<Optimization>(
            "/optimization/resnet18",
            "Optimization report",
          ),
          fetchJson<SystemInfo>("/system", "System information"),
        ]);

        setBenchmarks(benchmarkData);
        setBest(bestData);
        setOptimization(optimizationData);
        setSystemInfo(systemData);
      } catch (loadError) {
        setError(
          loadError instanceof Error
            ? loadError.message
            : "Dashboard request failed.",
        );
      } finally {
        setLoading(false);
      }
    }

    loadDashboard();
  }, []);

  const latestByRuntime = useMemo(() => {
    const seen = new Set<string>();

    return benchmarks.filter((benchmark) => {
      if (seen.has(benchmark.runtime_id)) {
        return false;
      }

      seen.add(benchmark.runtime_id);
      return true;
    });
  }, [benchmarks]);

  const chartData = latestByRuntime.map((benchmark) => ({
    runtime: formatRuntime(benchmark.runtime_id),
    latency: Number(benchmark.mean_ms.toFixed(2)),
    throughput: Number(
      benchmark.throughput_items_per_second.toFixed(2),
    ),
  }));

  const fastestThroughput = Math.max(
    ...latestByRuntime.map(
      (benchmark) =>
        benchmark.throughput_items_per_second,
    ),
    0,
  );

  if (loading) {
    return (
      <main className="center-screen">
        <div className="loader" />
        <p>Loading EdgeForge benchmark data...</p>
      </main>
    );
  }

  if (error) {
    return (
      <main className="center-screen">
        <h1>EdgeForge</h1>
        <p className="error">{error}</p>
        <p>API: {API_URL}</p>
      </main>
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div>
          <div className="logo-mark">EF</div>

          <h1>EdgeForge</h1>
          <p className="subtitle">
            AI Deployment Optimizer
          </p>
        </div>

        <nav>
          <span className="nav-item active">
            Performance
          </span>
          <span className="nav-item">
            Experiments
          </span>
          <span className="nav-item">
            Hardware
          </span>
        </nav>

        <div className="sidebar-footer">
          <span className="status-dot" />
          API connected
        </div>
      </aside>

      <main className="content">
        <header className="topbar">
          <div>
            <p className="eyebrow">
              DEPLOYMENT ANALYSIS
            </p>

            <h2>ResNet-18 Performance</h2>

            <p>
              Compare PyTorch, ONNX Runtime and
              OpenVINO deployment configurations.
            </p>
          </div>

          <div className="model-chip">
            resnet18_imagenet
          </div>
        </header>

        <section className="metrics-grid">
          <article className="metric-card highlight">
            <span>Recommended Runtime</span>

            <strong>
              {best
                ? formatRuntime(best.runtime_id)
                : "N/A"}
            </strong>

            <small>
              Lowest measured mean latency
            </small>
          </article>

          <article className="metric-card">
            <span>Best Latency</span>

            <strong>
              {best
                ? `${best.mean_ms.toFixed(2)} ms`
                : "N/A"}
            </strong>

            <small>
              Batch size 1
            </small>
          </article>

          <article className="metric-card">
            <span>Peak Throughput</span>

            <strong>
              {fastestThroughput.toFixed(1)}
            </strong>

            <small>items / second</small>
          </article>

          <article className="metric-card">
            <span>Runtimes Tested</span>

            <strong>
              {latestByRuntime.length}
            </strong>

            <small>
              CPU and GPU configurations
            </small>
          </article>
        </section>
        <section className="optimization-grid">
          <article className="panel optimization-card">
            <p className="eyebrow">INT8 OPTIMIZATION</p>
            <h3>Post-Training Quantization</h3>

            <div className="optimization-number">
              {optimization
                ? `${optimization.speedup.toFixed(2)}×`
                : "—"}
            </div>

            <p className="optimization-label">
              faster than FP32
            </p>

            <div className="comparison-row">
              <span>FP32 latency</span>
              <strong>
                {optimization
                  ? `${optimization.fp32_latency_ms.toFixed(2)} ms`
                  : "—"}
              </strong>
            </div>

            <div className="comparison-row">
              <span>INT8 latency</span>
              <strong>
                {optimization
                  ? `${optimization.int8_latency_ms.toFixed(2)} ms`
                  : "—"}
              </strong>
            </div>
          </article>

          <article className="panel optimization-card">
            <p className="eyebrow">MODEL COMPRESSION</p>
            <h3>Deployment Footprint</h3>

            <div className="optimization-number">
              {optimization
                ? `${optimization.size_reduction_percent.toFixed(1)}%`
                : "—"}
            </div>

            <p className="optimization-label">
              smaller model
            </p>

            <div className="comparison-row">
              <span>FP32</span>
              <strong>
                {optimization
                  ? `${optimization.fp32_size_mb.toFixed(2)} MB`
                  : "—"}
              </strong>
            </div>

            <div className="comparison-row">
              <span>INT8</span>
              <strong>
                {optimization
                  ? `${optimization.int8_size_mb.toFixed(2)} MB`
                  : "—"}
              </strong>
            </div>
          </article>

          <article className="panel optimization-card">
            <p className="eyebrow">HOST HARDWARE</p>
            <h3>Benchmark Machine</h3>

            <div className="hardware-row">
              <span>GPU</span>
              <strong>
                {systemInfo?.gpu ?? "Not detected"}
              </strong>
            </div>

            <div className="hardware-row">
              <span>RAM</span>
              <strong>
                {systemInfo
                  ? `${systemInfo.memory_gb} GB`
                  : "—"}
              </strong>
            </div>

            <div className="hardware-row">
              <span>CPU cores</span>
              <strong>
                {systemInfo?.cpu_cores ?? "—"}
              </strong>
            </div>

            <div className="hardware-row">
              <span>CUDA</span>
              <strong>
                {systemInfo?.cuda_available
                  ? "Available"
                  : "Unavailable"}
              </strong>
            </div>
          </article>
        </section>

        <section className="panel">
          <div className="panel-heading">
            <div>
              <h3>Mean Inference Latency</h3>
              <p>
                Lower latency indicates faster inference.
              </p>
            </div>

            <span className="unit">milliseconds</span>
          </div>

          <div className="chart-container">
            {chartData.length > 0 ? (
              <ResponsiveContainer
                width="100%"
                height={320}
              >
                <BarChart data={chartData}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    vertical={false}
                  />

                  <XAxis
                    dataKey="runtime"
                    tick={{ fontSize: 11 }}
                  />

                  <YAxis />

                  <Tooltip />

                  <Bar
                    dataKey="latency"
                    radius={[7, 7, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state">
                No benchmark data yet. Run the benchmark workflow,
                then refresh this page.
              </div>
            )}
          </div>
        </section>

        <section className="panel">
          <div className="panel-heading">
            <div>
              <h3>Runtime Comparison</h3>
              <p>
                Latest benchmark for each deployment
                configuration.
              </p>
            </div>
          </div>

          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Runtime</th>
                  <th>Mean</th>
                  <th>Median</th>
                  <th>Minimum</th>
                  <th>Maximum</th>
                  <th>Throughput</th>
                </tr>
              </thead>

              <tbody>
                {latestByRuntime.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="empty-table">
                      No results have been published yet.
                    </td>
                  </tr>
                ) : (
                  latestByRuntime.map((benchmark) => (
                    <tr key={benchmark.runtime_id}>
                      <td className="runtime-cell">
                        {formatRuntime(
                          benchmark.runtime_id,
                        )}

                        {best?.runtime_id ===
                          benchmark.runtime_id && (
                            <span className="recommended">
                              BEST
                            </span>
                          )}
                      </td>

                      <td>
                        {benchmark.mean_ms.toFixed(2)} ms
                      </td>

                      <td>
                        {benchmark.median_ms.toFixed(2)} ms
                      </td>

                      <td>
                        {benchmark.min_ms.toFixed(2)} ms
                      </td>

                      <td>
                        {benchmark.max_ms.toFixed(2)} ms
                      </td>

                      <td>
                        {benchmark.throughput_items_per_second.toFixed(
                          1,
                        )}{" "}
                        /s
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
