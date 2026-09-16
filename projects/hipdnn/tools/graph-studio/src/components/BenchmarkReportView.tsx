import { useMemo, useState } from "react";
import {
  METRICS,
  VALIDATION_LABEL,
  formatBytes,
  pluginKind,
  summarize,
  validationState,
} from "../benchmark/metrics";
import type { BenchmarkReport, EngineResult, GraphResults } from "../benchmark/types";

/**
 * Benchmark report view: suite totals, a per-graph engine comparison chart for
 * a chosen metric, and the full result table. Read-only — the report is handed
 * in, so wiring it to a live benchmark run is a matter of changing the prop.
 */

const ALL_ENGINES = "";

interface BenchmarkReportViewProps {
  report: BenchmarkReport;
  /** Where the report came from, shown as the heading. */
  sourceLabel: string;
  /** Marks the view as showing bundled sample data rather than a real run. */
  sample?: boolean;
  /** Opens the tensor inspector for one row. Omitted hides the row action. */
  onOpenTensors?: (graph: GraphResults, row: EngineResult) => void;
}

export function BenchmarkReportView({
  report,
  sourceLabel,
  sample,
  onOpenTensors,
}: BenchmarkReportViewProps) {
  const { metadata, graphs } = report;
  const [graphName, setGraphName] = useState(graphs[0]?.graph_name ?? "");
  const [metricId, setMetricId] = useState(METRICS[0].id);
  const [engineFilter, setEngineFilter] = useState(ALL_ENGINES);

  const summary = useMemo(() => summarize(report), [report]);
  const metric = METRICS.find((m) => m.id === metricId) ?? METRICS[0];
  const graph = graphs.find((g) => g.graph_name === graphName) ?? graphs[0];
  const results = graph?.results ?? [];

  const shown = engineFilter ? results.filter((r) => r.provider === engineFilter) : results;

  // Bars are scaled by magnitude; the winner is picked by the metric's direction.
  const values = shown.map((r) => metric.value(r));
  const scale = Math.max(...values, 0);
  const best = values.length
    ? metric.higherIsBetter
      ? Math.max(...values)
      : Math.min(...values)
    : 0;

  return (
    <div className="report">
      <header className="report__head">
        <div className="report__ident">
          <div className="report__eyebrow">
            Suite report
            {sample && <span className="report__sample">sample data</span>}
          </div>
          <h2 className="report__title">{sourceLabel}</h2>
          <div className="report__sub">
            {metadata.gpu_model ?? "unknown GPU"}
            <span className="report__sep">/</span>
            {metadata.hostname}
          </div>
        </div>
        <div className="report__pills">
          <span className="report__pill" data-tone="ok">
            {metadata.pass_combinations} success
          </span>
          <span className="report__pill" data-tone="error">
            {metadata.error_combinations} error
          </span>
          <span className="report__pill" data-tone="muted">
            {metadata.skip_combinations} skipped
          </span>
        </div>
      </header>

      <div className="report__cards">
        <Card label="Graphs" value={summary.graphs} />
        <Card label="Result rows" value={summary.rows} />
        <Card label="Validation passed" value={summary.validationPassed} tone="ok" />
        <Card label="Validation failed" value={summary.validationFailed} tone="error" />
      </div>

      <details className="report__env">
        <summary>Environment &amp; report counts</summary>
        <dl className="report__env-grid">
          <Fact label="Timestamp" value={metadata.timestamp} />
          <Fact label="GPU arch" value={metadata.gpu_arch} />
          <Fact label="ROCm" value={metadata.rocm_version} />
          <Fact label="hipDNN" value={metadata.hipdnn_version} />
          <Fact label="Python" value={metadata.python_version} />
          <Fact label="Kernel" value={metadata.kernel_version} />
          <Fact label="CPU" value={metadata.cpu_model} />
          <Fact
            label="CPU cores"
            value={metadata.cpu_count === null ? null : `${metadata.cpu_count}`}
          />
          <Fact
            label="NUMA nodes"
            value={metadata.numa_nodes === null ? null : `${metadata.numa_nodes}`}
          />
          <Fact
            label="Host RAM"
            value={metadata.total_ram_gb === null ? null : `${metadata.total_ram_gb} GB`}
          />
          <Fact label="Total graphs" value={`${metadata.total_graphs}`} />
          <Fact label="Combinations" value={`${metadata.total_combinations}`} />
          <Fact label="Passed" value={`${metadata.pass_combinations}`} />
          <Fact label="Failed" value={`${metadata.fail_combinations}`} />
          <Fact label="Skipped" value={`${metadata.skip_combinations}`} />
          <Fact label="Errored" value={`${metadata.error_combinations}`} />
        </dl>
      </details>

      <section className="report__panel">
        <div className="report__panel-head">
          <div>
            <h3 className="report__panel-title">Graph comparison</h3>
            <p className="report__panel-sub">Select an engine to inspect its results.</p>
          </div>
          <span className="report__count">
            {shown.length} result{shown.length === 1 ? "" : "s"}
          </span>
        </div>

        <div className="report__filters">
          <label className="report__field">
            <span>Graph</span>
            <select value={graph?.graph_name ?? ""} onChange={(e) => setGraphName(e.target.value)}>
              {graphs.map((g) => (
                <option key={g.graph_name} value={g.graph_name}>
                  {g.graph_name}
                </option>
              ))}
            </select>
          </label>
          <label className="report__field">
            <span>Metric</span>
            <select value={metricId} onChange={(e) => setMetricId(e.target.value)}>
              {METRICS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <label className="report__field">
            <span>Engine filter</span>
            <select value={engineFilter} onChange={(e) => setEngineFilter(e.target.value)}>
              <option value={ALL_ENGINES}>All engines</option>
              {results.map((r) => (
                <option key={r.provider} value={r.provider}>
                  {r.provider}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="report__chart-head">
          <strong>{metric.label}</strong>
          <span>{metric.hint}</span>
        </div>

        <div className="report__bars">
          {shown.length === 0 ? (
            <div className="report__empty">No results for this selection.</div>
          ) : (
            shown.map((result) => {
              const value = metric.value(result);
              const state = validationState(result.correctness);
              return (
                <div className="bar" key={result.provider}>
                  <div className="bar__name">
                    <span className="bar__engine">{result.provider}</span>
                    <span className="bar__kind">{pluginKind(result.plugin_path)}</span>
                  </div>
                  <div className="bar__track">
                    <div
                      className="bar__fill"
                      data-best={value === best}
                      style={{ width: `${scale > 0 ? (value / scale) * 100 : 0}%` }}
                    />
                  </div>
                  <div className="bar__value">{metric.format(value)}</div>
                  <span className="badge" data-state={state}>
                    {VALIDATION_LABEL[state]}
                  </span>
                </div>
              );
            })
          )}
        </div>
      </section>

      <section className="report__panel">
        <div className="report__panel-head">
          <h3 className="report__panel-title">All engine results</h3>
          <span className="report__panel-sub">Execution and validation are separate.</span>
        </div>
        <div className="report__tablewrap">
          <table className="report__table">
            <thead>
              <tr>
                <th>Engine</th>
                <th>Status</th>
                <th className="num">Executions/s</th>
                <th className="num">GPU mean (ms)</th>
                <th className="num">GPU p95 (ms)</th>
                <th className="num">Host mean (ms)</th>
                <th className="num">TFLOP/s</th>
                <th className="num">GB/s</th>
                <th className="num">Build (ms)</th>
                <th className="num">Workspace</th>
                <th>Validation</th>
                {onOpenTensors && <th>Tensors</th>}
              </tr>
            </thead>
            <tbody>
              {results.map((r) => {
                const state = validationState(r.correctness);
                const mean = r.gpu_kernel_stats.mean_ms;
                return (
                  <tr key={r.provider}>
                    <td>
                      <div className="report__engine">{r.provider}</div>
                      <div className="report__engine-ver">{r.engine_version}</div>
                    </td>
                    <td>
                      <span className="badge" data-status={r.status}>
                        {r.status}
                      </span>
                    </td>
                    <td className="num">{mean > 0 ? (1000 / mean).toFixed(3) : "—"}</td>
                    <td className="num">{mean.toFixed(4)}</td>
                    <td className="num">{r.gpu_kernel_stats.p95_ms.toFixed(4)}</td>
                    <td className="num">{r.host_stats.mean_ms.toFixed(4)}</td>
                    <td className="num">{r.derived_tflops_per_s.toFixed(3)}</td>
                    <td className="num">{r.derived_gbytes_per_s.toFixed(3)}</td>
                    <td className="num">{r.cpu_build_time_ms.toFixed(3)}</td>
                    <td className="num">{formatBytes(r.workspace_bytes)}</td>
                    <td>
                      <span
                        className="badge"
                        data-state={state}
                        title={r.correctness.error_message ?? undefined}
                      >
                        {VALIDATION_LABEL[state]}
                      </span>
                    </td>
                    {onOpenTensors && (
                      <td>
                        {r.tensor_manifest || graph?.input_tensor_manifest ? (
                          <button
                            type="button"
                            className="report__rowaction"
                            onClick={() => graph && onOpenTensors(graph, r)}
                          >
                            Inspect
                          </button>
                        ) : (
                          <span className="report__engine-ver">not captured</span>
                        )}
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function Card({ label, value, tone }: { label: string; value: number; tone?: "ok" | "error" }) {
  return (
    <div className="report__card">
      <span className="report__card-label">{label}</span>
      <span className="report__card-value" data-tone={tone}>
        {value}
      </span>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string | null }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{value ?? "—"}</dd>
    </>
  );
}
