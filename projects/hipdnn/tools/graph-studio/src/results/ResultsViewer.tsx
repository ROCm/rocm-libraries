/**
 * Renderer-only results presentation: summary, per-graph comparison, and
 * engine detail drill-down for a single `ResultDocument`. No React Flow,
 * engine runtime, filesystem, autosave, or command-execution imports here —
 * the only trace integration is the renderer-local `PerfettoFrame`.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import type {
  ComparisonMetric,
  CorrectnessState,
  ResultDocument,
  ResultGraph,
  ResultRow,
  TimingStats,
} from "./model";
import { asResultObject, comparisonValue } from "./model";
import { PerfettoFrame } from "./PerfettoFrame";

// ---------------------------------------------------------------------------
// Numeric guards — every magnitude pulled from producer-supplied `details` or
// `metadata` must be validated before it drives display width or arithmetic.
// Signed values (correctness diffs, oracle delta_ms) get their own helper.
// ---------------------------------------------------------------------------

function finiteNumber(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function finiteNonNegative(v: unknown): number | null {
  const n = finiteNumber(v);
  return n !== null && n >= 0 ? n : null;
}

const numberFormatter = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 3,
});
function formatNumber(v: number): string {
  return numberFormatter.format(v);
}

/** Renders a value that is either a finite number or Unavailable. Never a raw NaN/undefined. */
function formatMagnitude(v: unknown): string {
  const n = finiteNonNegative(v);
  return n === null ? "Unavailable" : formatNumber(n);
}

/**
 * Correctness difference fields are signed and may be legitimately infinite
 * (the producer emits bare Infinity on shape mismatch / non-finite tensors).
 * Preserve that literal instead of ever showing a fabricated zero or a
 * silently dropped comparison.
 */
function formatSignedOrInfinite(v: unknown): string {
  if (typeof v !== "number") return "Unavailable";
  if (v === Infinity) return "Infinity";
  if (v === -Infinity) return "-Infinity";
  if (Number.isFinite(v)) return formatNumber(v);
  return "Unavailable";
}

/** Signed but must stay finite (e.g. oracle_delta.delta_ms, which may regress negative). */
function formatSignedFinite(v: unknown): string {
  const n = finiteNumber(v);
  return n === null ? "Unavailable" : formatNumber(n);
}

/** Preserve diagnostic non-finite values as explicit JSON strings. */
function stringifyDetails(value: unknown): string {
  return (
    JSON.stringify(
      value,
      (_key, v) =>
        typeof v === "number" && !Number.isFinite(v) ? String(v) : v,
      2,
    ) ?? "null"
  );
}

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------

const STATUS_LABEL: Record<ResultRow["status"], string> = {
  success: "Success",
  error: "Error",
  skipped: "Skipped",
};

const CORRECTNESS_LABEL: Record<CorrectnessState, string> = {
  passed: "Passed",
  failed: "Failed",
  "not-checked": "Not checked",
  "not-run": "Not run",
  "validation-error": "Validation error",
  reference: "Reference",
};

const DOCUMENT_KIND_LABEL: Record<ResultDocument["kind"], string> = {
  suite: "Suite report",
  raw: "Raw timing report",
  native: "Native execution",
};

const STAT_FIELDS: { key: keyof TimingStats; label: string }[] = [
  { key: "mean_ms", label: "Mean" },
  { key: "median_ms", label: "Median" },
  { key: "std_ms", label: "Std dev" },
  { key: "min_ms", label: "Min" },
  { key: "max_ms", label: "Max" },
  { key: "p95_ms", label: "p95" },
  { key: "p99_ms", label: "p99" },
  { key: "total_ms", label: "Total" },
];

const METRICS: { key: ComparisonMetric; label: string; caption: string }[] = [
  {
    key: "executions-per-second",
    label: "Graph executions/s",
    caption: "Derived from GPU mean; higher is better",
  },
  { key: "gpu-mean-ms", label: "GPU mean (ms)", caption: "Lower is better" },
  {
    key: "tflops",
    label: "Analytical TFLOP/s",
    caption: "Emitted analytical throughput; never recomputed",
  },
  {
    key: "gbytes",
    label: "Analytical GB/s",
    caption: "Analytical I/O divided by GPU time, not measured memory traffic",
  },
];

const HEADER_METADATA_FIELDS: { key: string; label: string }[] = [
  { key: "timestamp", label: "Timestamp" },
  { key: "hostname", label: "Hostname" },
  { key: "gpu_model", label: "GPU model" },
  { key: "gpu_arch", label: "GPU architecture" },
  { key: "rocm_version", label: "ROCm" },
  { key: "cuda_version", label: "CUDA" },
  { key: "hipdnn_version", label: "hipDNN" },
  { key: "pytorch_version", label: "PyTorch" },
  {
    key: "pytorch_sdpa_backend_requested",
    label: "Requested PyTorch SDPA backend (not the executed kernel)",
  },
  {
    key: "pytorch_rocm_fa_library_requested",
    label: "Requested PyTorch ROCm Flash Attention library",
  },
];

function metadataFieldText(
  metadata: Readonly<Record<string, unknown>>,
  key: string,
): string {
  const v = metadata[key];
  if (v === undefined || v === null || v === "") return "Unavailable";
  if (typeof v === "string" || typeof v === "number" || typeof v === "boolean")
    return String(v);
  const obj = asResultObject(v);
  return obj ? stringifyDetails(obj) : "Unavailable";
}

// ---------------------------------------------------------------------------
// Graph label disambiguation
// ---------------------------------------------------------------------------

function computeGraphLabels(
  graphs: readonly ResultGraph[],
): Map<string, string> {
  const nameCounts = new Map<string, number>();
  for (const g of graphs)
    nameCounts.set(g.name, (nameCounts.get(g.name) ?? 0) + 1);
  const seen = new Map<string, number>();
  const labels = new Map<string, string>();
  for (const g of graphs) {
    const total = nameCounts.get(g.name) ?? 1;
    if (total <= 1) {
      labels.set(g.key, g.name);
      continue;
    }
    const ordinal = (seen.get(g.name) ?? 0) + 1;
    seen.set(g.name, ordinal);
    labels.set(g.key, `${g.name} #${ordinal} — ${g.path ?? "no path"}`);
  }
  return labels;
}

// ---------------------------------------------------------------------------
// Engine grouping (provider, engineId, pluginPath, engineVersion, role)
// ---------------------------------------------------------------------------

interface EngineGroup {
  readonly key: string;
  readonly provider: string;
  readonly engineId: string | null;
  readonly engineVersion: string | null;
  readonly pluginPath: string | null;
  readonly role: "engine" | "reference" | null;
  readonly rows: ResultRow[];
  readonly graphKeys: Set<string>;
}

function groupKeyFor(row: ResultRow): string {
  return JSON.stringify([
    row.provider,
    row.engineId,
    row.pluginPath,
    row.engineVersion,
    row.role,
  ]);
}

function buildEngineGroups(document: ResultDocument): EngineGroup[] {
  const groups = new Map<string, EngineGroup>();
  for (const graph of document.graphs) {
    for (const row of graph.rows) {
      const key = groupKeyFor(row);
      let group = groups.get(key);
      if (!group) {
        group = {
          key,
          provider: row.provider,
          engineId: row.engineId,
          engineVersion: row.engineVersion,
          pluginPath: row.pluginPath,
          role: row.role,
          rows: [],
          graphKeys: new Set(),
        };
        groups.set(key, group);
      }
      group.rows.push(row);
      group.graphKeys.add(graph.key);
    }
  }
  return [...groups.values()];
}

function countByCorrectness(
  rows: readonly ResultRow[],
): Partial<Record<CorrectnessState, number>> {
  const counts: Partial<Record<CorrectnessState, number>> = {};
  for (const row of rows)
    counts[row.correctness] = (counts[row.correctness] ?? 0) + 1;
  return counts;
}

// ---------------------------------------------------------------------------
// Small shared bits
// ---------------------------------------------------------------------------

function Badge({
  kind,
  state,
  text,
}: {
  kind: "status" | "correctness";
  state: string;
  text: string;
}) {
  return (
    <span className="results__badge" data-kind={kind} data-state={state}>
      {text}
    </span>
  );
}

function StatsTable({
  title,
  stats,
}: {
  title: string;
  stats: TimingStats | null;
}) {
  return (
    <div className="results__timing-card">
      <h4>{title}</h4>
      {stats ? (
        <table className="results__stats-table">
          <thead>
            <tr>
              <th scope="col">Statistic</th>
              <th scope="col">Time (ms)</th>
            </tr>
          </thead>
          <tbody>
            {STAT_FIELDS.map((field) => (
              <tr key={field.key}>
                <th scope="row">{field.label}</th>
                <td>{formatMagnitude(stats[field.key])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="results__unavailable">Unavailable</p>
      )}
    </div>
  );
}

function Disclosure({
  summary,
  children,
}: {
  summary: string;
  children: React.ReactNode;
}) {
  return (
    <details className="results__disclosure">
      <summary>{summary}</summary>
      {children}
    </details>
  );
}

// ---------------------------------------------------------------------------
// ResultsSummary — header, counters, and the grouped engine table
// ---------------------------------------------------------------------------

function ResultsSummary({ document }: { document: ResultDocument }) {
  let totalRows = 0;
  const correctnessCounts: Partial<Record<CorrectnessState, number>> = {};
  const statusCounts = { success: 0, error: 0, skipped: 0 };
  for (const graph of document.graphs) {
    totalRows += graph.rows.length;
    for (const row of graph.rows) {
      statusCounts[row.status]++;
      correctnessCounts[row.correctness] =
        (correctnessCounts[row.correctness] ?? 0) + 1;
    }
  }
  const combinationEntries = Object.entries(document.metadata).filter(([key]) =>
    key.endsWith("_combinations"),
  );

  return (
    <section className="results__report">
      <header className="results__report-heading">
        <div>
          <span className="results__eyebrow">
            {DOCUMENT_KIND_LABEL[document.kind]}
          </span>
          <h1 className="results__title">{document.label}</h1>
          <p className="results__report-context">
            {metadataFieldText(document.metadata, "gpu_model")}{" "}
            <span aria-hidden="true">/</span>{" "}
            {metadataFieldText(document.metadata, "hostname")}
          </p>
        </div>
        <div
          className="results__report-status"
          aria-label="Execution status counts"
        >
          {(Object.keys(STATUS_LABEL) as ResultRow["status"][]).map(
            (status) => (
              <Badge
                key={status}
                kind="status"
                state={status}
                text={`${statusCounts[status]} ${STATUS_LABEL[status].toLowerCase()}`}
              />
            ),
          )}
        </div>
      </header>
      <div className="results__overview" aria-label="Rows in this report">
        <div>
          <span>Graphs</span>
          <strong>{document.graphs.length}</strong>
        </div>
        <div>
          <span>Result rows</span>
          <strong>{totalRows}</strong>
        </div>
        <div data-state="passed">
          <span>Validation passed</span>
          <strong>{correctnessCounts.passed ?? 0}</strong>
        </div>
        <div data-state="failed">
          <span>Validation failed</span>
          <strong>{correctnessCounts.failed ?? 0}</strong>
        </div>
      </div>
      <Disclosure summary="Environment & report counts">
        <div className="results__metadata-grid">
          {HEADER_METADATA_FIELDS.map(({ key, label }) => (
            <div className="results__field" key={key}>
              <span className="results__field-label">{label}</span>
              <span className="results__field-value">
                {metadataFieldText(document.metadata, key)}
              </span>
            </div>
          ))}
        </div>
        <div className="results__counts">
          <div>
            <h3>Rows in this report</h3>
            <dl className="results__count-list">
              {(Object.keys(CORRECTNESS_LABEL) as CorrectnessState[]).map(
                (state) => (
                  <div key={state}>
                    <dt>{CORRECTNESS_LABEL[state]}</dt>
                    <dd>{correctnessCounts[state] ?? 0}</dd>
                  </div>
                ),
              )}
            </dl>
          </div>
          {combinationEntries.length > 0 && (
            <div>
              <h3>Reported suite counts</h3>
              <dl className="results__count-list">
                {combinationEntries.map(([key, value]) => (
                  <div key={key}>
                    <dt>{key.replace(/_/g, " ")}</dt>
                    <dd>{formatMagnitude(value)}</dd>
                  </div>
                ))}
              </dl>
              <p className="results__note">
                Reported counts exclude explicitly tagged reference rows and
                include successful unchecked engine rows as passes. Untagged
                PyTorch timing rows may be included.
              </p>
            </div>
          )}
        </div>
      </Disclosure>
      {document.warnings.length > 0 && (
        <ul className="results__warnings" aria-label="Parser warnings">
          {document.warnings.map((warning, index) => (
            <li key={index}>{warning}</li>
          ))}
        </ul>
      )}
    </section>
  );
}

function EngineCoverage({
  groups,
  graphLabels,
  onNavigate,
}: {
  groups: EngineGroup[];
  graphLabels: Map<string, string>;
  onNavigate: (graphKey: string, groupKey: string) => void;
}) {
  if (!groups.length) return null;
  return (
    <Disclosure
      summary={`Engine coverage across graphs · ${groups.length} configurations`}
    >
      <div className="results__table-scroll">
        <table className="results__table results__engine-table">
          <thead>
            <tr>
              <th scope="col">Engine configuration</th>
              <th scope="col">Role</th>
              <th scope="col">Graph coverage</th>
              <th scope="col">Validation</th>
              <th scope="col">
                <span className="results__sr-only">Open results</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {groups.map((group) => {
              const counts = countByCorrectness(group.rows);
              const graphKeys = [...group.graphKeys];
              return (
                <tr key={group.key}>
                  <td>
                    <strong>{group.provider}</strong>
                    <span className="results__cell-secondary">
                      {group.engineId ?? "ID unavailable"} ·{" "}
                      {group.engineVersion ?? "Version unavailable"}
                    </span>
                    {group.pluginPath && (
                      <span
                        className="results__cell-secondary"
                        title={group.pluginPath}
                      >
                        {group.pluginPath}
                      </span>
                    )}
                  </td>
                  <td>{group.role ?? "Unavailable"}</td>
                  <td>
                    <div className="results__graph-links">
                      {graphKeys.map((key) => (
                        <button
                          key={key}
                          type="button"
                          className="results__link-button"
                          onClick={() => onNavigate(key, group.key)}
                        >
                          {graphLabels.get(key) ?? key}
                        </button>
                      ))}
                    </div>
                  </td>
                  <td>
                    <div className="results__badge-list">
                      {(Object.keys(counts) as CorrectnessState[]).map(
                        (state) => (
                          <Badge
                            key={state}
                            kind="correctness"
                            state={state}
                            text={`${counts[state]} ${CORRECTNESS_LABEL[state].toLowerCase()}`}
                          />
                        ),
                      )}
                    </div>
                  </td>
                  <td>
                    <button
                      type="button"
                      className="results__link-button"
                      onClick={() => onNavigate(graphKeys[0], group.key)}
                      aria-label={`Show results for ${group.provider} ${group.engineId ?? ""}`}
                    >
                      Inspect
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Disclosure>
  );
}

// ---------------------------------------------------------------------------
// EngineComparison — graph/engine selects, the four-metric chart, and the
// adjacent numeric table (or the native single-execution tile).
// ---------------------------------------------------------------------------

function NativeTile({ row }: { row: ResultRow }) {
  const elapsed = finiteNonNegative(row.nativeElapsedMs);
  return (
    <div className="results__native-tile">
      <h3>Single execution wall time (ms)</h3>
      {row.status === "success" ? (
        <p className="results__native-value">
          {elapsed === null ? "Unavailable" : formatNumber(elapsed)}
        </p>
      ) : (
        <p className="results__native-value" data-error="true">
          Execution failed
        </p>
      )}
      <p className="results__note">
        Host wall time around execute and device synchronization; no warmup,
        repeated benchmark, or correctness comparison.
      </p>
    </div>
  );
}

function EngineComparison({
  document,
  graphLabels,
  activeGraphKey,
  onActiveGraphKeyChange,
  engineFilterKey,
  onEngineFilterKeyChange,
  metric,
  onMetricChange,
  selectedRowKey,
  onSelectRow,
}: {
  document: ResultDocument;
  graphLabels: Map<string, string>;
  activeGraphKey: string | null;
  onActiveGraphKeyChange: (key: string) => void;
  engineFilterKey: string;
  onEngineFilterKeyChange: (key: string) => void;
  metric: ComparisonMetric;
  onMetricChange: (m: ComparisonMetric) => void;
  selectedRowKey: string | null;
  onSelectRow: (rowKey: string) => void;
}) {
  const activeGraph =
    document.graphs.find((graph) => graph.key === activeGraphKey) ?? null;
  const graphRows = activeGraph?.rows ?? [];
  const isNative = document.kind === "native";
  const engineOptions = new Map<string, string>();
  for (const row of graphRows) {
    engineOptions.set(
      groupKeyFor(row),
      `${row.engineName} · ${row.pluginPath ?? row.engineId ?? "Timing record"}`,
    );
  }
  const filteredRows = engineFilterKey
    ? graphRows.filter((row) => groupKeyFor(row) === engineFilterKey)
    : graphRows;
  const activeMetric =
    METRICS.find((item) => item.key === metric) ?? METRICS[0];
  const chartable = filteredRows
    .map((row) => ({ row, value: comparisonValue(row, metric) }))
    .filter(
      (entry): entry is { row: ResultRow; value: number } =>
        entry.value !== null,
    );
  const maxValue = chartable.reduce(
    (max, entry) => Math.max(max, entry.value),
    0,
  );

  return (
    <section className="results__section results__comparison">
      <div className="results__section-heading">
        <div>
          <h2>{isNative ? "Current execution" : "Graph comparison"}</h2>
          <p>Select an engine to inspect its results.</p>
        </div>
        <span className="results__section-count">
          {graphRows.length} result{graphRows.length === 1 ? "" : "s"}
        </span>
      </div>
      <div className="results__controls">
        <label className="results__control results__control--graph">
          <span>Graph</span>
          <select
            value={activeGraphKey ?? ""}
            onChange={(event) => onActiveGraphKeyChange(event.target.value)}
            disabled={!document.graphs.length}
          >
            {document.graphs.map((graph) => (
              <option key={graph.key} value={graph.key}>
                {graphLabels.get(graph.key) ?? graph.name}
              </option>
            ))}
          </select>
        </label>
        {!isNative && (
          <>
            <label className="results__control">
              <span>Metric</span>
              <select
                value={metric}
                onChange={(event) =>
                  onMetricChange(event.target.value as ComparisonMetric)
                }
                disabled={!graphRows.length}
              >
                {METRICS.map((item) => (
                  <option key={item.key} value={item.key}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="results__control">
              <span>Engine filter</span>
              <select
                value={engineFilterKey}
                onChange={(event) =>
                  onEngineFilterKeyChange(event.target.value)
                }
                disabled={!engineOptions.size}
              >
                <option value="">All engines</option>
                {[...engineOptions].map(([key, label]) => (
                  <option key={key} value={key}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
          </>
        )}
      </div>
      {!graphRows.length ? (
        <p className="results__empty">
          {activeGraph
            ? "No engine results for this graph"
            : "No graphs in this report"}
        </p>
      ) : isNative ? (
        <>
          <NativeTile row={graphRows[0]} />
          <button
            type="button"
            className="results__link-button"
            aria-pressed={selectedRowKey === graphRows[0].key}
            onClick={() => onSelectRow(graphRows[0].key)}
          >
            Show execution details
          </button>
        </>
      ) : (
        <>
          <div className="results__chart-heading">
            <h3>{activeMetric.label}</h3>
            <p>{activeMetric.caption}</p>
          </div>
          <div
            className="results__chart"
            role="group"
            aria-label={`${activeMetric.label} chart`}
          >
            {chartable.length === 0 ? (
              <p className="results__empty">
                No comparable values for this metric
              </p>
            ) : (
              chartable.map(({ row, value }) => (
                <button
                  key={row.key}
                  type="button"
                  className="results__bar-row"
                  data-state={row.correctness}
                  aria-pressed={selectedRowKey === row.key}
                  aria-label={`${row.engineName} ${row.engineId ?? ""}: ${formatNumber(value)} ${activeMetric.label}, ${CORRECTNESS_LABEL[row.correctness]}`}
                  onClick={() => onSelectRow(row.key)}
                >
                  <span className="results__bar-label">
                    <strong>{row.engineName}</strong>
                    <small title={row.pluginPath ?? row.engineId ?? undefined}>
                      {row.role === "reference"
                        ? "Reference"
                        : (row.pluginPath?.split(/[\\/]/).pop() ??
                          row.engineId ??
                          "Timing record")}
                    </small>
                  </span>
                  <span className="results__bar-track">
                    <span
                      className="results__bar-fill"
                      style={{
                        width: `${maxValue > 0 ? (value / maxValue) * 100 : 0}%`,
                      }}
                    />
                  </span>
                  <span className="results__bar-value">
                    {formatNumber(value)}
                    {metric === "tflops" && row.partialFlops && (
                      <small>Partial coverage</small>
                    )}
                  </span>
                  <Badge
                    kind="correctness"
                    state={row.correctness}
                    text={
                      row.correctness === "failed"
                        ? "Validation failed"
                        : CORRECTNESS_LABEL[row.correctness]
                    }
                  />
                </button>
              ))
            )}
          </div>
          <div className="results__table-heading">
            <h3>All engine results</h3>
            <span>Execution and validation are separate</span>
          </div>
          <div className="results__table-scroll">
            <table className="results__table">
              <thead>
                <tr>
                  <th scope="col">Engine</th>
                  <th scope="col">Execution</th>
                  <th scope="col">Validation</th>
                  <th scope="col" className="results__numeric">
                    {activeMetric.label}
                  </th>
                  <th scope="col" className="results__numeric">
                    GPU mean (ms)
                  </th>
                  <th scope="col" className="results__numeric">
                    Host mean (ms)
                  </th>
                  <th scope="col">
                    <span className="results__sr-only">Inspect engine</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {filteredRows.map((row) => {
                  const gpuMean = comparisonValue(row, "gpu-mean-ms");
                  const metricValue = comparisonValue(row, metric);
                  return (
                    <tr
                      key={row.key}
                      data-selected={selectedRowKey === row.key}
                    >
                      <td>
                        <strong>{row.engineName}</strong>
                        <span className="results__cell-secondary">
                          {row.engineId ?? "ID unavailable"}
                        </span>
                        {row.role === "reference" && (
                          <span className="results__cell-secondary">
                            Reference row
                          </span>
                        )}
                        {row.status === "success" && gpuMean === null && (
                          <span className="results__cell-secondary">
                            GPU timing unavailable
                          </span>
                        )}
                      </td>
                      <td>
                        <Badge
                          kind="status"
                          state={row.status}
                          text={STATUS_LABEL[row.status]}
                        />
                      </td>
                      <td>
                        <Badge
                          kind="correctness"
                          state={row.correctness}
                          text={CORRECTNESS_LABEL[row.correctness]}
                        />
                      </td>
                      <td className="results__numeric results__metric-value">
                        {formatMagnitude(metricValue)}
                        {metric === "tflops" &&
                          row.partialFlops &&
                          metricValue !== null && (
                            <small>Partial coverage</small>
                          )}
                      </td>
                      <td className="results__numeric">
                        {formatMagnitude(gpuMean)}
                      </td>
                      <td className="results__numeric">
                        {formatMagnitude(row.hostStats?.mean_ms)}
                      </td>
                      <td>
                        <button
                          type="button"
                          className="results__link-button"
                          aria-pressed={selectedRowKey === row.key}
                          aria-label={`Inspect ${row.engineName} ${row.engineId ?? ""}`}
                          onClick={() => onSelectRow(row.key)}
                        >
                          Inspect <span aria-hidden="true">→</span>
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// EngineDetails — full drill-down for one selected row
// ---------------------------------------------------------------------------

function deriveOracleCorrectness(
  core: Record<string, unknown> | null,
): CorrectnessState {
  if (!core) return "not-checked";
  const executionSuccess = core.execution_success;
  const toleranceMatch = core.tolerance_match;
  if (executionSuccess === false) return "validation-error";
  if (executionSuccess === true) {
    if (toleranceMatch === true) return "passed";
    if (toleranceMatch === false) return "failed";
  }
  return "not-checked";
}

function OracleSection({ row }: { row: ResultRow }) {
  const oracle = asResultObject(row.details.oracle);
  const delta = asResultObject(row.details.oracle_delta);
  if (!oracle && !delta) return null;

  const oracleCorrectnessCore = oracle
    ? asResultObject(oracle.correctness)
    : null;
  const oracleCorrectness = deriveOracleCorrectness(oracleCorrectnessCore);
  const tuningAvailable = oracle?.tuning_available;

  const baseline = finiteNonNegative(delta?.baseline_mean_ms);
  const oracleMean = finiteNonNegative(delta?.oracle_mean_ms);
  const speedup = finiteNonNegative(delta?.speedup);
  const deltaMs = finiteNumber(delta?.delta_ms);
  const basis =
    delta?.basis === "host"
      ? "Host basis"
      : delta?.basis === "gpu_kernel"
        ? "GPU kernel basis"
        : null;

  const suppressReason =
    tuningAvailable === false
      ? "Tuning was not available for this plan."
      : row.correctness === "failed" ||
          row.correctness === "validation-error" ||
          oracleCorrectness === "failed" ||
          oracleCorrectness === "validation-error"
        ? "Correctness failed; speedup suppressed."
        : !oracle ||
            !basis ||
            deltaMs === null ||
            baseline === null ||
            baseline <= 0 ||
            oracleMean === null ||
            oracleMean <= 0 ||
            speedup === null
          ? "Baseline/oracle timing unavailable; speedup suppressed."
          : null;

  return (
    <div className="results__oracle">
      <h4>Oracle / tuning</h4>
      {oracle && (
        <>
          <p>
            Plan:{" "}
            {typeof oracle.plan_name === "string"
              ? oracle.plan_name
              : "Unavailable"}{" "}
            · Tuning available:{" "}
            {tuningAvailable === true
              ? "Yes"
              : tuningAvailable === false
                ? "No"
                : "Unavailable"}
          </p>
          <p>
            Compiled plans: {formatMagnitude(oracle.compiled_plans_benchmarked)}{" "}
            benchmarked / {formatMagnitude(oracle.compiled_plans_total)} total,{" "}
            {formatMagnitude(oracle.compiled_plans_failed)} failed. Sweep
            minimum: {formatMagnitude(oracle.sweep_min_time_ms)} ms.
          </p>
          <p>
            Tuned correctness:{" "}
            <Badge
              kind="correctness"
              state={oracleCorrectness}
              text={CORRECTNESS_LABEL[oracleCorrectness]}
            />
          </p>
          <p>
            Configured tolerances: rtol{" "}
            {formatMagnitude(oracleCorrectnessCore?.rtol)}, atol{" "}
            {formatMagnitude(oracleCorrectnessCore?.atol)}
          </p>
          <p>
            Max abs diff:{" "}
            {formatSignedOrInfinite(oracleCorrectnessCore?.max_abs_diff)} · Max
            rel diff:{" "}
            {formatSignedOrInfinite(oracleCorrectnessCore?.max_rel_diff)}
          </p>
          {typeof oracleCorrectnessCore?.error_message === "string" && (
            <p>{oracleCorrectnessCore.error_message}</p>
          )}
          <p>
            Tuned build time: {formatMagnitude(oracle.cpu_build_time_ms)} ms ·
            Plan index: {formatMagnitude(oracle.compiled_plan_index)} · Rank:{" "}
            {formatMagnitude(oracle.rank)}
          </p>
          <StatsTable
            title="Tuned GPU statistics"
            stats={(oracle.gpu_kernel_stats as TimingStats) ?? null}
          />
          <StatsTable
            title="Tuned host statistics"
            stats={(oracle.host_stats as TimingStats) ?? null}
          />
          <StatsTable
            title="Warm-baseline GPU statistics"
            stats={
              (oracle.warm_baseline_gpu_kernel_stats as TimingStats) ?? null
            }
          />
          <StatsTable
            title="Warm-baseline host statistics"
            stats={(oracle.warm_baseline_host_stats as TimingStats) ?? null}
          />
          <Disclosure summary="Tuned plan knobs and flags">
            <pre className="results__pre">
              {stringifyDetails({
                knob_settings: oracle.knob_settings,
                exhaustive_requested: oracle.exhaustive_requested,
                exhaustive_supported: oracle.exhaustive_supported,
                exhaustive_enabled: oracle.exhaustive_enabled,
              })}
            </pre>
          </Disclosure>
        </>
      )}
      {delta && (
        <div className="results__oracle-delta">
          <h5>Emitted delta{basis ? ` (${basis})` : ""}</h5>
          <p>
            Baseline:{" "}
            {baseline === null ? "Unavailable" : `${formatNumber(baseline)} ms`}{" "}
            · Oracle:{" "}
            {oracleMean === null
              ? "Unavailable"
              : `${formatNumber(oracleMean)} ms`}{" "}
            · Δ: {formatSignedFinite(delta.delta_ms)}
            {deltaMs !== null ? " ms" : ""}
          </p>
          {suppressReason ? (
            <p className="results__note" data-suppressed="true">
              {suppressReason}
            </p>
          ) : (
            <p className="results__speedup">
              {formatNumber(speedup as number)}x speedup
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function traceAvailability(
  trace: Record<string, unknown> | null,
): "none" | "available" | "unavailable" {
  if (!trace) return "none";
  const format = trace.format;
  const path = trace.path;
  const skipped = trace.skipped;
  const errorTail = trace.error_tail;
  const returncode = trace.returncode;
  const ok =
    format === "pftrace" &&
    typeof path === "string" &&
    path !== "" &&
    (skipped === undefined || skipped === null || skipped === "") &&
    (errorTail === undefined || errorTail === null || errorTail === "") &&
    (returncode === undefined || returncode === null || returncode === 0);
  return ok ? "available" : "unavailable";
}

function ProfilingSection({
  row,
  frameKey,
  engineLabel,
}: {
  row: ResultRow;
  frameKey: string;
  engineLabel: string;
}) {
  const extra = asResultObject(row.details.extra_metrics);
  const trace = extra ? asResultObject(extra.trace) : null;
  const availability =
    trace && row.status !== "success"
      ? "unavailable"
      : traceAvailability(trace);

  return (
    <div className="results__profiling">
      <h4>Profiling trace</h4>
      {availability === "none" && (
        <p className="results__note">Trace not recorded</p>
      )}
      {availability === "unavailable" && trace && (
        <p className="results__note" data-suppressed="true">
          Trace unavailable
          {typeof trace.skipped === "string" && trace.skipped
            ? ` — skipped: ${trace.skipped}`
            : ""}
          {typeof trace.error_tail === "string" && trace.error_tail
            ? ` — error: ${trace.error_tail}`
            : ""}
          {typeof trace.returncode === "number" && trace.returncode !== 0
            ? ` — exit code ${trace.returncode}`
            : ""}
        </p>
      )}
      {availability === "available" && trace && (
        <>
          <p className="results__note">
            Report-declared trace path (inert; not fetched):{" "}
            {String(trace.path)}
          </p>
          <PerfettoFrame
            key={frameKey}
            tracePath={trace.path as string}
            engineLabel={engineLabel}
          />
        </>
      )}
      {Array.isArray(trace?.warnings) && (
        <ul className="results__warnings">
          {trace.warnings.map(
            (warning, index) =>
              typeof warning === "string" && <li key={index}>{warning}</li>,
          )}
        </ul>
      )}
      {extra && (
        <>
          {extra.pmc !== undefined && (
            <Disclosure summary="PMC counters">
              <pre className="results__pre">{stringifyDetails(extra.pmc)}</pre>
            </Disclosure>
          )}
          {extra.perf !== undefined && (
            <Disclosure summary="Perf data">
              <pre className="results__pre">{stringifyDetails(extra.perf)}</pre>
            </Disclosure>
          )}
          {extra.roofline !== undefined && (
            <Disclosure summary="Roofline data">
              <pre className="results__pre">
                {stringifyDetails(extra.roofline)}
              </pre>
            </Disclosure>
          )}
        </>
      )}
    </div>
  );
}

function NativeProvenance({ row }: { row: ResultRow }) {
  const details = row.details;
  const graphSource = details.graphSource;
  const label =
    typeof details.graphLabel === "string" ? details.graphLabel : "Unavailable";
  return (
    <div className="results__native-provenance">
      <p className="results__note" data-suppressed="true">
        Built plan snapshot; later canvas edits are not included
      </p>
      {graphSource === "imported" && <p>Imported hipDNN JSON — {label}</p>}
      {typeof details.submittedGraphJson === "string" && (
        <Disclosure summary="Submitted graph JSON">
          <pre className="results__pre">{details.submittedGraphJson}</pre>
        </Disclosure>
      )}
      {typeof details.builtGraphJson === "string" && (
        <Disclosure summary="Built graph JSON">
          <pre className="results__pre">{details.builtGraphJson}</pre>
        </Disclosure>
      )}
    </div>
  );
}

function EngineDetails({
  document,
  graphLabels,
  graph,
  row,
  onClose,
}: {
  document: ResultDocument;
  graphLabels: Map<string, string>;
  graph: ResultGraph;
  row: ResultRow;
  onClose: () => void;
}) {
  const correctness = asResultObject(row.details.correctness);
  const nativeError = asResultObject(row.details.error);
  const isNative = document.kind === "native";
  const hasOracle =
    asResultObject(row.details.oracle) ||
    asResultObject(row.details.oracle_delta);
  const metrics: readonly (readonly [string, number | null | undefined])[] =
    isNative
      ? [["Single execution wall time (ms)", row.nativeElapsedMs]]
      : [
          ["GPU mean (ms)", comparisonValue(row, "gpu-mean-ms")],
          ["Host mean (ms)", row.hostStats?.mean_ms],
          ["Analytical TFLOP/s", comparisonValue(row, "tflops")],
          ["Analytical GB/s", comparisonValue(row, "gbytes")],
        ];

  return (
    <section className="results__section results__details">
      <div className="results__section-heading">
        <div>
          <span className="results__eyebrow">Engine details</span>
          <h2>{row.engineName}</h2>
          <p>{graphLabels.get(graph.key) ?? graph.name}</p>
        </div>
        <button type="button" onClick={onClose}>
          Back to comparison
        </button>
      </div>
      <div className="results__badge-list">
        <Badge
          kind="status"
          state={row.status}
          text={STATUS_LABEL[row.status]}
        />
        <Badge
          kind="correctness"
          state={row.correctness}
          text={CORRECTNESS_LABEL[row.correctness]}
        />
        {row.role && <span className="results__role">{row.role} row</span>}
      </div>
      <dl className="results__identity">
        {[
          ["Provider", row.provider],
          ["Engine ID", row.engineId],
          ["Version", row.engineVersion],
          ["Plugin", row.pluginPath],
        ].map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value ?? "Unavailable"}</dd>
          </div>
        ))}
        {isNative && (
          <div>
            <dt>Workspace (bytes)</dt>
            <dd>{formatMagnitude(row.details.workspaceBytes)}</dd>
          </div>
        )}
      </dl>
      {["error_message", "skip_reason", "oracle_error"].map(
        (key) =>
          typeof row.details[key] === "string" && (
            <p className="results__callout" key={key}>
              {row.details[key]}
            </p>
          ),
      )}
      {nativeError && (
        <p className="results__callout">
          {typeof nativeError.code === "string" ? nativeError.code : "Error"}:{" "}
          {typeof nativeError.message === "string"
            ? nativeError.message
            : "Unavailable"}
        </p>
      )}
      {Array.isArray(row.details.warnings) && (
        <ul className="results__warnings">
          {row.details.warnings.map(
            (warning, index) =>
              typeof warning === "string" && <li key={index}>{warning}</li>,
          )}
        </ul>
      )}
      <div className="results__detail-metrics">
        {metrics.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{formatMagnitude(value)}</strong>
            {label === "Analytical TFLOP/s" && row.partialFlops && (
              <small>Partial analytical coverage</small>
            )}
          </div>
        ))}
      </div>
      {isNative ? (
        <NativeProvenance row={row} />
      ) : (
        <>
          <div className="results__correctness-detail">
            <div className="results__section-heading">
              <h3>Correctness comparison</h3>
              <span>
                {correctness?.tolerance_match == null ||
                row.correctness === "not-run" ||
                row.correctness === "validation-error"
                  ? "Configured tolerances"
                  : "Compared against reference"}
              </span>
            </div>
            <dl className="results__values">
              <div>
                <dt>Relative tolerance (rtol)</dt>
                <dd>{formatMagnitude(correctness?.rtol)}</dd>
              </div>
              <div>
                <dt>Absolute tolerance (atol)</dt>
                <dd>{formatMagnitude(correctness?.atol)}</dd>
              </div>
              <div>
                <dt>Max absolute difference</dt>
                <dd>{formatSignedOrInfinite(correctness?.max_abs_diff)}</dd>
              </div>
              <div>
                <dt>Max relative difference</dt>
                <dd>{formatSignedOrInfinite(correctness?.max_rel_diff)}</dd>
              </div>
            </dl>
            {typeof correctness?.error_message === "string" && (
              <p className="results__note">{correctness.error_message}</p>
            )}
          </div>
          <Disclosure summary="Timing statistics & resource metrics">
            <div className="results__timing-grid">
              <StatsTable title="GPU timing" stats={row.gpuStats} />
              <StatsTable
                title="Host timing (producer-defined; may include synchronization)"
                stats={row.hostStats}
              />
            </div>
            <dl className="results__values">
              <div>
                <dt>Whole-run elapsed time (ms)</dt>
                <dd>{formatMagnitude(row.details.elapsed_time_ms)}</dd>
              </div>
              <div>
                <dt>Build time (ms)</dt>
                <dd>{formatMagnitude(row.details.cpu_build_time_ms)}</dd>
              </div>
              {[
                ["workspace_bytes", "Workspace (bytes)"],
                ["analytical_flops", "Analytical FLOPs"],
                ["analytical_io_bytes", "Analytical I/O (bytes)"],
                ["cpu_user_time_per_iter_us", "CPU user / iteration (µs)"],
                ["cpu_kernel_time_per_iter_us", "CPU kernel / iteration (µs)"],
                ["vram_used_mb", "Process-wide VRAM (MiB)"],
              ]
                .filter(([key]) => row.details[key] !== undefined)
                .map(([key, label]) => (
                  <div key={key}>
                    <dt>{label}</dt>
                    <dd>{formatMagnitude(row.details[key])}</dd>
                  </div>
                ))}
            </dl>
            {document.kind === "raw" && (
              <p className="results__note">
                Measured host samples:{" "}
                {Array.isArray(row.details.host_timings)
                  ? row.details.host_timings.length
                  : 0}{" "}
                · Measured GPU samples:{" "}
                {Array.isArray(row.details.kernel_timings)
                  ? row.details.kernel_timings.length
                  : 0}
              </p>
            )}
          </Disclosure>
          {hasOracle && (
            <Disclosure summary="Oracle tuning & warm-baseline comparison">
              <OracleSection row={row} />
            </Disclosure>
          )}
          {document.kind === "suite" ? (
            <Disclosure summary="Profiling trace & artifacts">
              <ProfilingSection
                row={row}
                frameKey={`${document.id}:${graph.key}:${row.key}`}
                engineLabel={row.engineName}
              />
            </Disclosure>
          ) : (
            <p className="results__note">Trace not recorded</p>
          )}
        </>
      )}
      <div className="results__source-details">
        <Disclosure summary="Original report">
          <pre className="results__pre">
            {document.sourceText ?? "Not available for this document"}
          </pre>
        </Disclosure>
        <Disclosure summary="Parsed row details">
          <pre className="results__pre">{stringifyDetails(row.details)}</pre>
        </Disclosure>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// ResultsViewer — top-level composition and selection state
// ---------------------------------------------------------------------------

export function ResultsViewer({ document }: { document: ResultDocument }) {
  const [activeGraphKey, setActiveGraphKey] = useState<string | null>(null);
  const [engineFilterKey, setEngineFilterKey] = useState<string>("");
  const [selectedRowKey, setSelectedRowKey] = useState<string | null>(null);
  const [metric, setMetric] = useState<ComparisonMetric>(
    "executions-per-second",
  );
  const comparisonRef = useRef<HTMLDivElement>(null);
  const detailsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (selectedRowKey) {
      detailsRef.current?.focus({ preventScroll: true });
      detailsRef.current?.scrollIntoView({ block: "start" });
    }
  }, [selectedRowKey]);

  useEffect(() => {
    const firstWithRows = document.graphs.find((g) => g.rows.length > 0);
    const initial = firstWithRows ?? document.graphs[0] ?? null;
    setActiveGraphKey(initial?.key ?? null);
    setEngineFilterKey("");
    setSelectedRowKey(null);
    setMetric("executions-per-second");
    // Only re-run when the document identity changes, per the viewer's
    // ownership of graph/engine/row selection.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [document.id]);

  const graphLabels = useMemo(
    () => computeGraphLabels(document.graphs),
    [document],
  );
  const groups = useMemo(() => buildEngineGroups(document), [document]);

  const activeGraph =
    document.graphs.find((g) => g.key === activeGraphKey) ?? null;
  const selectedRow =
    activeGraph?.rows.find((r) => r.key === selectedRowKey) ?? null;

  const handleNavigate = (graphKey: string, groupKey: string) => {
    setActiveGraphKey(graphKey);
    setEngineFilterKey(groupKey);
    setSelectedRowKey(null);
    comparisonRef.current?.scrollIntoView({ block: "start" });
  };

  return (
    <div className="results__body">
      <ResultsSummary document={document} />
      {document.graphs.length === 0 ? (
        <p className="results__empty">No graphs in this report</p>
      ) : (
        <div ref={comparisonRef}>
          <EngineComparison
            document={document}
            graphLabels={graphLabels}
            activeGraphKey={activeGraphKey}
            onActiveGraphKeyChange={(key) => {
              setActiveGraphKey(key);
              setEngineFilterKey("");
              setSelectedRowKey(null);
            }}
            engineFilterKey={engineFilterKey}
            onEngineFilterKeyChange={(key) => {
              setEngineFilterKey(key);
              setSelectedRowKey(null);
            }}
            metric={metric}
            onMetricChange={setMetric}
            selectedRowKey={selectedRowKey}
            onSelectRow={setSelectedRowKey}
          />
        </div>
      )}
      {activeGraph && selectedRow && (
        <div ref={detailsRef} tabIndex={-1} className="results__detail-target">
          <EngineDetails
            key={`${document.id}:${activeGraph.key}:${selectedRow.key}`}
            document={document}
            graphLabels={graphLabels}
            graph={activeGraph}
            row={selectedRow}
            onClose={() => {
              comparisonRef.current
                ?.querySelector<HTMLButtonElement>(
                  'button[aria-pressed="true"]',
                )
                ?.focus({ preventScroll: true });
              setSelectedRowKey(null);
              comparisonRef.current?.scrollIntoView({ block: "start" });
            }}
          />
        </div>
      )}
      <EngineCoverage
        groups={groups}
        graphLabels={graphLabels}
        onNavigate={handleNavigate}
      />
    </div>
  );
}
