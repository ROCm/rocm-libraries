import {
  VALIDATION_LABEL,
  formatBytes,
  formatMagnitude,
  traceAvailable,
  validationState,
} from "../benchmark/metrics";
import type {
  BenchmarkReport,
  Correctness,
  DurationStats,
  EngineResult,
  GraphResults,
  Oracle,
} from "../benchmark/types";
import type { ReadBase } from "../platform/types";
import { PerfettoFrame } from "./PerfettoFrame";

interface EngineDetailProps {
  report: BenchmarkReport;
  graph: GraphResults;
  row: EngineResult;
  onClose: () => void;
  base: ReadBase | null;
  onGrantDirectory: () => Promise<void>;
}

const STAT_FIELDS: { key: keyof DurationStats; label: string }[] = [
  { key: "mean_ms", label: "Mean" },
  { key: "median_ms", label: "Median" },
  { key: "std_ms", label: "Std dev" },
  { key: "min_ms", label: "Min" },
  { key: "max_ms", label: "Max" },
  { key: "p95_ms", label: "p95" },
  { key: "p99_ms", label: "p99" },
  { key: "total_ms", label: "Total" },
];

function StatsTable({ title, stats }: { title: string; stats: DurationStats | null }) {
  return (
    <div className="report__timing-card">
      <h4>{title}</h4>
      {stats ? (
        <table className="report__stats-table">
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
        <p className="report__unavailable">Unavailable</p>
      )}
    </div>
  );
}

function Disclosure({ summary, children }: { summary: string; children: React.ReactNode }) {
  return (
    <details className="report__disclosure">
      <summary>{summary}</summary>
      {children}
    </details>
  );
}

/**
 * A run can execute cleanly while its correctness comparison itself errors out
 * (bad shapes, a crashed reference) — distinct from a clean comparison that
 * simply mismatched tolerances.
 */
function correctnessFailed(c: Correctness): boolean {
  return !c.execution_success || c.tolerance_match === false;
}

type OracleCorrectnessState = "passed" | "failed" | "validation-error" | "not-checked";

function oracleCorrectnessState(c: Correctness | null): OracleCorrectnessState {
  if (!c) return "not-checked";
  if (!c.execution_success) return "validation-error";
  if (c.tolerance_match === true) return "passed";
  if (c.tolerance_match === false) return "failed";
  return "not-checked";
}

const ORACLE_CORRECTNESS_LABEL: Record<OracleCorrectnessState, string> = {
  passed: "Passed",
  failed: "Failed",
  "validation-error": "Validation error",
  "not-checked": "Not checked",
};

function CorrectnessDetail({ correctness }: { correctness: Correctness }) {
  return (
    <div className="report__correctness-detail">
      <div className="report__section-heading">
        <h3>Correctness comparison</h3>
        <span>
          {correctness.tolerance_match === null ? "Configured tolerances" : "Compared against reference"}
        </span>
      </div>
      <dl className="report__values">
        <div>
          <dt>Relative tolerance (rtol)</dt>
          <dd>{formatMagnitude(correctness.rtol)}</dd>
        </div>
        <div>
          <dt>Absolute tolerance (atol)</dt>
          <dd>{formatMagnitude(correctness.atol)}</dd>
        </div>
        <div>
          <dt>Max absolute difference</dt>
          <dd>{formatMagnitude(correctness.max_abs_diff)}</dd>
        </div>
        <div>
          <dt>Max relative difference</dt>
          <dd>{formatMagnitude(correctness.max_rel_diff)}</dd>
        </div>
      </dl>
      {correctness.error_message && <p className="report__note">{correctness.error_message}</p>}
    </div>
  );
}

/**
 * A speedup figure is only meaningful when tuning ran, the tuned plan's own
 * correctness held, and both sides of the comparison have a positive
 * duration — otherwise the number is misleading, so a reason is shown instead.
 */
function oracleSuppressReason(row: EngineResult, oracle: Oracle | null): string | null {
  const delta = row.oracle_delta;
  const oracleState = oracleCorrectnessState(oracle?.correctness ?? null);
  if (oracle?.tuning_available === false) return "Tuning was not available for this plan.";
  if (correctnessFailed(row.correctness) || oracleState === "failed" || oracleState === "validation-error") {
    return "Correctness failed; speedup suppressed.";
  }
  if (
    !oracle ||
    !delta ||
    delta.baseline_mean_ms === null ||
    delta.baseline_mean_ms <= 0 ||
    delta.oracle_mean_ms === null ||
    delta.oracle_mean_ms <= 0 ||
    delta.speedup === null
  ) {
    return "Baseline/oracle timing unavailable; speedup suppressed.";
  }
  return null;
}

function OracleSection({ row }: { row: EngineResult }) {
  const oracle = row.oracle ?? null;
  const delta = row.oracle_delta ?? null;
  if (!oracle && !delta && !row.oracle_error) return null;

  const oracleState = oracleCorrectnessState(oracle?.correctness ?? null);
  const suppressReason = oracleSuppressReason(row, oracle);

  return (
    <div className="report__oracle">
      <h4>Oracle / tuning</h4>
      {row.oracle_error && (
        <p className="report__note" data-suppressed="true">
          {row.oracle_error}
        </p>
      )}
      {oracle && (
        <>
          <p>
            Plan: {oracle.plan_name ?? "Unavailable"} · Tuning available:{" "}
            {oracle.tuning_available === true ? "Yes" : oracle.tuning_available === false ? "No" : "Unavailable"}
          </p>
          <p>
            Compiled plans: {formatMagnitude(oracle.compiled_plans_benchmarked)} benchmarked /{" "}
            {formatMagnitude(oracle.compiled_plans_total)} total, {formatMagnitude(oracle.compiled_plans_failed)}{" "}
            failed. Sweep minimum: {formatMagnitude(oracle.sweep_min_time_ms)} ms.
          </p>
          <p>
            Tuned correctness:{" "}
            <span className="badge" data-state={oracleState}>
              {ORACLE_CORRECTNESS_LABEL[oracleState]}
            </span>
          </p>
          <p>
            Configured tolerances: rtol {formatMagnitude(oracle.correctness?.rtol)}, atol{" "}
            {formatMagnitude(oracle.correctness?.atol)}
          </p>
          <p>
            Max abs diff: {formatMagnitude(oracle.correctness?.max_abs_diff)} · Max rel diff:{" "}
            {formatMagnitude(oracle.correctness?.max_rel_diff)}
          </p>
          {oracle.correctness?.error_message && <p>{oracle.correctness.error_message}</p>}
          <p>
            Tuned build time: {formatMagnitude(oracle.cpu_build_time_ms)} ms · Plan index:{" "}
            {formatMagnitude(oracle.compiled_plan_index)} · Rank: {formatMagnitude(oracle.rank)}
          </p>
          <StatsTable title="Tuned GPU statistics" stats={oracle.gpu_kernel_stats} />
          <StatsTable title="Tuned host statistics" stats={oracle.host_stats} />
          <StatsTable title="Warm-baseline GPU statistics" stats={oracle.warm_baseline_gpu_kernel_stats} />
          <StatsTable title="Warm-baseline host statistics" stats={oracle.warm_baseline_host_stats} />
          <Disclosure summary="Tuned plan knobs and flags">
            <pre className="report__pre">
              {JSON.stringify(
                {
                  knob_settings: oracle.knob_settings,
                  exhaustive_requested: oracle.exhaustive_requested,
                  exhaustive_supported: oracle.exhaustive_supported,
                  exhaustive_enabled: oracle.exhaustive_enabled,
                },
                null,
                2,
              )}
            </pre>
          </Disclosure>
        </>
      )}
      {delta && (
        <div className="report__oracle-delta">
          <h5>Emitted delta ({delta.basis === "host" ? "host basis" : "GPU kernel basis"})</h5>
          <p>
            Baseline: {formatMagnitude(delta.baseline_mean_ms)}
            {delta.baseline_mean_ms !== null ? " ms" : ""} · Oracle: {formatMagnitude(delta.oracle_mean_ms)}
            {delta.oracle_mean_ms !== null ? " ms" : ""} · Δ: {formatMagnitude(delta.delta_ms)}
            {delta.delta_ms !== null ? " ms" : ""}
          </p>
          {suppressReason ? (
            <p className="report__note" data-suppressed="true">
              {suppressReason}
            </p>
          ) : (
            <p className="report__speedup">{formatMagnitude(delta.speedup)}x speedup</p>
          )}
        </div>
      )}
    </div>
  );
}

function ProfilingSection({
  row,
  frameKey,
  engineLabel,
  base,
  onGrantDirectory,
}: {
  row: EngineResult;
  frameKey: string;
  engineLabel: string;
  base: ReadBase | null;
  onGrantDirectory: () => Promise<void>;
}) {
  const extra = row.extra_metrics ?? null;
  const trace = extra?.trace ?? null;
  const available = traceAvailable(row);

  return (
    <div className="report__profiling">
      <h4>Profiling trace</h4>
      {!trace && <p className="report__note">Trace not recorded</p>}
      {trace && !available && (
        <p className="report__note" data-suppressed="true">
          Trace unavailable
          {trace.skipped ? ` — skipped: ${trace.skipped}` : ""}
          {trace.error_tail ? ` — error: ${trace.error_tail}` : ""}
          {trace.returncode !== null && trace.returncode !== 0 ? ` — exit code ${trace.returncode}` : ""}
        </p>
      )}
      {trace && available && (
        <>
          <p className="report__note">Report-declared trace path (inert; not fetched): {trace.path}</p>
          <PerfettoFrame
            key={frameKey}
            tracePath={trace.path ?? ""}
            engineLabel={engineLabel}
            base={base}
            onGrantDirectory={onGrantDirectory}
          />
        </>
      )}
      {trace && trace.warnings.length > 0 && (
        <ul className="report__warnings">
          {trace.warnings.map((warning, index) => (
            <li key={index}>{warning}</li>
          ))}
        </ul>
      )}
      {extra?.pmc && (
        <Disclosure summary="PMC counters">
          <pre className="report__pre">{JSON.stringify(extra.pmc, null, 2)}</pre>
        </Disclosure>
      )}
      {extra?.perf && (
        <Disclosure summary="Perf data">
          <pre className="report__pre">{JSON.stringify(extra.perf, null, 2)}</pre>
        </Disclosure>
      )}
      {extra?.roofline && (
        <Disclosure summary="Roofline data">
          <pre className="report__pre">{JSON.stringify(extra.roofline, null, 2)}</pre>
        </Disclosure>
      )}
    </div>
  );
}

/**
 * Full drill-down for one engine row: identity, timing, correctness, tuning,
 * and (for suite reports) profiling. Reached from the results table's row
 * action; closed back to the summary/comparison/table view.
 */
export function EngineDetail({ report, graph, row, onClose, base, onGrantDirectory }: EngineDetailProps) {
  const state = validationState(row.correctness);
  const hasOracle = Boolean(row.oracle || row.oracle_delta || row.oracle_error);
  const metrics: readonly (readonly [string, number, boolean?])[] = [
    ["GPU mean (ms)", row.gpu_kernel_stats.mean_ms],
    ["Host mean (ms)", row.host_stats.mean_ms],
    ["Analytical TFLOP/s", row.derived_tflops_per_s, row.analytical_flops_partial],
    ["Analytical GB/s", row.derived_gbytes_per_s],
  ];

  return (
    <div className="report">
      <section className="report__details">
        <div className="report__section-heading">
          <div>
            <span className="report__eyebrow">Engine details</span>
            <h2>{row.engine_name}</h2>
            <p>{graph.graph_name}</p>
          </div>
          <button type="button" onClick={onClose}>
            Back to report
          </button>
        </div>
        <div className="report__badge-list">
          <span className="badge" data-status={row.status}>
            {row.status}
          </span>
          {row.role === "reference" ? (
            <span className="badge" data-state="reference">
              Reference
            </span>
          ) : (
            <span className="badge" data-state={state}>
              {VALIDATION_LABEL[state]}
            </span>
          )}
        </div>
        <dl className="report__identity">
          {(
            [
              ["Provider", row.provider],
              ["Engine", row.engine_name],
              ["Version", row.engine_version || null],
              ["Engine ID", row.engine_id],
              ["Plugin", row.plugin_path || null],
              ["Role", row.role ?? "engine"],
              ["Started at", row.started_at || null],
            ] as const
          ).map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value ?? "Unavailable"}</dd>
            </div>
          ))}
        </dl>
        {(row.error_message || row.skip_reason) && (
          <p className="report__callout">{row.error_message ?? row.skip_reason}</p>
        )}
        {row.warnings.length > 0 && (
          <ul className="report__warnings">
            {row.warnings.map((warning, index) => (
              <li key={index}>{warning}</li>
            ))}
          </ul>
        )}
        <div className="report__detail-metrics">
          {metrics.map(([label, value, partial]) => (
            <div key={label}>
              <span>{label}</span>
              <strong>
                {partial ? "≥ " : ""}
                {formatMagnitude(value)}
              </strong>
              {partial && <small>Partial analytical coverage</small>}
            </div>
          ))}
        </div>
        <CorrectnessDetail correctness={row.correctness} />
        <Disclosure summary="Timing statistics & resource metrics">
          <div className="report__timing-grid">
            <StatsTable title="GPU timing" stats={row.gpu_kernel_stats} />
            <StatsTable
              title="Host timing (producer-defined; may include synchronization)"
              stats={row.host_stats}
            />
          </div>
          <dl className="report__values">
            <div>
              <dt>Whole-run elapsed time (ms)</dt>
              <dd>{formatMagnitude(row.elapsed_time_ms)}</dd>
            </div>
            <div>
              <dt>Build time (ms)</dt>
              <dd>{formatMagnitude(row.cpu_build_time_ms)}</dd>
            </div>
            <div>
              <dt>Workspace</dt>
              <dd>{formatBytes(row.workspace_bytes)}</dd>
            </div>
            <div>
              <dt>Analytical FLOPs</dt>
              <dd>{formatMagnitude(row.analytical_flops)}</dd>
            </div>
            <div>
              <dt>Analytical I/O</dt>
              <dd>{formatBytes(row.analytical_io_bytes)}</dd>
            </div>
            <div>
              <dt>CPU user / iteration (µs)</dt>
              <dd>{formatMagnitude(row.cpu_user_time_per_iter_us)}</dd>
            </div>
            <div>
              <dt>CPU kernel / iteration (µs)</dt>
              <dd>{formatMagnitude(row.cpu_kernel_time_per_iter_us)}</dd>
            </div>
          </dl>
        </Disclosure>
        {hasOracle && (
          <Disclosure summary="Oracle tuning & warm-baseline comparison">
            <OracleSection row={row} />
          </Disclosure>
        )}
        {report.kind === "suite" ? (
          <Disclosure summary="Profiling trace & artifacts">
            <ProfilingSection
              row={row}
              frameKey={`${graph.graph_name}:${graph.graph_path}:${row.provider}:${row.engine_id ?? ""}`}
              engineLabel={row.engine_name}
              base={base}
              onGrantDirectory={onGrantDirectory}
            />
          </Disclosure>
        ) : (
          <p className="report__note">Trace not recorded</p>
        )}
      </section>
    </div>
  );
}
