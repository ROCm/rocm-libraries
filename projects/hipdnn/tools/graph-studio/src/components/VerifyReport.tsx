import { useState } from "react";
import { parseReport } from "../benchmark/report";
import type { BenchmarkReport, EngineResult, GraphResults } from "../benchmark/types";
import { platform } from "../platform";
import { BenchmarkReportView } from "./BenchmarkReportView";
import { TensorView, type TensorHint } from "./TensorView";

/**
 * Verify tab content: a benchmark report, and the tensor inspector reached from
 * a report row. Empty until a run produces a report or one is opened.
 */

/** A report produced in this session, as opposed to one opened from disk. */
export interface ShownReport {
  readonly report: BenchmarkReport;
  readonly label: string;
  /** Where the report file lives; artifact paths inside it are anchored to it. */
  readonly source?: string;
}

interface VerifyReportProps {
  /**
   * The freshest result this session produced — a Studio execution or a
   * benchmark run — shown in place of the opened report until dismissed.
   */
  readonly current?: ShownReport | null;
  /** Why the last run left no readable report. */
  readonly currentError?: string | null;
  onDismissCurrent?(): void;
}

export function VerifyReport({ current, currentError, onDismissCurrent }: VerifyReportProps) {
  const [loaded, setLoaded] = useState<ShownReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Non-null means the tensor inspector is showing; the array carries the
  // manifest paths the report recorded for the chosen row.
  const [hints, setHints] = useState<readonly TensorHint[] | null>(null);

  const openReport = async () => {
    const opened = await platform.openTextFile(".json,application/json");
    if (!opened) return;
    try {
      setLoaded({
        report: parseReport(opened.contents),
        label: opened.handle.name,
        source: typeof opened.handle.token === "string" ? opened.handle.token : undefined,
      });
      setError(null);
      onDismissCurrent?.();
      setHints(null);
    } catch (failure) {
      setError((failure as Error).message);
    }
  };

  const shown = current ?? loaded;
  const shownError = error ?? currentError ?? null;

  if (hints) {
    return <TensorView hints={hints} reportPath={shown?.source} onBack={() => setHints(null)} />;
  }

  return (
    <div className="verify">
      <div className="verify__bar">
        <button type="button" onClick={() => void openReport()}>
          Open report…
        </button>
        {current && (
          <button type="button" onClick={onDismissCurrent}>
            Close {current.label}
          </button>
        )}
        <button type="button" data-primary="true" onClick={() => setHints([])}>
          Inspect tensors…
        </button>
        {shownError && (
          <span className="verify__note" data-tone="error">
            {shownError}
          </span>
        )}
      </div>
      {shown ? (
        <BenchmarkReportView
          report={shown.report}
          sourceLabel={shown.label}
          onOpenTensors={(graph, row) => setHints(tensorHints(graph, row))}
        />
      ) : (
        <div className="verify__empty">
          <p>
            Click the <strong>Start Benchmarking</strong> button to generate benchmarking
            statistics.
          </p>
        </div>
      )}
    </div>
  );
}

/** The captures worth loading when the inspector opens from a report row. */
function tensorHints(graph: GraphResults, row: EngineResult): TensorHint[] {
  const reference = graph.results.find((r) => r.role === "reference" && r.tensor_manifest);
  const hints: TensorHint[] = [];
  if (row.tensor_manifest) {
    hints.push({ label: `${row.provider} output`, path: row.tensor_manifest, role: "output" });
  }
  if (reference?.tensor_manifest && reference !== row) {
    hints.push({
      label: `${reference.provider} reference`,
      path: reference.tensor_manifest,
      role: "reference",
    });
  }
  if (graph.input_tensor_manifest) {
    hints.push({ label: "graph inputs", path: graph.input_tensor_manifest, role: "input" });
  }
  return hints;
}
