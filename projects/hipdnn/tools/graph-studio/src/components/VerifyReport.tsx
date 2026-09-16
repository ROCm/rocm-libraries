import { useState } from "react";
import { parseReport } from "../benchmark/report";
import { SAMPLE_LABEL, sampleReport } from "../benchmark/sample";
import type { BenchmarkReport, EngineResult, GraphResults } from "../benchmark/types";
import { platform } from "../platform";
import type { DirectoryRef, FileHandleRef, ReadBase } from "../platform/types";
import { BenchmarkReportView } from "./BenchmarkReportView";
import { TensorView, type TensorHint } from "./TensorView";

/**
 * Verify tab content: a benchmark report, and the tensor inspector reached from
 * a report row. The bundled sample stays until a real `results.json` is opened.
 */

interface Loaded {
  readonly report: BenchmarkReport;
  readonly label: string;
  readonly sample: boolean;
  /** The file the report was opened from, if any — a real base for `readRelated`. */
  readonly handle: FileHandleRef | null;
}

const SAMPLE: Loaded = { report: sampleReport, label: SAMPLE_LABEL, sample: true, handle: null };

interface VerifyReportProps {
  /** A Studio execution, shown in place of the opened report until dismissed. */
  readonly native?: BenchmarkReport | null;
  onDismissNative?(): void;
}

export function VerifyReport({ native, onDismissNative }: VerifyReportProps) {
  const [loaded, setLoaded] = useState<Loaded>(SAMPLE);
  const [error, setError] = useState<string | null>(null);
  // Non-null means the tensor inspector is showing; the array carries the
  // manifest paths the report recorded for the chosen row.
  const [hints, setHints] = useState<readonly TensorHint[] | null>(null);
  // A folder the user granted this session, used once the report's own
  // handle cannot resolve relative paths (the web build's file handle names
  // no parent directory). Persists across reports so it is asked once.
  const [granted, setGranted] = useState<DirectoryRef | null>(null);

  const openReport = async () => {
    const opened = await platform.openTextFile(".json,application/json");
    if (!opened) return;
    try {
      setLoaded({
        report: parseReport(opened.contents),
        label: opened.handle.name,
        sample: false,
        handle: opened.handle,
      });
      setError(null);
      onDismissNative?.();
      setHints(null);
    } catch (failure) {
      setError((failure as Error).message);
    }
  };

  const requestDirectory = async () => {
    const dir = await platform.openDirectory();
    if (dir) setGranted(dir);
  };

  const shown: Loaded = native
    ? { report: native, label: "current execution", sample: false, handle: null }
    : loaded;
  // A granted folder outlives whichever report is on screen; it wins once it
  // exists since it is the more general base (also good for tensor manifests).
  const base: ReadBase | null = granted ?? shown.handle;

  if (hints) {
    return (
      <TensorView
        hints={hints}
        onBack={() => setHints(null)}
        base={base}
        onGrantDirectory={requestDirectory}
      />
    );
  }

  return (
    <div className="verify">
      <div className="verify__bar">
        <button type="button" onClick={() => void openReport()}>
          Open report…
        </button>
        {native ? (
          <button type="button" onClick={onDismissNative}>
            Close execution
          </button>
        ) : (
          <button type="button" data-primary="true" onClick={() => setHints([])}>
            Inspect tensors…
          </button>
        )}
        {!shown.sample && !platform.canReadRelated(base) && platform.canGrantDirectory() && (
          <button type="button" onClick={() => void requestDirectory()}>
            Use run folder…
          </button>
        )}
        {granted && <span className="verify__note">reading artifacts from {granted.name}</span>}
        {shown.sample && <span className="verify__note">showing bundled sample data</span>}
        {error && (
          <span className="verify__note" data-tone="error">
            {error}
          </span>
        )}
      </div>
      <BenchmarkReportView
        report={shown.report}
        sourceLabel={shown.label}
        sample={shown.sample}
        onOpenTensors={(graph, row) => setHints(tensorHints(graph, row))}
        base={base}
        onGrantDirectory={requestDirectory}
      />
    </div>
  );
}

/** Manifest paths worth showing when the inspector opens from a row. */
function tensorHints(graph: GraphResults, row: EngineResult): TensorHint[] {
  const reference = graph.results.find((r) => r.role === "reference" && r.tensor_manifest);
  const hints: TensorHint[] = [];
  if (row.tensor_manifest) hints.push({ label: `${row.provider} output`, path: row.tensor_manifest });
  if (reference?.tensor_manifest && reference !== row) {
    hints.push({ label: `${reference.provider} reference`, path: reference.tensor_manifest });
  }
  if (graph.input_tensor_manifest) {
    hints.push({ label: "graph inputs", path: graph.input_tensor_manifest });
  }
  return hints;
}
