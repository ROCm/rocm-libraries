import { useState } from "react";
import { parseReport } from "../benchmark/report";
import { SAMPLE_LABEL, sampleReport } from "../benchmark/sample";
import type { BenchmarkReport, EngineResult, GraphResults } from "../benchmark/types";
import { platform } from "../platform";
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
}

const SAMPLE: Loaded = { report: sampleReport, label: SAMPLE_LABEL, sample: true };

export function VerifyReport() {
  const [loaded, setLoaded] = useState<Loaded>(SAMPLE);
  const [error, setError] = useState<string | null>(null);
  // Non-null means the tensor inspector is showing; the array carries the
  // manifest paths the report recorded for the chosen row.
  const [hints, setHints] = useState<readonly TensorHint[] | null>(null);

  const openReport = async () => {
    const opened = await platform.openTextFile(".json,application/json");
    if (!opened) return;
    try {
      setLoaded({ report: parseReport(opened.contents), label: opened.handle.name, sample: false });
      setError(null);
      setHints(null);
    } catch (failure) {
      setError((failure as Error).message);
    }
  };

  if (hints) return <TensorView hints={hints} onBack={() => setHints(null)} />;

  return (
    <div className="verify">
      <div className="verify__bar">
        <button type="button" onClick={() => void openReport()}>
          Open report…
        </button>
        <button type="button" onClick={() => setHints([])}>
          Tensor artifacts…
        </button>
        {loaded.sample && <span className="verify__note">showing bundled sample data</span>}
        {error && (
          <span className="verify__note" data-tone="error">
            {error}
          </span>
        )}
      </div>
      <BenchmarkReportView
        report={loaded.report}
        sourceLabel={loaded.label}
        sample={loaded.sample}
        onOpenTensors={(graph, row) => setHints(tensorHints(graph, row))}
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
