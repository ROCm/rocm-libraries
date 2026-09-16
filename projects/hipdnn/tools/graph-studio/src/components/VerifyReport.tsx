import { parseReport } from "../benchmark/report";
import type { BenchmarkReport, EngineResult, GraphResults } from "../benchmark/types";
import { platform } from "../platform";
import type { DirectoryRef, FileHandleRef, PlatformBridge, ReadBase } from "../platform/types";
import { BenchmarkReportView } from "./BenchmarkReportView";
import type { TensorHint } from "./TensorView";

/**
 * Verify tab content: the benchmark report a run produced, or one opened from
 * disk. Empty until either happens. The captures a report row recorded are
 * inspected in the Tensors tab, not here.
 */

/** A report on screen, and the file it came from if it has one. */
export interface ShownReport {
  readonly report: BenchmarkReport;
  readonly label: string;
  /**
   * The file the report was read from — a base `readRelated` can resolve its
   * artifact paths against. Null for a report with no file behind it, such as
   * a native execution snapshot.
   */
  readonly handle: FileHandleRef | null;
}

interface VerifyReportProps {
  /** The report on screen, owned by the app so every tab shows the same run. */
  readonly shown?: ShownReport | null;
  /** Why the last run or open left no readable report. */
  readonly shownError?: string | null;
  /** Publishes a report, or an error in place of one, to the whole app. */
  onShow?(report: ShownReport | null, error: string | null): void;
  /** Hands a row's captures to the Tensors tab. */
  onOpenTensors?(hints: readonly TensorHint[]): void;
  /**
   * A folder granted this session, used once a report's own handle cannot
   * resolve relative paths (a web file handle names no parent directory). Held
   * by the app so the Tensors tab resolves against the same grant.
   */
  readonly granted?: DirectoryRef | null;
  onGranted?(dir: DirectoryRef): void;
}

export function VerifyReport({
  shown = null,
  shownError = null,
  onShow,
  onOpenTensors,
  granted = null,
  onGranted,
}: VerifyReportProps) {
  const openReport = async () => {
    const opened = await platform.openTextFile(".json,application/json");
    if (!opened) return;
    try {
      onShow?.(
        { report: parseReport(opened.contents), label: opened.handle.name, handle: opened.handle },
        null,
      );
    } catch (failure) {
      onShow?.(null, (failure as Error).message);
    }
  };

  const requestDirectory = async () => {
    const dir = await platform.openDirectory();
    if (dir) onGranted?.(dir);
  };

  /**
   * One action for a results folder: grant it, find the report inside, and use
   * the folder as the base — so traces and tensors need no second pick.
   */
  const openRunFolder = async () => {
    const dir = await platform.openDirectory();
    if (!dir) return;
    onGranted?.(dir);
    try {
      const found = await findReportIn(platform, dir);
      onShow?.({ report: parseReport(found.text), label: found.name, handle: null }, null);
    } catch (failure) {
      // The folder still stands as a base: a report picked by hand resolves its
      // artifacts from it, which is the older two-step flow.
      onShow?.(null, (failure as Error).message);
    }
  };

  // A granted folder outlives whichever report is on screen; it wins once it
  // exists since it is the more general base (also good for tensor manifests).
  const base: ReadBase | null = granted ?? shown?.handle ?? null;

  return (
    <div className="verify">
      <div className="verify__bar">
        {/* One pick for a whole past run. A host that cannot grant a folder
            falls back to the report file alone, which still renders — only its
            traces and captures stay out of reach. */}
        {platform.canGrantDirectory() ? (
          <button
            type="button"
            title="Pick the folder that holds results.json — its traces and tensor captures load with it."
            onClick={() => void openRunFolder()}
          >
            Open past run…
          </button>
        ) : (
          <button type="button" onClick={() => void openReport()}>
            Open report…
          </button>
        )}
        {shown && (
          <button type="button" onClick={() => onShow?.(null, null)}>
            Close {shown.label}
          </button>
        )}
        {shown && !platform.canReadRelated(base) && platform.canGrantDirectory() && (
          <button
            type="button"
            title="Pick the folder that holds results.json, so its traces and tensor captures resolve."
            onClick={() => void requestDirectory()}
          >
            Use results folder…
          </button>
        )}
        {granted && <span className="verify__note">results folder: {granted.name}</span>}
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
          onOpenTensors={(graph, row) => onOpenTensors?.(tensorHints(graph, row))}
          base={base}
          onGrantDirectory={requestDirectory}
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

/**
 * Finds the report in a run directory. `results.json` is what the harness
 * writes, so it wins; otherwise a lone JSON file is unambiguous enough to
 * open, and anything else asks rather than guessing.
 */
export async function findReportIn(
  bridge: Pick<PlatformBridge, "listFiles" | "readRelated">,
  dir: DirectoryRef,
): Promise<{ name: string; text: string }> {
  const files = await bridge.listFiles(dir);
  const jsons = files.filter((name) => name.endsWith(".json"));
  const name = jsons.find((candidate) => candidate === "results.json") ?? jsons[0];
  if (!name || (jsons.length > 1 && name !== "results.json")) {
    const detail = jsons.length > 1 ? `${jsons.length} JSON files` : "no JSON file";
    throw new Error(
      `${dir.name}/ has ${detail} — folder kept as the artifact source, now open the report itself`,
    );
  }
  const bytes = await bridge.readRelated(dir, name);
  if (!bytes) throw new Error(`Could not read ${name} in ${dir.name}/`);
  return { name, text: new TextDecoder().decode(bytes) };
}
