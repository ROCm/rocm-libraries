import { useCallback, useEffect, useRef, useState } from "react";
import {
  engine,
  type BuildHandle,
  type BuildPlanResult,
  type EngineInfo,
  type EngineOption,
  type LogEntry,
  type LogLevel,
} from "../engine";
import { serializeGraph } from "../graph/serialize";
import { platform } from "../platform";
import type { Graph } from "../graph/model";

/**
 * Engine control panel: build and execute the current graph through the hipDNN
 * runtime, surfacing structured errors and captured engine logs. The graph is
 * sent as Studio JSON; the native addon constructs it via the hipDNN frontend
 * builder. In the web build the engine reports unavailable and the actions stay
 * disabled; the JSON preview still works everywhere.
 */

type Phase = "idle" | "busy" | "ok" | "error";

const LOG_LEVEL_KEY = "hipdnn.logLevel";
const LOG_LEVELS: readonly LogLevel[] = ["off", "error", "warn", "info"];

interface EnginePanelProps {
  /** Produces the current graph on demand (avoids re-render churn on every edit). */
  getGraph(): Graph;
  /** Bumped by New/Open so the panel drops a stale build (disables Execute). */
  resetKey: number;
}

export function EnginePanel({ getGraph, resetKey }: EnginePanelProps) {
  const [info, setInfo] = useState<EngineInfo | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [status, setStatus] = useState<string>("");
  const [statusKind, setStatusKind] = useState<"info" | "error">("info");
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [logLevel, setLogLevel] = useState<LogLevel>("warn");
  const [handle, setHandle] = useState<BuildHandle | null>(null);
  const [selectedEngine, setSelectedEngine] = useState<EngineOption | null>(null);
  const [engines, setEngines] = useState<readonly EngineOption[]>([]);
  // "" = let hipDNN's heuristics pick; otherwise a decimal engine id.
  const [engineChoice, setEngineChoice] = useState("");
  const [enginesBusy, setEnginesBusy] = useState(false);
  const [serializedGraph, setSerializedGraph] = useState<string | null>(null);
  // Which JSON produced the live plan: the canvas graph, or an imported hipDNN one.
  const [planSource, setPlanSource] = useState<"studio" | "imported">("studio");
  const logRef = useRef<HTMLDivElement>(null);

  // The engine list is queried for the *current* graph, but getGraph changes on
  // every edit — a ref keeps the query callbacks stable so effects don't refire
  // on each keystroke.
  const getGraphRef = useRef(getGraph);
  getGraphRef.current = getGraph;

  useEffect(() => {
    let cancelled = false;
    void engine.info().then((result) => {
      if (!cancelled) setInfo(result);
    });
    // Restore + apply the persisted log level.
    void platform.store.get(LOG_LEVEL_KEY).then((saved) => {
      if (cancelled) return;
      const level = (LOG_LEVELS as readonly string[]).includes(saved ?? "")
        ? (saved as LogLevel)
        : "warn";
      setLogLevel(level);
      void engine.setLogLevel(level);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Keep the log view pinned to the newest line.
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [entries]);

  // New/Open replaced the graph: drop any built plan so Execute disables and the
  // engine/status readouts don't describe a graph that no longer exists.
  useEffect(() => {
    setHandle((prev) => {
      if (prev) void engine.release(prev);
      return null;
    });
    setSelectedEngine(null);
    setSerializedGraph(null);
    setPlanSource("studio");
    setEngines([]);
    setEngineChoice("");
    setStatus("");
    setPhase("idle");
  }, [resetKey]);

  const onLevelChange = useCallback((level: LogLevel) => {
    setLogLevel(level);
    void engine.setLogLevel(level);
    void platform.store.set(LOG_LEVEL_KEY, level);
  }, []);

  const appendCaptured = useCallback((captured?: readonly LogEntry[]) => {
    if (captured && captured.length > 0) setEntries((prev) => [...prev, ...captured]);
  }, []);

  const note = useCallback((severity: LogEntry["severity"], message: string) => {
    setEntries((prev) => [...prev, { severity, message }]);
  }, []);
  // Ask hipDNN which engines apply to the current graph. `loud` reports failures
  // (an empty or invalid graph has no engines) — the automatic refreshes stay quiet.
  const refreshEngines = useCallback(
    async (loud: boolean) => {
      setEnginesBusy(true);
      const result = await engine.listEngines(serializeGraph(getGraphRef.current()));
      setEnginesBusy(false);
      if (!result.ok) {
        setEngines([]);
        if (loud) {
          const code = result.error?.code ?? "UNKNOWN";
          note("ERROR", `Engine query failed [${code}]: ${result.error?.message ?? ""}`);
        }
        return;
      }
      setEngines(result.engines);
      setEngineChoice((prev) =>
        prev && !result.engines.some((e) => e.id === prev) ? "" : prev,
      );
      if (loud) note("INFO", `${result.engines.length} engine(s) available for this graph.`);
    },
    [note],
  );

  // Populate the picker once the runtime is up, and again whenever the graph is
  // replaced. Edits in between are covered by the refresh button and by builds.
  useEffect(() => {
    if (info?.available) void refreshEngines(false);
  }, [info?.available, resetKey, refreshEngines]);

  // Shared tail for both compile paths (Studio graph and imported hipDNN JSON).
  const applyBuildResult = useCallback(
    (result: BuildPlanResult, source: "studio" | "imported") => {
      appendCaptured(result.captured);
      if (result.engines && result.engines.length > 0) setEngines(result.engines);
      if (!result.ok || !result.handle) {
        setPhase("error");
        const code = result.error?.code ?? "UNKNOWN";
        setStatus([`Build failed [${code}]`, result.error?.message ?? "", ...result.log].join("\n"));
        setStatusKind("error");
        note("ERROR", `Build failed [${code}]: ${result.error?.message ?? ""}`);
        return;
      }
      setHandle(result.handle);
      setSelectedEngine(result.selectedEngine ?? null);
      setSerializedGraph(result.serializedGraph ?? null);
      setPlanSource(source);
      setPhase("ok");
      setStatus([`Build OK.`, `Workspace: ${result.workspaceSize ?? 0} bytes`, ...result.log].join("\n"));
      setStatusKind("info");
      note("INFO", `Build OK${result.selectedEngine ? ` on ${result.selectedEngine.name}` : ""}.`);
      // hipDNN ignores a pinned engine that has no solution for this graph.
      if (engineChoice && result.selectedEngine && result.selectedEngine.id !== engineChoice) {
        const requested = engines.find((e) => e.id === engineChoice)?.name ?? engineChoice;
        note(
          "WARN",
          `Engine ${requested} is not applicable to this graph; hipDNN used ${result.selectedEngine.name}.`,
        );
      }
    },
    [appendCaptured, note, engineChoice, engines],
  );

  const startBuild = useCallback(() => {
    setPhase("busy");
    setStatus("Building…");
    setStatusKind("info");
    setHandle((prev) => {
      if (prev) void engine.release(prev);
      return null;
    });
    setSelectedEngine(null);
  }, []);

  const doBuild = useCallback(async () => {
    const graphJson = serializeGraph(getGraph());
    startBuild();
    applyBuildResult(
      await engine.build(graphJson, engineChoice ? { engineId: engineChoice } : {}),
      "studio",
    );
  }, [getGraph, engineChoice, startBuild, applyBuildResult]);

  // Compile a graph handed over in hipDNN's canonical JSON. The canvas keeps
  // showing the Studio graph, so the plan is badged as imported until the next
  // Build.
  const doImportHipdnnJson = useCallback(async () => {
    const file = await platform.openTextFile(".json");
    if (!file) return;
    startBuild();
    note("INFO", `Loading hipDNN JSON from ${file.handle.name}.`);
    applyBuildResult(
      await engine.buildHipdnnJson(file.contents, engineChoice ? { engineId: engineChoice } : {}),
      "imported",
    );
  }, [engineChoice, startBuild, applyBuildResult, note]);

  const doExportHipdnnJson = useCallback(async () => {
    if (!serializedGraph) return;
    const base = getGraph().name.replace(/\s+/g, "_") || "graph";
    const written = await platform.saveTextFile(serializedGraph, {
      suggestedName: `${base}.hipdnn.json`,
    });
    if (written) note("INFO", `Wrote hipDNN JSON to ${written.name}.`);
  }, [serializedGraph, getGraph, note]);

  const doExecute = useCallback(async () => {
    if (!handle) return;
    setPhase("busy");
    setStatus("Executing…");
    setStatusKind("info");
    const result = await engine.execute(handle, { randomizeInputs: true });
    appendCaptured(result.captured);
    if (result.ok) {
      setPhase("ok");
      setStatus([`Execute OK.`, `Elapsed: ${result.elapsedMs ?? "?"} ms`, ...result.log].join("\n"));
      setStatusKind("info");
      note("INFO", `Execute OK (${result.elapsedMs?.toFixed(1) ?? "?"} ms).`);
    } else {
      setPhase("error");
      const code = result.error?.code ?? "UNKNOWN";
      setStatus([`Execute failed [${code}]`, result.error?.message ?? "", ...result.log].join("\n"));
      setStatusKind("error");
      note("ERROR", `Execute failed [${code}]: ${result.error?.message ?? ""}`);
    }
  }, [handle, appendCaptured, note]);

  const copyLog = useCallback(() => {
    const text = entries.map((e) => `[${e.severity}] ${e.message.trimEnd()}`).join("\n");
    void navigator.clipboard?.writeText(text);
  }, [entries]);

  const available = info?.available ?? false;

  return (
    <div className="engine">
      <div className="engine__controls">
        <div className="engine__header">
          <h2 className="panel__title">hipDNN Engine</h2>
          <div className="engine__status">
            <span className="engine__dot" data-state={phase} />
            <span>{info ? info.backend : "checking…"}</span>
          </div>
          {info?.device && <div className="engine__device">{info.device}</div>}
          {selectedEngine && (
            <div className="engine__selected" title={`Engine id ${selectedEngine.id}`}>
              <span className="engine__selected-label">Engine</span>
              <span className="engine__selected-name">{selectedEngine.name}</span>
            </div>
          )}
          {planSource === "imported" && (
            <div className="engine__plan-source">plan from imported hipDNN JSON</div>
          )}
        </div>

        <div className="engine__pick">
          <label className="engine__pick-label" htmlFor="engine-pick">
            Engine
          </label>
          <select
            id="engine-pick"
            value={engineChoice}
            onChange={(e) => setEngineChoice(e.target.value)}
            disabled={!available || phase === "busy"}
            title="Engine used for the next build"
          >
            <option value="">Best (heuristics)</option>
            {engines.map((e, i) => (
              <option key={e.id} value={e.id}>
                {i === 0 ? `${e.name} (top ranked)` : e.name}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="engine__pick-refresh"
            onClick={() => void refreshEngines(true)}
            disabled={!available || enginesBusy || phase === "busy"}
            title="Re-query the engines that apply to the current graph"
          >
            {enginesBusy ? "…" : "\u27f3"}
          </button>
        </div>

        <div className="engine__actions">
          <button
            type="button"
            className="engine__btn engine__btn--build"
            onClick={() => void doBuild()}
            disabled={!available || phase === "busy"}
          >
            Build
          </button>
          <button
            type="button"
            className="engine__btn engine__btn--execute"
            onClick={() => void doExecute()}
            disabled={!available || !handle || phase === "busy"}
          >
            Execute
          </button>
        </div>

        <div className="engine__io">
          <button
            type="button"
            onClick={() => void doExportHipdnnJson()}
            disabled={!serializedGraph}
            title="Save hipDNN's canonical JSON for the built plan's graph"
          >
            Export hipDNN JSON
          </button>
          <button
            type="button"
            onClick={() => void doImportHipdnnJson()}
            disabled={!available || phase === "busy"}
            title="Compile and run a graph given in hipDNN's canonical JSON"
          >
            Import hipDNN JSON…
          </button>
        </div>

        {status && (
          <pre className="engine__log" data-kind={statusKind}>
            {status}
          </pre>
        )}
      </div>

      <div className="engine__logpane">
        <div className="engine__loghead">
          <span>Log capture ({entries.length})</span>
          <div className="engine__logtools">
            <label className="engine__level">
              <span>Log level</span>
              <select value={logLevel} onChange={(e) => onLevelChange(e.target.value as LogLevel)}>
                <option value="off">Off</option>
                <option value="error">Error</option>
                <option value="warn">Warn</option>
                <option value="info">Info</option>
              </select>
            </label>
            <div className="engine__logbtns">
              <button type="button" onClick={copyLog} disabled={entries.length === 0}>
                Copy
              </button>
              <button type="button" onClick={() => setEntries([])} disabled={entries.length === 0}>
                Clear
              </button>
            </div>
          </div>
        </div>
        <div className="engine__capture" ref={logRef}>
          {entries.length === 0 ? (
            <div className="engine__capture-empty">No log entries captured yet.</div>
          ) : (
            entries.map((e, i) => (
              <div className="engine__line" data-sev={e.severity} key={i}>
                <span className="engine__sev">{e.severity}</span>
                <span className="engine__msg">{e.message.trimEnd()}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
