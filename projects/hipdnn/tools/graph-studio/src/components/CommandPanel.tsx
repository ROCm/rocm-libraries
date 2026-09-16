import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { commandRunner } from "../command";
import { engine } from "../engine";
import { serializeGraph } from "../graph/serialize";
import type { Graph } from "../graph/model";

/**
 * Runs a fixed command against the current graph. The graph is written out as
 * hipDNN's canonical JSON — the form other hipDNN tools deserialize — and its
 * path replaces `${current_graph}` in the command line. Output streams into the
 * collapsible log at the bottom as the process produces it.
 *
 * A command that also names `${results_json}` is handed a path to write a
 * report to; whatever it wrote is passed to `onResults` once the run ends.
 */

/** A checkbox beside the run button that appends arguments when ticked. */
export interface CommandOption {
  readonly id: string;
  readonly label: string;
  /** Appended to the command line verbatim, space-separated. */
  readonly args: string;
}

const NO_OPTIONS: readonly CommandOption[] = [];

/** Output is rendered as one flowing stream; `note` lines are ours, not the child's. */
type Stream = "stdout" | "stderr" | "note";

interface Segment {
  stream: Stream;
  text: string;
}

interface CommandPanelProps {
  /** Distinct per tab: separates the output and the temp files. */
  scope: string;
  getGraph(): Graph;
  /** Command line to run, with `${current_graph}` and `${results_json}` unresolved. */
  command: string;
  /** Verb on the run button, replaced by "Stop" while the command is running. */
  runLabel: string;
  /** Opt-in arguments, offered as checkboxes beside the run button. */
  options?: readonly CommandOption[];
  /**
   * Called with the contents of the `${results_json}` file the run produced and
   * the path it was read from. `null` contents mean the command asked for a
   * report and did not leave a readable one, which clears whatever the previous
   * run published.
   */
  onResults?(json: string | null, reportPath: string): void;
  /**
   * Main content for the tab. With it the log docks below at a fixed height;
   * without it an open log fills the panel.
   */
  children?: React.ReactNode;
}

export function CommandPanel({
  scope,
  getGraph,
  command,
  runLabel,
  options = NO_OPTIONS,
  onResults,
  children,
}: CommandPanelProps) {
  const [segments, setSegments] = useState<Segment[]>([]);
  const [running, setRunning] = useState(false);
  // Keyed by option id; absent and false both mean "not appended".
  const [chosen, setChosen] = useState<Record<string, boolean>>({});
  // The log is a diagnostic, not the point of the tab: it stays shut until asked
  // for, including while a run is producing output.
  const [logOpen, setLogOpen] = useState(false);
  const runIdRef = useRef<string | null>(null);
  const outputRef = useRef<HTMLDivElement>(null);

  const append = useCallback((stream: Stream, text: string) => {
    setSegments((prev) => {
      const last = prev[prev.length - 1];
      // Chunks arrive split at arbitrary points; merging same-stream runs keeps
      // the DOM small and lets line breaks come from the text itself.
      if (last && last.stream === stream) {
        return [...prev.slice(0, -1), { stream, text: last.text + text }];
      }
      return [...prev, { stream, text }];
    });
  }, []);

  const note = useCallback((text: string) => {
    setSegments((prev) => {
      const last = prev[prev.length - 1];
      const lead = last && !last.text.endsWith("\n") ? "\n" : "";
      const line = `${lead}${text}\n`;
      if (last && last.stream === "note") {
        return [...prev.slice(0, -1), { stream: "note", text: last.text + line }];
      }
      return [...prev, { stream: "note", text: line }];
    });
  }, []);

  useEffect(
    () =>
      commandRunner.onOutput((chunk) => {
        if (chunk.id !== runIdRef.current) return;
        append(chunk.stream, chunk.text);
      }),
    [append],
  );

  // Also runs on expand, so opening the log lands at the newest output rather
  // than wherever the pane was when it closed.
  useEffect(() => {
    const pane = outputRef.current;
    if (pane) pane.scrollTop = pane.scrollHeight;
  }, [segments, logOpen]);

  const resolved = useMemo(
    () => [command, ...options.filter((o) => chosen[o.id]).map((o) => o.args)].join(" "),
    [chosen, command, options],
  );

  const doExecute = useCallback(async () => {
    const graph = getGraph();
    const id = `${scope}-${Date.now()}`;
    runIdRef.current = id;
    setRunning(true);
    setSegments([{ stream: "note", text: `$ ${resolved}\n` }]);

    const serialized = await engine.serializeGraph(serializeGraph(graph));
    if (!serialized.ok || !serialized.serializedGraph) {
      const code = serialized.error?.code ?? "UNKNOWN";
      note(`[graph serialization failed: ${code}] ${serialized.error?.message ?? ""}`);
      runIdRef.current = null;
      setRunning(false);
      return;
    }

    const result = await commandRunner.execute({
      id,
      command: resolved,
      graphJson: serialized.serializedGraph,
      graphName: graph.name,
      scope,
    });

    runIdRef.current = null;
    setRunning(false);
    if (result.error) note(`[error] ${result.error}`);
    else if (result.signal) note(`[terminated by ${result.signal}]`);
    else note(`[exit ${result.exitCode ?? 0}]`);
    if (result.graphPath) note(`graph: ${result.graphPath}`);
    if (result.tensorsPath) note(`tensors: ${result.tensorsPath}`);
    if (result.resultsPath) {
      if (result.resultsJson !== undefined) note(`results: ${result.resultsPath}`);
      else note(`[results] ${result.resultsError ?? "not produced"}`);
      onResults?.(result.resultsJson ?? null, result.resultsPath);
    }
  }, [getGraph, note, onResults, resolved, scope]);

  const doStop = useCallback(() => {
    const id = runIdRef.current;
    if (id) void commandRunner.cancel(id);
  }, []);

  return (
    <div className="command" data-docked={children != null}>
      <div className="command__bar">
        <button
          type="button"
          className="command__run"
          data-running={running}
          onClick={() => (running ? doStop() : void doExecute())}
          disabled={!commandRunner.available}
        >
          {running ? "Stop" : runLabel}
        </button>
        {options.map((option) => (
          <label className="command__option" key={option.id}>
            <input
              type="checkbox"
              checked={chosen[option.id] ?? false}
              disabled={running || !commandRunner.available}
              onChange={(event) =>
                setChosen((prev) => ({ ...prev, [option.id]: event.target.checked }))
              }
            />
            {option.label}
          </label>
        ))}
        {!commandRunner.available && (
          <span className="command__unavailable">
            Running commands is only available in the Electron desktop build.
          </span>
        )}
      </div>
      {children != null && <div className="command__main">{children}</div>}
      <div className="command__log" data-open={logOpen}>
        <button
          type="button"
          className="command__toggle"
          aria-expanded={logOpen}
          onClick={() => setLogOpen((open) => !open)}
        >
          <span className="command__chevron" aria-hidden="true">
            ▸
          </span>
          Output log
          {running && <span className="command__running">running…</span>}
        </button>
        {logOpen && (
          <div className="command__output" ref={outputRef}>
            {segments.length === 0 ? (
              <div className="command__output-empty">No output yet.</div>
            ) : (
              segments.map((segment, i) => (
                <span className="command__seg" data-stream={segment.stream} key={i}>
                  {segment.text}
                </span>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}
