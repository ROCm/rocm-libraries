import { useCallback, useEffect, useRef, useState } from "react";
import { commandRunner, GRAPH_PLACEHOLDER } from "../command";
import { engine } from "../engine";
import { serializeGraph } from "../graph/serialize";
import { platform } from "../platform";
import type { Graph } from "../graph/model";

/**
 * Runs an external command against the current graph. The graph is written out
 * as hipDNN's canonical JSON — the form other hipDNN tools deserialize — and
 * its path replaces `${current_graph}` in the command line. Output streams into
 * the pane below as the process produces it.
 */

const DEFAULT_COMMAND = `notepad.exe ${GRAPH_PLACEHOLDER}`;

/** Output is rendered as one flowing stream; `note` lines are ours, not the child's. */
type Stream = "stdout" | "stderr" | "note";

interface Segment {
  stream: Stream;
  text: string;
}

interface CommandPanelProps {
  /** Distinct per tab: separates the saved command, the output and the temp file. */
  scope: string;
  getGraph(): Graph;
}

export function CommandPanel({ scope, getGraph }: CommandPanelProps) {
  const [command, setCommand] = useState(DEFAULT_COMMAND);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [running, setRunning] = useState(false);
  const runIdRef = useRef<string | null>(null);
  const outputRef = useRef<HTMLDivElement>(null);
  // The saved command must not be clobbered by the default before it loads.
  const loadedRef = useRef(false);
  const storeKey = `hipdnn.command.${scope}`;

  useEffect(() => {
    let cancelled = false;
    void platform.store
      .get(storeKey)
      .catch(() => null)
      .then((saved) => {
        if (cancelled) return;
        if (saved) setCommand(saved);
        loadedRef.current = true;
      });
    return () => {
      cancelled = true;
    };
  }, [storeKey]);

  useEffect(() => {
    if (!loadedRef.current) return;
    const timer = setTimeout(() => void platform.store.set(storeKey, command), 400);
    return () => clearTimeout(timer);
  }, [command, storeKey]);

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

  const note = useCallback(
    (text: string) => {
      setSegments((prev) => {
        const last = prev[prev.length - 1];
        const lead = last && !last.text.endsWith("\n") ? "\n" : "";
        const line = `${lead}${text}\n`;
        if (last && last.stream === "note") {
          return [...prev.slice(0, -1), { stream: "note", text: last.text + line }];
        }
        return [...prev, { stream: "note", text: line }];
      });
    },
    [],
  );

  useEffect(
    () =>
      commandRunner.onOutput((chunk) => {
        if (chunk.id !== runIdRef.current) return;
        append(chunk.stream, chunk.text);
      }),
    [append],
  );

  useEffect(() => {
    const pane = outputRef.current;
    if (pane) pane.scrollTop = pane.scrollHeight;
  }, [segments]);

  const doExecute = useCallback(async () => {
    const graph = getGraph();
    const id = `${scope}-${Date.now()}`;
    runIdRef.current = id;
    setRunning(true);
    setSegments([{ stream: "note", text: `$ ${command}\n` }]);

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
      command,
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
  }, [command, getGraph, note, scope]);

  const doStop = useCallback(() => {
    const id = runIdRef.current;
    if (id) void commandRunner.cancel(id);
  }, []);

  return (
    <div className="command">
      <div className="command__bar">
        <label className="command__label" htmlFor={`command-${scope}`}>
          Command:
        </label>
        <input
          id={`command-${scope}`}
          className="command__input"
          value={command}
          spellCheck={false}
          disabled={running}
          onChange={(event) => setCommand(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !running) void doExecute();
          }}
          title={`${GRAPH_PLACEHOLDER} is replaced with the graph written as hipDNN JSON`}
        />
        <button
          type="button"
          className="command__run"
          data-running={running}
          onClick={() => (running ? doStop() : void doExecute())}
          disabled={!commandRunner.available || command.trim() === ""}
        >
          {running ? "Stop" : "Execute"}
        </button>
      </div>
      <div className="command__output" ref={outputRef}>
        {segments.length === 0 ? (
          <div className="command__output-empty">
            {commandRunner.available
              ? `No output yet. ${GRAPH_PLACEHOLDER} is replaced with the current graph, written as hipDNN JSON.`
              : "Running commands is only available in the Electron desktop build."}
          </div>
        ) : (
          segments.map((segment, i) => (
            <span className="command__seg" data-stream={segment.stream} key={i}>
              {segment.text}
            </span>
          ))
        )}
      </div>
    </div>
  );
}
