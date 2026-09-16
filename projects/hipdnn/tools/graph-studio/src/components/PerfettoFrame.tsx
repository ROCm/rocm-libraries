import { useCallback, useEffect, useRef, useState } from "react";

const PERFETTO_ORIGIN = "https://ui.perfetto.dev";
const PERFETTO_SRC = "https://ui.perfetto.dev/#!/?mode=embedded";
const PING_INTERVAL_MS = 250;
const READY_TIMEOUT_MS = 15000;

type FrameStatus =
  | { kind: "handshaking" }
  | { kind: "sent" }
  | { kind: "error"; message: string };

interface Session {
  readonly file: File;
  readonly generation: number;
}

/**
 * Renders an optional embedded Perfetto trace viewer for a profiling result.
 * `tracePath` is inert text (the report's descriptor path); it is never fetched
 * or otherwise used to load bytes. The user supplies the actual trace file
 * through the picker/drop target below, and only that file's bytes are ever
 * sent to the iframe.
 */
export function PerfettoFrame(props: { tracePath: string; engineLabel: string }): JSX.Element {
  const { tracePath, engineLabel } = props;
  const [session, setSession] = useState<Session | null>(null);
  const [status, setStatus] = useState<FrameStatus | null>(null);
  const [pickError, setPickError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const generationRef = useRef(-1);

  useEffect(() => {
    generationRef.current = session?.generation ?? -1;
  }, [session]);

  const acceptFile = useCallback((candidate: File | null) => {
    if (!candidate) return; // cancellation: preserve current trace/session
    if (candidate.size === 0) {
      setPickError("The selected trace file is empty.");
      return;
    }
    setPickError(null);
    setSession((prev) => ({ file: candidate, generation: (prev?.generation ?? -1) + 1 }));
  }, []);

  const reload = useCallback(() => {
    setSession((prev) => (prev ? { file: prev.file, generation: prev.generation + 1 } : prev));
  }, []);

  // Runs one handshake+send attempt for the current session's generation.
  // Guarded throughout by `generationRef` so a stale read or PONG from a
  // superseded file/row can never send bytes to or update the current frame.
  useEffect(() => {
    if (!session) return;
    const { file: activeFile, generation: myGeneration } = session;
    const iframe = iframeRef.current;
    if (!iframe) return;

    setStatus({ kind: "handshaking" });

    let pinged = false;
    let bufferResult: ArrayBuffer | null = null;
    let settled = false;

    const clearTimers = () => {
      window.clearInterval(pingTimer);
      window.clearTimeout(readyTimer);
      window.removeEventListener("message", onMessage);
    };

    const trySend = () => {
      if (settled || !pinged || !bufferResult) return;
      const win = iframe.contentWindow;
      if (!win) return;
      settled = true;
      win.postMessage(
        { perfetto: { buffer: bufferResult, title: activeFile.name, fileName: activeFile.name } },
        PERFETTO_ORIGIN,
        [bufferResult],
      );
      setStatus({ kind: "sent" });
    };

    const onMessage = (event: MessageEvent) => {
      if (settled || generationRef.current !== myGeneration) return;
      if (event.origin !== PERFETTO_ORIGIN) return;
      if (event.source !== iframe.contentWindow) return;
      if (event.data !== "PONG") return;
      pinged = true;
      clearTimers();
      trySend();
    };

    const pingTimer = window.setInterval(() => {
      if (settled || generationRef.current !== myGeneration) return;
      iframe.contentWindow?.postMessage("PING", PERFETTO_ORIGIN);
    }, PING_INTERVAL_MS);

    const readyTimer = window.setTimeout(() => {
      if (settled || generationRef.current !== myGeneration) return;
      settled = true;
      clearTimers();
      setStatus({
        kind: "error",
        message:
          "Perfetto did not respond within 15 seconds. Embedded Perfetto is unavailable. Benchmark results remain available.",
      });
    }, READY_TIMEOUT_MS);

    window.addEventListener("message", onMessage);

    activeFile.arrayBuffer().then(
      (buf) => {
        if (settled || generationRef.current !== myGeneration) return;
        bufferResult = buf;
        trySend();
      },
      () => {
        if (settled || generationRef.current !== myGeneration) return;
        settled = true;
        clearTimers();
        setStatus({ kind: "error", message: "Could not read the selected trace file." });
      },
    );

    return () => {
      settled = true;
      clearTimers();
    };
  }, [session]);

  return (
    <div className="perfetto-frame">
      <p className="perfetto-frame__path">
        Report trace path: <code>{tracePath}</code>
      </p>
      <p>Choose the .pftrace file to load this profiling result.</p>
      <div
        className="perfetto-frame__drop"
        onDragOver={(event) => {
          event.preventDefault();
          event.stopPropagation();
        }}
        onDrop={(event) => {
          event.preventDefault();
          event.stopPropagation();
          acceptFile(event.dataTransfer.files.item(0));
        }}
      >
        <button type="button" onClick={() => inputRef.current?.click()}>
          Select .pftrace file
        </button>
        {" or drop it here"}
        <input
          ref={inputRef}
          type="file"
          accept=".pftrace"
          className="perfetto-frame__input"
          style={{ display: "none" }}
          onChange={(event) => {
            acceptFile(event.currentTarget.files?.item(0) ?? null);
            event.currentTarget.value = "";
          }}
        />
      </div>
      {pickError && (
        <p className="perfetto-frame__error" role="alert">
          {pickError}
        </p>
      )}
      {session && (
        <div className="perfetto-frame__viewer">
          <p className="perfetto-frame__selected">User-selected trace: {session.file.name}</p>
          <iframe
            key={session.generation}
            ref={iframeRef}
            src={PERFETTO_SRC}
            title={`Perfetto trace viewer — ${engineLabel}`}
            referrerPolicy="no-referrer"
            sandbox="allow-scripts allow-same-origin"
            width="100%"
            height={600}
          />
          <p className="perfetto-frame__status" role="status">
            {status?.kind === "sent" && "Trace sent to embedded Perfetto"}
            {status?.kind === "handshaking" && "Connecting to embedded Perfetto…"}
            {status?.kind === "error" && status.message}
          </p>
          {status?.kind === "sent" && (
            <p>If Perfetto asks whether to open the trace, confirm inside the frame.</p>
          )}
          <button type="button" onClick={reload}>
            Reload trace viewer
          </button>
        </div>
      )}
    </div>
  );
}
