import { useCallback, useEffect, useRef, useState } from "react";
import { platform } from "../platform";
import type { PlatformBridge, ReadBase } from "../platform/types";

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
  readonly source: "auto" | "manual";
}

/** How a report-declared relative path should be obtained. */
export type RelatedRead =
  | { readonly kind: "autoload"; readonly bytes: Uint8Array }
  | { readonly kind: "grant" }
  | { readonly kind: "manual"; readonly reason: string };

/**
 * Decides how to get the bytes a report names by relative path: read them
 * automatically when the host can resolve `base` with no further prompting,
 * ask for a one-time folder grant when it cannot, or fall back to a manual
 * pick carrying the read's own failure reason when the host can resolve
 * paths in general but this particular one was missing or escaped the base.
 */
export async function resolveRelated(
  bridge: Pick<PlatformBridge, "canReadRelated" | "readRelated">,
  base: ReadBase | null | undefined,
  relativePath: string,
): Promise<RelatedRead> {
  if (!base || !bridge.canReadRelated(base)) return { kind: "grant" };
  try {
    const bytes = await bridge.readRelated(base, relativePath);
    if (bytes === null) return { kind: "grant" };
    return { kind: "autoload", bytes };
  } catch (error) {
    return { kind: "manual", reason: (error as Error).message };
  }
}

/**
 * Renders an optional embedded Perfetto trace viewer for a profiling result.
 * When the host can resolve `base` (the report's own directory, or a folder
 * the user granted), the trace loads itself with no prompt. Otherwise — or if
 * that automatic read fails — the picker/drop target below is the fallback,
 * and only the bytes it or the autoload actually obtained are ever sent to
 * the iframe.
 */
export function PerfettoFrame(props: {
  tracePath: string;
  engineLabel: string;
  base: ReadBase | null;
  onGrantDirectory: () => Promise<void>;
}): JSX.Element {
  const { tracePath, engineLabel, base, onGrantDirectory } = props;
  const [session, setSession] = useState<Session | null>(null);
  const [status, setStatus] = useState<FrameStatus | null>(null);
  const [pickError, setPickError] = useState<string | null>(null);
  const [offerGrant, setOfferGrant] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const generationRef = useRef(-1);

  useEffect(() => {
    generationRef.current = session?.generation ?? -1;
  }, [session]);

  // Tries the automatic path once per (tracePath, base) pair. `cancelled`
  // guards against a row switch (this component is remounted per row, but
  // the previous instance's in-flight read must not land on a fresh one) and
  // against a superseded base landing after a newer grant.
  useEffect(() => {
    let cancelled = false;
    resolveRelated(platform, base, tracePath).then((plan) => {
      if (cancelled) return;
      if (plan.kind === "autoload") {
        setOfferGrant(false);
        setPickError(null);
        const name = tracePath.split("/").pop() || tracePath;
        setSession((prev) => ({
          file: new File([new Uint8Array(plan.bytes)], name),
          generation: (prev?.generation ?? -1) + 1,
          source: "auto",
        }));
      } else if (plan.kind === "grant") {
        setOfferGrant(true);
      } else {
        setOfferGrant(false);
        setPickError(`Automatic read failed: ${plan.reason}`);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [tracePath, base]);

  const acceptFile = useCallback((candidate: File | null) => {
    if (!candidate) return; // cancellation: preserve current trace/session
    if (candidate.size === 0) {
      setPickError("The selected trace file is empty.");
      return;
    }
    setPickError(null);
    setSession((prev) => ({
      file: candidate,
      generation: (prev?.generation ?? -1) + 1,
      source: "manual",
    }));
  }, []);

  const reload = useCallback(() => {
    setSession((prev) => (prev ? { ...prev, generation: prev.generation + 1 } : prev));
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
      {offerGrant && (
        <button type="button" className="grant-button" onClick={() => void onGrantDirectory()}>
          Use run folder…
        </button>
      )}
      <p>
        {session?.source === "auto"
          ? "Loaded from the run folder. Pick a file only to override it."
          : "Choose the .pftrace file to load this profiling result."}
      </p>
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
          {session?.source === "auto" ? "Use a different file" : "Select .pftrace file"}
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
          <p className="perfetto-frame__selected">
            {session.source === "auto" ? "Automatically loaded trace" : "User-selected trace"}:{" "}
            {session.file.name}
          </p>
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
