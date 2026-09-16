/**
 * Command bridge: runs an external command line against the current graph.
 *
 * The host writes the graph out as hipDNN JSON, substitutes that path for
 * `${current_graph}` in the command, spawns it, and streams the combined
 * terminal output back. Only the Electron build can do this; the web bridge
 * reports itself unavailable, mirroring platform/ and engine/.
 */

/** Placeholder replaced with the path of the written hipDNN JSON graph. */
export const GRAPH_PLACEHOLDER = "${current_graph}";

/**
 * Placeholder replaced with a host-owned path the command is expected to write
 * its `results.json` to. When it appears, the host clears any previous file
 * before the run and reads back whatever the command produced.
 */
export const RESULTS_PLACEHOLDER = "${results_json}";

/**
 * Placeholder replaced with a host-owned directory for the command's tensor
 * captures. Like the report, it is cleared before the run.
 */
export const TENSORS_PLACEHOLDER = "${tensor_dir}";

export interface CommandRequest {
  /** Correlates streamed output and cancellation with this run. */
  readonly id: string;
  /** Command line as typed by the user, before substitution. */
  readonly command: string;
  /** hipDNN canonical JSON the host writes to the temp file. */
  readonly graphJson: string;
  /** Base name for the temp file; the host sanitizes it. */
  readonly graphName: string;
  /** Keeps each tab's temp file separate. */
  readonly scope: string;
}

export interface CommandChunk {
  readonly id: string;
  readonly stream: "stdout" | "stderr";
  readonly text: string;
}

export interface CommandResult {
  readonly ok: boolean;
  /** Null when the process was killed by a signal. */
  readonly exitCode?: number | null;
  readonly signal?: string | null;
  /** Set when the command could not be started or the graph could not be written. */
  readonly error?: string;
  /** Command line after substitution, for echoing into the output pane. */
  readonly resolvedCommand?: string;
  readonly graphPath?: string;
  /** Set when the command line used `${tensor_dir}`. */
  readonly tensorsPath?: string;
  /** Set when the command line used `${results_json}`. */
  readonly resultsPath?: string;
  /** Contents of `resultsPath` after the run; absent when nothing was written. */
  readonly resultsJson?: string;
  /** Why `resultsPath` could not be read, when the command asked for one. */
  readonly resultsError?: string;
}

export interface CommandRunner {
  /** False in the browser, where no process can be spawned. */
  readonly available: boolean;

  /** Resolves when the process exits; output arrives via onOutput meanwhile. */
  execute(request: CommandRequest): Promise<CommandResult>;

  /** Terminate a running command (and its shell) by request id. */
  cancel(id: string): Promise<void>;

  /** Subscribe to output chunks from every run; returns an unsubscribe. */
  onOutput(listener: (chunk: CommandChunk) => void): () => void;
}
