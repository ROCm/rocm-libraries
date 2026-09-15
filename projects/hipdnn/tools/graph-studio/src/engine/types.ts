/**
 * Engine bridge: the seam to a native hipDNN runtime.
 *
 * The renderer never links hipDNN directly — a graph is handed across as JSON
 * and the host (Electron main process, via an N-API addon) drives the real
 * `hipdnn_frontend::graph::Graph::from_json → build → execute` pipeline. The web
 * build supplies a stub that reports the engine as unavailable, so the UI
 * degrades cleanly in a plain browser and lights up under Electron.
 *
 * Mirrors the hipDNN frontend API surface the user asked for: build, execute,
 * and structured error reporting.
 */

export type EngineErrorCode =
  | "OK"
  | "INVALID_VALUE"
  | "HIPDNN_BACKEND_ERROR"
  | "ATTRIBUTE_NOT_SET"
  | "GRAPH_NOT_SUPPORTED"
  | "SHAPE_DEDUCTION_FAILED"
  | "INVALID_TENSOR_NAME"
  | "INVALID_VARIANT_PACK"
  | "GRAPH_EXECUTION_PLAN_CREATION_FAILED"
  | "GRAPH_EXECUTION_FAILED"
  | "HEURISTIC_QUERY_FAILED"
  | "UNSUPPORTED_GRAPH_FORMAT"
  | "HANDLE_ERROR"
  | "ENGINE_UNAVAILABLE";

/** Structured mirror of hipdnn_frontend::Error. */
export interface EngineError {
  readonly code: EngineErrorCode;
  readonly message: string;
}

/** Verbosity for the hipDNN global logger (maps to setGlobalLogLevel). */
export type LogLevel = "off" | "error" | "warn" | "info";

export type LogSeverity = "INFO" | "WARN" | "ERROR" | "FATAL";

/** A single log line captured from hipDNN during a build/execute call. */
export interface LogEntry {
  readonly severity: LogSeverity;
  readonly message: string;
}

export interface EngineInfo {
  readonly available: boolean;
  /** e.g. "hipDNN 1.x" or "unavailable (web build)". */
  readonly backend: string;
  /** GPU/device description when the host could query it. */
  readonly device?: string;
}

/**
 * A hipDNN engine applicable to a graph. `id` is the backend's 64-bit engine id
 * as a decimal string: it is a name hash and would lose precision as a number.
 */
export interface EngineOption {
  readonly id: string;
  readonly name: string;
}

export interface BuildOptions {
  /** Pin the plan to this engine; omitted lets hipDNN's heuristics choose. */
  readonly engineId?: string;
}

export interface ListEnginesResult {
  readonly ok: boolean;
  readonly error?: EngineError;
  /** Applicable engines, best first per hipDNN's heuristics. */
  readonly engines: readonly EngineOption[];
  readonly log: readonly string[];
  readonly captured?: readonly LogEntry[];
}

export interface BuildResult {
  readonly ok: boolean;
  readonly error?: EngineError;
  /** Bytes of scratch memory the compiled plan needs. */
  readonly workspaceSize?: number;
  /** hipDNN's canonical JSON for the built graph; feeds buildHipdnnJson(). */
  readonly serializedGraph?: string;
  /** Engine the compiled plan actually uses. */
  readonly selectedEngine?: EngineOption;
  /** Applicable engines, best first — free of charge from the build's heuristics query. */
  readonly engines?: readonly EngineOption[];
  readonly log: readonly string[];
  /** Log lines captured from hipDNN's logger during this call. */
  readonly captured?: readonly LogEntry[];
}

/** Result of translating a Studio graph into hipDNN's canonical JSON. */
export interface SerializeGraphResult {
  readonly ok: boolean;
  readonly error?: EngineError;
  /** hipDNN JSON accepted by deserialize()/buildHipdnnJson(). */
  readonly serializedGraph?: string;
  readonly log: readonly string[];
  readonly captured?: readonly LogEntry[];
}

/** A build result plus the token execute()/release() take when it succeeded. */
export interface BuildPlanResult extends BuildResult {
  readonly handle?: BuildHandle;
}

export interface ExecuteOptions {
  /**
   * Fill device tensors with random data instead of requiring host buffers.
   * The only mode the JSON-over-IPC path supports today: real host buffers
   * would need a shared-memory channel, which is a later addition.
   */
  readonly randomizeInputs: true;
}

export interface ExecuteResult {
  readonly ok: boolean;
  readonly error?: EngineError;
  readonly elapsedMs?: number;
  readonly log: readonly string[];
  readonly captured?: readonly LogEntry[];
}

/**
 * A build handle is opaque to the renderer: the host keeps the compiled Graph
 * alive and hands back a token the renderer passes to execute()/release().
 */
export type BuildHandle = string;

export interface EngineBridge {
  info(): Promise<EngineInfo>;

  /** Validate + compile a Studio JSON graph. */
  build(graphJson: string, options?: BuildOptions): Promise<BuildPlanResult>;

  /** Compile a graph given in hipDNN's own canonical JSON (see serializedGraph). */
  buildHipdnnJson(hipdnnJson: string, options?: BuildOptions): Promise<BuildPlanResult>;

  /**
   * hipDNN's canonical JSON for a Studio graph without compiling a plan — the
   * form other hipDNN tools deserialize.
   */
  serializeGraph(graphJson: string): Promise<SerializeGraphResult>;

  /** Heuristic-ranked engines applicable to a graph, without compiling a plan. */
  listEngines(graphJson: string): Promise<ListEnginesResult>;

  /** Execute a previously built graph. */
  execute(handle: BuildHandle, options: ExecuteOptions): Promise<ExecuteResult>;

  /** Drop a compiled graph the host is holding. */
  release(handle: BuildHandle): Promise<void>;

  /** Set hipDNN's global log verbosity for subsequent build/execute calls. */
  setLogLevel(level: LogLevel): Promise<void>;
}
