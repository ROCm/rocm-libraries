/**
 * Flow bridge: launches an agent-flow orchestrator run against the current
 * graph and monitors it to completion.
 *
 * The host writes the graph out as hipDNN JSON, hands that path to whichever
 * inputs the user pointed at the canvas, and talks to the orchestrator over
 * MCP. A run lasts minutes to hours, so `launch` resolves as soon as the run
 * has started and state arrives afterwards as events. Only the Electron build
 * can do this; the web bridge reports itself unavailable, mirroring command/,
 * platform/ and engine/.
 *
 * Every type here describes *structure*. None of them names a flow, a step, a
 * loop group, an output key or a file, because none of them may: the flow the
 * user picks is data, and the renderer must render one it has never seen. An
 * output is known to be a file because the flow declared its type as a path
 * (`RunStep.outputTypes`), and artifacts are discovered (`RunStatus.artifacts`)
 * rather than looked up by name.
 */

/** How a step's declaration classifies: a prompt/result contract, or argv and an exit code. */
export type StepKind = "agent" | "tool";

/** Reconciled run state: the run manifest and the orchestrator's process record together. */
export type RunState =
  | "running"
  | "ok"
  | "failed"
  | "cancelled"
  | "crashed"
  | "unknown";

/** How a discovered artifact was found. */
export type ArtifactSource = "output" | "step" | "run";

/** A call that could not produce a payload; `error` is displayable prose. */
export interface FlowFailure {
  readonly ok: false;
  readonly error: string;
}

/** A bridge call: the payload, or why there isn't one. */
export type FlowResponse<T> = ({ readonly ok: true } & T) | FlowFailure;

/** A bridge call with nothing to return but success or a reason. */
export type FlowAck = { readonly ok: true } | FlowFailure;

export interface FlowOutputSpec {
  /** Output key as the flow declares it. */
  readonly name: string;
  /** Declared type. A path-typed output is the flow's own statement that the value is a file. */
  readonly type: string;
}

export interface FlowInputSpec {
  /** Input key as the flow declares it. */
  readonly name: string;
  /** Prose from the flow, shown as help text. */
  readonly description: string | null;
  /** Declared type; chooses the form control. */
  readonly type: string;
  /** Whether a launch must supply a value. */
  readonly required: boolean;
  /** Declared default, pre-filled into the control. */
  readonly default: unknown;
  /** Whether a path-typed value is existence-checked before the run starts. */
  readonly exists: boolean;
}

export interface FlowStepSpec {
  readonly id: string;
  /** Owning loop group, or null for a step outside every loop. */
  readonly group: string | null;
  readonly tool: string;
  readonly kind: StepKind;
  /** Per-step deadline in seconds, when the flow declares one. */
  readonly timeout: number | null;
  readonly outputs: readonly FlowOutputSpec[];
}

export interface FlowLoopSpec {
  readonly id: string;
  /** Iteration budget the flow declares; the UI caps its own control at this. */
  readonly maxIterations: number;
  /** Exit condition as written. */
  readonly until: string;
  readonly onExhausted: string;
  readonly onStepFailure: string;
  readonly stepIds: readonly string[];
}

/** One flow as declared: everything needed to present a flow never seen before. */
export interface FlowSummary {
  readonly name: string;
  readonly description: string | null;
  readonly path: string;
  readonly stepCount: number;
  readonly tools: readonly string[];
  /** Every step in flow order, grouped or not. */
  readonly steps: readonly FlowStepSpec[];
  /** Every loop group in flow order; empty when the flow has none. */
  readonly loops: readonly FlowLoopSpec[];
  readonly inputs: readonly FlowInputSpec[];
  /** Non-null when the flow failed to load; the other fields are best-effort. */
  readonly loadError: string | null;
}

export interface FlowListResult {
  readonly flows: readonly FlowSummary[];
}

/** One flow's declared inputs, enough to build its form. */
export interface FlowInputsResult {
  readonly flow: string;
  readonly path: string;
  readonly inputs: readonly FlowInputSpec[];
}

export interface ValidateRequest {
  readonly flow: string;
  /** Optional; supplying values checks that each tool's executable resolves. */
  readonly inputs?: Readonly<Record<string, unknown>>;
  readonly profile?: string;
}

/** Preflight outcome: reference checks, tool coverage and executable resolution. */
export interface ValidateResult {
  readonly ok: boolean;
  readonly flow: string;
  readonly stepCount: number;
  readonly inputCount: number;
  readonly tools: readonly string[];
  /** Tools the flow invokes that the tool registry does not define. */
  readonly missingTools: readonly string[];
  /** Executable path per tool name, where one resolved. */
  readonly resolvedTools: Readonly<Record<string, string>>;
  readonly errors: readonly string[];
}

export interface LaunchRequest {
  readonly flow: string;
  /** Values for the inputs the flow declares, as typed into its form. */
  readonly inputs: Readonly<Record<string, unknown>>;
  /**
   * Input names the user pointed at the canvas. The host writes the graph once
   * and sets every one of these to that path. No input name is special.
   */
  readonly graphInputs: readonly string[];
  /** hipDNN canonical JSON the host writes to the temp file. */
  readonly graphJson: string;
  /** Base name for the temp file; the host sanitizes it. Not an input value. */
  readonly graphName: string;
  /** Requested loop budget. The server clamps it to what the flow declares. */
  readonly maxIterations?: number;
  readonly profile?: string;
  /** Free text echoed back in status; unused by the orchestrator. */
  readonly label?: string;
}

/** Launch acknowledgement. The run outlives the call. */
export interface LaunchResult {
  /** Run identity, shared by the run manifest and the run directory. */
  readonly runId: string;
  readonly runDir: string;
  /** URI of the authoritative run manifest. */
  readonly runJsonUri: string;
  readonly flow: string;
  readonly status: RunState;
  readonly pid: number | null;
  readonly startedAt: string;
  /** Budget actually applied; null when the flow declares no loop. */
  readonly maxIterations: number | null;
  /** Non-fatal adjustments, such as a requested budget being clamped. */
  readonly warnings: readonly string[];
  /** Where the run's copy of the canvas graph was written, for display. */
  readonly graphPath?: string;
}

/** Cancellation outcome. Cancelling an unknown or finished run is not an error. */
export interface CancelResult {
  readonly runId: string;
  /** Whether this call killed a live process tree. */
  readonly cancelled: boolean;
  readonly state: RunState;
  readonly killedPid: number | null;
  readonly message: string;
}

export interface RunLoop {
  readonly id: string;
  readonly iterations: number;
  /** Budget actually applied to this group. */
  readonly budget: number;
  readonly satisfied: boolean;
  /** Exit condition as written. */
  readonly until: string;
  /**
   * The condition as written and as measured, already formatted by the
   * orchestrator. Rendered verbatim: the renderer neither re-evaluates it nor
   * re-words it.
   */
  readonly untilMeasured: string;
  readonly budgetSource: string;
}

export interface RunStep {
  readonly id: string;
  /** Owning loop group, or null when the step is outside every loop. */
  readonly group: string | null;
  /** Zero-based iteration, or null when the step is outside every loop. */
  readonly iteration: number | null;
  readonly tool: string | null;
  readonly kind: StepKind | null;
  /** Status as the orchestrator recorded it, including a step skipped by its condition. */
  readonly status: string;
  readonly exitCode: number | null;
  readonly timedOut: boolean;
  readonly durationS: number;
  /** Step directory, relative to the run directory. */
  readonly dir: string | null;
  /** Extracted values, keyed by the flow's own output names. */
  readonly outputs: Readonly<Record<string, unknown>>;
  /**
   * Declared type per output name. This is how a consumer knows which values
   * are files and which are text, without knowing what any of them mean.
   */
  readonly outputTypes: Readonly<Record<string, string>>;
  readonly error: string | null;
}

/** The step the run is executing right now. */
export interface RunCurrentStep {
  readonly id: string;
  readonly group: string | null;
  readonly iteration: number | null;
}

/**
 * One discovered artifact. Found by walking the run — a path-typed output, a
 * file under a step's directory, or a file in the run root — never by asking
 * for a name.
 */
export interface FlowArtifact {
  readonly source: ArtifactSource;
  readonly stepId: string | null;
  readonly iteration: number | null;
  /** Output key, when the entry is a path-typed output. */
  readonly outputName: string | null;
  /** Display label: the output key, or the file name. */
  readonly label: string;
  /** Addressable URI, passed back to `artifact` and `revealArtifact` unmodified. */
  readonly uri: string;
  readonly mimeType: string;
  /** Structural hint set by the host, such as the run's own feedback channel. */
  readonly role: string | null;
}

/** The merged view of a run: step-level state from the manifest, process-level state from the host. */
export interface RunStatus {
  readonly runId: string;
  readonly runDir: string;
  readonly flow: string;
  readonly flowPath: string;
  /** The reconciled status, and the only one the UI should present as the outcome. */
  readonly status: RunState;
  /** Verbatim status from the run manifest. */
  readonly engineStatus: string | null;
  /** Verbatim process state; null when nothing is supervising the run. */
  readonly processState: string | null;
  readonly error: string | null;
  readonly startedAt: string | null;
  readonly durationS: number;
  /** Worker exit code once it has exited. */
  readonly exitCode: number | null;
  readonly provenance: Readonly<Record<string, unknown>>;
  /** Bound input values, keyed by the flow's own input names. */
  readonly inputs: Readonly<Record<string, unknown>>;
  /** Resolved flow vars, keyed by the flow's own var names. */
  readonly vars: Readonly<Record<string, unknown>>;
  /** One entry per loop group the run has: zero, one, or several. */
  readonly loops: readonly RunLoop[];
  /** Every step record, in order. */
  readonly steps: readonly RunStep[];
  readonly currentStep: RunCurrentStep | null;
  readonly artifacts: readonly FlowArtifact[];
  readonly runJsonUri: string;
  /** Supplementary prose. Shown as a log; never parsed for state. */
  readonly logTail: readonly string[];
}

/** Contents of one artifact, read through the host. */
export interface ArtifactContents {
  readonly uri: string;
  readonly mimeType: string;
  readonly text: string;
  /** True when the file exceeded the read cap and `text` is its head. */
  readonly truncated: boolean;
  /** Full size in bytes, whether or not the read was truncated. */
  readonly size: number;
}

/** The authoritative update: a new view of one run. */
export interface FlowStatusEvent {
  readonly kind: "status";
  readonly runId: string;
  readonly status: RunStatus;
}

/** Supplementary prose from the run. Never state. */
export interface FlowLogEvent {
  readonly kind: "log";
  readonly runId: string;
  readonly text: string;
}

/** A nudge for the header. The timeline is driven by status events, not by this. */
export interface FlowProgressEvent {
  readonly kind: "progress";
  readonly runId: string;
  /** Step records that have reached a terminal status. */
  readonly progress: number;
  /** Upper bound derived from the flow; absent when the flow cannot be read. */
  readonly total?: number;
  readonly message?: string;
}

/**
 * The host lost the orchestrator. Server-scoped: it carries no `runId`, so a
 * panel filtering events by run id must handle it unconditionally or it will
 * wait forever on a run that can no longer report.
 */
export interface FlowDisconnectedEvent {
  readonly kind: "disconnected";
  readonly reason: string;
}

/** The host reconnected and re-subscribed its live runs. Server-scoped. */
export interface FlowReconnectedEvent {
  readonly kind: "reconnected";
}

export type FlowEvent =
  | FlowStatusEvent
  | FlowLogEvent
  | FlowProgressEvent
  | FlowDisconnectedEvent
  | FlowReconnectedEvent;

export interface FlowRunner {
  /** False in the browser, where no orchestrator can be reached. */
  readonly available: boolean;

  /** Everything the host is willing to run. */
  list(): Promise<FlowResponse<FlowListResult>>;

  /** One flow's declared inputs, enough to build its form. */
  inputs(flow: string): Promise<FlowResponse<FlowInputsResult>>;

  /** Check a flow without running it. */
  validate(request: ValidateRequest): Promise<FlowResponse<ValidateResult>>;

  /** Resolves once the run has started, not once it has finished. */
  launch(request: LaunchRequest): Promise<FlowResponse<LaunchResult>>;

  /** Terminate a run and everything it spawned. */
  cancel(runId: string): Promise<FlowResponse<CancelResult>>;

  /** Poll a run. Correct on its own, without any event ever arriving. */
  status(runId: string, logTail?: number): Promise<FlowResponse<RunStatus>>;

  /** Read one artifact by the URI the run reported for it. */
  artifact(uri: string): Promise<FlowResponse<ArtifactContents>>;

  /** Show one artifact in the desktop file manager. */
  revealArtifact(uri: string): Promise<FlowAck>;

  /** Subscribe to events from every run; returns an unsubscribe. */
  onEvent(listener: (event: FlowEvent) => void): () => void;
}
