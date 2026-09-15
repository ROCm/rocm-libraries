import type {
  ArtifactSource,
  FlowArtifact,
  RunCurrentStep,
  RunState,
  RunStatus,
  RunStep,
  StepKind,
} from "./types";

/**
 * Projects a run into the shape the timeline draws. Pure: no React, no IPC, no
 * clock — the same input always yields the same output, which is what makes it
 * the tested seam of the Authoring tab.
 *
 * It decides *layout*, never *meaning*. Every string it carries is either read
 * out of the run or assembled from numbers the run reported; nothing here
 * interprets a step, names a flow, or decides whether a run went well. The
 * order of nodes is the order the run's own step records appear in, so a flow
 * with fifteen steps, four loop groups and steps outside every group projects
 * exactly as well as a flow with two steps in one group — neither is a case in
 * the code below.
 */

/** How an output's value is drawn, chosen from the type the flow declared for it. */
export type OutputRender = "link" | "value" | "json" | "text";

export interface TimelineOutput {
  /** Output key, as the flow declares it. */
  readonly name: string;
  /** Declared type, verbatim; shown so an unfamiliar flow explains itself. */
  readonly type: string;
  readonly value: unknown;
  readonly render: OutputRender;
  /** The value rendered for display; for `json`, pretty-printed. */
  readonly text: string;
  /**
   * The discovered artifact this output resolved to. Non-null only when the
   * flow declared the output a path *and* the run reported a matching
   * artifact — a value that merely looks like a filename is still just text.
   */
  readonly artifactUri: string | null;
}

export interface TimelineStep {
  readonly kind: "step";
  /** Stable identity for a list key: a step id repeats across iterations. */
  readonly key: string;
  readonly id: string;
  readonly group: string | null;
  readonly iteration: number | null;
  readonly tool: string | null;
  readonly toolKind: StepKind | null;
  readonly status: string;
  readonly exitCode: number | null;
  readonly timedOut: boolean;
  readonly durationText: string;
  /** Step directory relative to the run directory, when the run recorded one. */
  readonly dir: string | null;
  readonly error: string | null;
  readonly current: boolean;
  readonly outputs: readonly TimelineOutput[];
}

/** One observed pass of a loop group. */
export interface TimelineIteration {
  /** Iteration number as the run recorded it. */
  readonly index: number;
  /** One-based position for display, so a run never shows "iteration 0". */
  readonly ordinal: number;
  readonly steps: readonly TimelineStep[];
  readonly current: boolean;
}

export interface TimelineGroup {
  readonly kind: "group";
  readonly key: string;
  readonly id: string;
  /** Iterations the run reports for this group. */
  readonly iterations: number;
  /** Budget actually applied to this group. */
  readonly budget: number;
  readonly satisfied: boolean;
  /** Exit condition as the flow wrote it. */
  readonly untilText: string;
  /** The same condition as written and as measured, rendered verbatim. */
  readonly untilMeasured: string;
  readonly budgetSource: string;
  readonly iterationRows: readonly TimelineIteration[];
  readonly current: boolean;
}

export type TimelineNode = TimelineStep | TimelineGroup;

/** Discovered artifacts, bucketed by where they were found. */
export interface TimelineArtifactGroup {
  readonly key: string;
  readonly source: ArtifactSource;
  readonly stepId: string | null;
  readonly iteration: number | null;
  readonly artifacts: readonly FlowArtifact[];
}

/**
 * How the run ended, or has not. `headline` is the run's own status followed by
 * each group's measured exit condition; `detail` carries the run's error and
 * process facts verbatim. No outcome word is invented here — every word in
 * `headline` comes from the run.
 */
export interface TimelineTerminal {
  readonly statusKind: RunState;
  readonly headline: string;
  readonly detail: readonly string[];
}

export interface Timeline {
  readonly runId: string;
  readonly runDir: string;
  readonly flow: string;
  readonly flowPath: string;
  readonly status: RunState;
  readonly running: boolean;
  readonly durationText: string;
  readonly startedAt: string | null;
  readonly currentStep: RunCurrentStep | null;
  /** Ungrouped steps and loop groups, in the run's own order. */
  readonly nodes: readonly TimelineNode[];
  /** Step records that have reached a terminal status, over all records seen. */
  readonly completedSteps: number;
  readonly totalSteps: number;
  readonly terminal: TimelineTerminal;
  /**
   * True when this run executed nothing but agents — no step of kind `tool`
   * ran, so nothing was compiled and nothing was executed. Measured per run: a
   * flow with a build or test step turns it off by itself.
   */
  readonly caution: boolean;
  readonly artifactGroups: readonly TimelineArtifactGroup[];
  readonly artifactCount: number;
  /** Bound input values, keyed by the flow's own input names. */
  readonly inputs: Readonly<Record<string, unknown>>;
}

/**
 * Declared output type to the control that draws it. A type this table does not
 * know is drawn as text, which is readable for anything.
 */
const OUTPUT_RENDER: Readonly<Record<string, OutputRender>> = {
  path: "link",
  json: "json",
  count: "value",
  int: "value",
  float: "value",
  bool: "value",
  string: "text",
};

/** Step statuses that mean the record is settled, for the completed count. */
const SETTLED: Readonly<Record<string, true>> = {
  ok: true,
  failed: true,
  skipped: true,
  timeout: true,
  error: true,
  cancelled: true,
};

/** Step statuses that mean the step never started, so it executed nothing. */
const NOT_EXECUTED: Readonly<Record<string, true>> = { skipped: true, pending: true };

/** Seconds as the run reports them, in units a multi-hour run stays readable in. */
function durationText(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const rest = `${String(Math.floor(total / 60) % 60).padStart(2, "0")}m ${String(total % 60).padStart(2, "0")}s`;
  return hours > 0 ? `${hours}h ${rest}` : rest;
}

/** Everything accumulated for one loop group while walking the run's steps. */
interface GroupBuild {
  readonly id: string;
  readonly rows: Map<number, { steps: TimelineStep[]; current: boolean }>;
  current: boolean;
}

function buildOutputs(step: RunStep, byOutput: ReadonlyMap<string, FlowArtifact>): TimelineOutput[] {
  return Object.keys(step.outputs).map((name) => {
    const type = step.outputTypes[name] ?? "";
    const render = OUTPUT_RENDER[type] ?? "text";
    const value = step.outputs[name];
    // Only a declared path may resolve to an artifact. A string whose value
    // happens to look like a file name stays a string.
    const artifact =
      render === "link"
        ? byOutput.get(`${step.id}\u0000${step.iteration ?? ""}\u0000${name}`)
        : undefined;
    let text = "";
    if (value !== null && value !== undefined) {
      if (render === "json") text = JSON.stringify(value, null, 2) ?? String(value);
      else text = typeof value === "string" ? value : JSON.stringify(value) ?? String(value);
    }
    return { name, type, value, render, text, artifactUri: artifact?.uri ?? null };
  });
}

export function buildTimeline(status: RunStatus): Timeline {
  // Path-typed outputs the run already discovered as files, keyed by the step,
  // iteration and output name that produced them.
  const byOutput = new Map<string, FlowArtifact>();
  for (const artifact of status.artifacts) {
    if (artifact.source !== "output" || artifact.outputName === null) continue;
    byOutput.set(
      `${artifact.stepId ?? ""}\u0000${artifact.iteration ?? ""}\u0000${artifact.outputName}`,
      artifact,
    );
  }

  const current = status.currentStep;
  const loopById = new Map(status.loops.map((loop) => [loop.id, loop]));
  // Ungrouped steps and group builders interleaved in the run's own order.
  const order: Array<TimelineStep | GroupBuild> = [];
  const groups = new Map<string, GroupBuild>();

  let completedSteps = 0;
  let executedTool = false;

  status.steps.forEach((step, index) => {
    if (SETTLED[step.status]) completedSteps += 1;
    if (step.kind === "tool" && !NOT_EXECUTED[step.status]) executedTool = true;

    const isCurrent =
      current !== null &&
      current.id === step.id &&
      current.group === step.group &&
      current.iteration === step.iteration;

    const node: TimelineStep = {
      kind: "step",
      key: `${step.group ?? ""}:${step.iteration ?? ""}:${step.id}:${index}`,
      id: step.id,
      group: step.group,
      iteration: step.iteration,
      tool: step.tool,
      toolKind: step.kind,
      status: step.status,
      exitCode: step.exitCode,
      timedOut: step.timedOut,
      durationText: durationText(step.durationS),
      dir: step.dir,
      error: step.error,
      current: isCurrent,
      outputs: buildOutputs(step, byOutput),
    };

    if (step.group === null) {
      order.push(node);
      return;
    }

    let build = groups.get(step.group);
    if (!build) {
      build = { id: step.group, rows: new Map(), current: current?.group === step.group };
      groups.set(step.group, build);
      order.push(build);
    }
    if (isCurrent) build.current = true;

    const iteration = step.iteration ?? 0;
    const row = build.rows.get(iteration);
    if (row) {
      row.steps.push(node);
      row.current ||= isCurrent;
    } else {
      build.rows.set(iteration, { steps: [node], current: isCurrent });
    }
  });

  // A group the run declared but has not reached yet still renders, so its
  // budget and exit condition are visible before its first step exists.
  for (const loop of status.loops) {
    if (groups.has(loop.id)) continue;
    const build: GroupBuild = { id: loop.id, rows: new Map(), current: current?.group === loop.id };
    groups.set(loop.id, build);
    order.push(build);
  }

  const nodes: TimelineNode[] = order.map((entry) => {
    if ("kind" in entry) return entry;
    const loop = loopById.get(entry.id);
    const iterationRows: TimelineIteration[] = [...entry.rows.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([index, row]) => ({
        index,
        ordinal: index + 1,
        steps: row.steps,
        current: row.current,
      }));
    return {
      kind: "group",
      key: `group:${entry.id}`,
      id: entry.id,
      iterations: loop?.iterations ?? iterationRows.length,
      budget: loop?.budget ?? 0,
      satisfied: loop?.satisfied ?? false,
      untilText: loop?.until ?? "",
      untilMeasured: loop?.untilMeasured ?? "",
      budgetSource: loop?.budgetSource ?? "",
      iterationRows,
      current: entry.current,
    };
  });

  const measured = status.loops.map(
    (loop) => `${loop.id}: ${loop.iterations} of ${loop.budget} iterations, ${loop.untilMeasured}`,
  );
  const detail: string[] = [];
  if (status.error) detail.push(status.error);
  if (status.exitCode !== null) detail.push(`worker exit code ${status.exitCode}`);
  if (status.engineStatus && status.engineStatus !== status.status) {
    detail.push(`run manifest: ${status.engineStatus}`);
  }
  if (status.processState && status.processState !== status.status) {
    detail.push(`process: ${status.processState}`);
  }

  // Grouped by where each artifact was found, in the order the run listed them.
  const artifactGroups: TimelineArtifactGroup[] = [];
  const bucket = new Map<string, FlowArtifact[]>();
  for (const artifact of status.artifacts) {
    const key = `${artifact.source}\u0000${artifact.stepId ?? ""}\u0000${artifact.iteration ?? ""}`;
    const existing = bucket.get(key);
    if (existing) {
      existing.push(artifact);
      continue;
    }
    const artifacts = [artifact];
    bucket.set(key, artifacts);
    artifactGroups.push({
      key,
      source: artifact.source,
      stepId: artifact.stepId,
      iteration: artifact.iteration,
      artifacts,
    });
  }

  return {
    runId: status.runId,
    runDir: status.runDir,
    flow: status.flow,
    flowPath: status.flowPath,
    status: status.status,
    running: status.status === "running",
    durationText: durationText(status.durationS),
    startedAt: status.startedAt,
    currentStep: current,
    nodes,
    completedSteps,
    totalSteps: status.steps.length,
    terminal: {
      statusKind: status.status,
      headline: measured.length > 0 ? `${status.status} — ${measured.join("; ")}` : status.status,
      detail,
    },
    caution: status.steps.length > 0 && !executedTool,
    artifactGroups,
    artifactCount: status.artifacts.length,
    inputs: status.inputs,
  };
}
