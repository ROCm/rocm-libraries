// Regression suite for the Implement tab's projection (src/flow/timeline.ts).
// Bun's built-in test runner; no framework dependency.
//
// The suite is generic about a run's contents: every assertion about steps,
// outputs, groups and artifacts reads the names it needs out of the fixture it
// was handed. Node ordering is the deliberate exception. It is pinned to
// sequences read off each fixture by hand, because a helper that derives the
// expectation from the same rule the projection applies agrees with a wrong
// rule as readily as a right one. The two fixtures are deliberately
// dissimilar: one captured from a two-step, single-loop run, one synthetic
// with a step outside every loop, two loop groups of different sizes and
// budgets, a skipped step, and a non-agent step that exited non-zero.
import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { buildTimeline, type OutputRender, type TimelineStep } from "../src/flow/timeline";
import type { RunState, RunStatus, RunStep } from "../src/flow/types";

/** Fixtures are captured/synthesized `flow_status` payloads: RunStatus by construction. */
function loadFixture(name: string): RunStatus {
  return JSON.parse(readFileSync(join(import.meta.dir, "fixtures", name), "utf8")) as RunStatus;
}

const scaffolding = loadFixture("run-scaffolding.json");
const multigroup = loadFixture("run-multigroup.json");
const fixtures: readonly (readonly [string, RunStatus])[] = [
  ["scaffolding", scaffolding],
  ["multigroup", multigroup],
];

/** The declared type each renderer is expected to be chosen by. */
const RENDER_BY_TYPE: Readonly<Record<string, OutputRender>> = {
  path: "link",
  json: "json",
  count: "value",
  int: "value",
  float: "value",
  bool: "value",
  string: "text",
};

/** Words the UI is not allowed to produce; a flow's own output may still say them. */
const VERDICT_WORDS = ["pass", "passed", "verified", "correct", "clean", "validated"];

/** How the projection names a node, so a sequence can be compared to a literal. */
function nodeOrder(status: RunStatus): string[] {
  return buildTimeline(status).nodes.map((node) =>
    node.kind === "group" ? `group:${node.id}` : `step:${node.id}`,
  );
}

/** Every step node in the tree, ungrouped and grouped alike. */
function allSteps(status: RunStatus): TimelineStep[] {
  const found: TimelineStep[] = [];
  for (const node of buildTimeline(status).nodes) {
    if (node.kind === "step") found.push(node);
    else for (const row of node.iterationRows) found.push(...row.steps);
  }
  return found;
}

function withStatus(status: RunStatus, state: RunState): RunStatus {
  return { ...status, status: state };
}

function withSteps(status: RunStatus, steps: readonly RunStep[]): RunStatus {
  return { ...status, steps };
}

test("the same projection renders both run shapes, in the run's own order", () => {
  // Read off tests/fixtures/run-scaffolding.json by hand: all four step records
  // belong to the one loop group, so the run projects to a single group node.
  expect(nodeOrder(scaffolding)).toEqual(["group:review_cycle"]);
  // Read off tests/fixtures/run-multigroup.json by hand: an ungrouped step
  // first, then four records of the first loop group, then three of the second.
  expect(nodeOrder(multigroup)).toEqual([
    "step:seed_descriptor",
    "group:descriptor_pass",
    "group:integration_pass",
  ]);
  for (const [name, status] of fixtures) {
    expect(allSteps(status).length, name).toBe(status.steps.length);
  }
});

test("a group appears where its first step does, not at a fixed place", () => {
  const ungrouped = multigroup.steps.filter((step) => step.group === null);
  const grouped = multigroup.steps.filter((step) => step.group !== null);
  expect(ungrouped.length).toBe(1);
  // The same run with its one ungrouped step moved from the front to the back:
  // its node has to follow both groups instead of leading them.
  const moved = withSteps(multigroup, [...grouped, ...ungrouped]);
  expect(nodeOrder(moved)).toEqual([
    "group:descriptor_pass",
    "group:integration_pass",
    "step:seed_descriptor",
  ]);
});

test("each loop group carries its own budget, iteration count and rows", () => {
  for (const [name, status] of fixtures) {
    const groups = buildTimeline(status).nodes.filter((node) => node.kind === "group");
    expect(groups.length, name).toBe(status.loops.length);
    for (const group of groups) {
      const loop = status.loops.find((candidate) => candidate.id === group.id);
      expect(loop, `${name}:${group.id}`).toBeDefined();
      expect(group.budget).toBe(loop!.budget);
      expect(group.iterations).toBe(loop!.iterations);
      expect(group.satisfied).toBe(loop!.satisfied);
      const iterations = new Set(
        status.steps.filter((step) => step.group === group.id).map((step) => step.iteration),
      );
      expect(group.iterationRows.length, `${name}:${group.id}`).toBe(iterations.size);
      for (const row of group.iterationRows) {
        const records = status.steps.filter(
          (step) => step.group === group.id && step.iteration === row.index,
        );
        expect(row.steps.length).toBe(records.length);
        expect(row.steps.map((step) => step.id)).toEqual(records.map((step) => step.id));
        expect(row.ordinal).toBe(row.index + 1);
      }
    }
  }
});

test("groups differing in step count and budget both render from one fixture", () => {
  const groups = buildTimeline(multigroup).nodes.filter((node) => node.kind === "group");
  expect(groups.length).toBeGreaterThan(1);
  expect(new Set(groups.map((group) => group.budget)).size).toBeGreaterThan(1);
  const stepCounts = groups.map((group) => group.iterationRows[0]?.steps.length ?? 0);
  expect(new Set(stepCounts).size).toBeGreaterThan(1);
  // A pre-step outside every loop stays a top-level node.
  const ungrouped = multigroup.steps.filter((step) => step.group === null);
  expect(ungrouped.length).toBeGreaterThan(0);
  const topLevel = buildTimeline(multigroup).nodes.filter((node) => node.kind === "step");
  expect(topLevel.map((node) => node.id)).toEqual(ungrouped.map((step) => step.id));
});

test("every group renders its measured exit condition verbatim, and groups differ", () => {
  for (const [name, status] of fixtures) {
    for (const node of buildTimeline(status).nodes) {
      if (node.kind !== "group") continue;
      const loop = status.loops.find((candidate) => candidate.id === node.id)!;
      expect(node.untilMeasured, `${name}:${node.id}`).toBe(loop.untilMeasured);
      expect(node.untilText).toBe(loop.until);
    }
  }
  const measured = buildTimeline(multigroup)
    .nodes.filter((node) => node.kind === "group")
    .map((node) => node.untilMeasured);
  expect(new Set(measured).size).toBe(measured.length);
});

test("a flow with no loop at all renders every step as a top-level node", () => {
  const flat: RunStatus = {
    ...multigroup,
    loops: [],
    steps: multigroup.steps.map((step) => ({ ...step, group: null, iteration: null })),
  };
  const timeline = buildTimeline(flat);
  expect(timeline.nodes.length).toBe(flat.steps.length);
  expect(timeline.nodes.every((node) => node.kind === "step")).toBe(true);
  expect(timeline.terminal.headline).toBe(flat.status);
});

test("a declared loop with no step recorded yet still shows its budget and condition", () => {
  const started: RunStatus = { ...multigroup, steps: [], artifacts: [], currentStep: null };
  const timeline = buildTimeline(started);
  expect(timeline.nodes.length).toBe(started.loops.length);
  for (const node of timeline.nodes) {
    expect(node.kind).toBe("group");
    if (node.kind !== "group") continue;
    const loop = started.loops.find((candidate) => candidate.id === node.id)!;
    expect(node.budget).toBe(loop.budget);
    expect(node.untilMeasured).toBe(loop.untilMeasured);
    expect(node.iterationRows).toEqual([]);
  }
});

test("outputs are rendered by their declared type, never by their name", () => {
  for (const [name, status] of fixtures) {
    const steps = allSteps(status);
    for (const step of steps) {
      const record = status.steps.find(
        (candidate) =>
          candidate.id === step.id &&
          candidate.group === step.group &&
          candidate.iteration === step.iteration,
      )!;
      expect(step.outputs.map((output) => output.name), `${name}:${step.id}`).toEqual(
        Object.keys(record.outputs),
      );
      for (const output of step.outputs) {
        const declared = record.outputTypes[output.name];
        expect(output.type, `${name}:${step.id}:${output.name}`).toBe(declared);
        expect(output.render).toBe(RENDER_BY_TYPE[declared] ?? "text");
        expect(output.value).toEqual(record.outputs[output.name]);
      }
    }
    // Every path-typed output the run discovered as an artifact resolves to its URI.
    for (const artifact of status.artifacts) {
      if (artifact.source !== "output") continue;
      const output = steps
        .find(
          (step) => step.id === artifact.stepId && step.iteration === artifact.iteration,
        )!
        .outputs.find((candidate) => candidate.name === artifact.outputName)!;
      expect(output.render, `${name}:${artifact.label}`).toBe("link");
      expect(output.artifactUri).toBe(artifact.uri);
    }
  }
});

test("a string output that looks like a path is not promoted to an artifact", () => {
  const pathy = allSteps(multigroup)
    .flatMap((step) => step.outputs)
    .filter((output) => output.type === "string" && output.text.includes("/"));
  expect(pathy.length).toBeGreaterThan(0);
  for (const output of pathy) {
    expect(output.render).toBe("text");
    expect(output.artifactUri).toBeNull();
  }

  // The declared type is the only signal: re-declaring a genuine file output as
  // a string drops the link even though the run still lists its artifact.
  const link = status2Output(multigroup);
  const redeclared: RunStatus = {
    ...multigroup,
    steps: multigroup.steps.map((step) =>
      step === link.record
        ? { ...step, outputTypes: { ...step.outputTypes, [link.name]: "string" } }
        : step,
    ),
  };
  const output = allSteps(redeclared)
    .flatMap((step) => step.outputs)
    .find((candidate) => candidate.name === link.name)!;
  expect(output.render).toBe("text");
  expect(output.artifactUri).toBeNull();
  // The artifact itself is still listed; only the output stopped claiming it.
  expect(buildTimeline(redeclared).artifactCount).toBe(multigroup.artifacts.length);
});

/** The first path-typed output in a run that the run also discovered as an artifact. */
function status2Output(status: RunStatus): { record: RunStep; name: string } {
  for (const artifact of status.artifacts) {
    if (artifact.source !== "output" || artifact.outputName === null) continue;
    const record = status.steps.find(
      (step) => step.id === artifact.stepId && step.iteration === artifact.iteration,
    );
    if (record) return { record, name: artifact.outputName };
  }
  throw new Error("fixture has no path-typed output artifact");
}

test("the caution is measured per run and disappears when something executed", () => {
  // Every step in the captured run is an agent: nothing was compiled or run.
  expect(scaffolding.steps.every((step) => step.kind === "agent")).toBe(true);
  expect(buildTimeline(scaffolding).caution).toBe(true);

  // The synthetic run contains steps that executed a non-agent tool.
  expect(multigroup.steps.some((step) => step.kind === "tool" && step.status !== "skipped")).toBe(true);
  expect(buildTimeline(multigroup).caution).toBe(false);

  // A non-agent step that never ran executed nothing, so it is not evidence.
  const allSkipped: RunStatus = {
    ...multigroup,
    steps: multigroup.steps.map((step) =>
      step.kind === "tool" ? { ...step, status: "skipped", exitCode: null } : step,
    ),
  };
  expect(buildTimeline(allSkipped).caution).toBe(true);

  // A run with no step record yet has measured nothing to be cautious about.
  expect(buildTimeline({ ...multigroup, steps: [] }).caution).toBe(false);
});

test("the headline is the run's status plus the measured condition, and invents no verdict", () => {
  for (const [name, status] of fixtures) {
    for (const state of ["ok", "failed", "cancelled", "crashed", "unknown", "running"] as const) {
      const timeline = buildTimeline(withStatus(status, state));
      expect(timeline.terminal.statusKind, `${name}:${state}`).toBe(state);
      expect(timeline.terminal.headline.startsWith(state)).toBe(true);
      for (const loop of status.loops) {
        expect(timeline.terminal.headline).toContain(loop.untilMeasured);
        expect(timeline.terminal.headline).toContain(`${loop.iterations} of ${loop.budget}`);
      }
      // Subtract everything the run itself supplied — a loop group is free to
      // be named after any English word — and check what the projection added.
      let added = timeline.terminal.headline;
      for (const piece of [
        state,
        ...status.loops.flatMap((loop) => [
          loop.id,
          loop.untilMeasured,
          String(loop.iterations),
          String(loop.budget),
        ]),
      ]) {
        added = added.split(piece).join(" ");
      }
      const words = added.toLowerCase().split(/[^a-z]+/);
      for (const banned of VERDICT_WORDS) {
        expect(words, `${name}:${state}:${banned}`).not.toContain(banned);
      }
      // A reviewing step's own text output is never promoted into the headline.
      for (const step of status.steps) {
        for (const value of Object.values(step.outputs)) {
          if (typeof value !== "string" || value.length < 4) continue;
          expect(timeline.terminal.headline).not.toContain(value);
        }
      }
    }
  }
});

test("terminal states stay distinct and a failed run surfaces its error verbatim", () => {
  const seen = new Set<string>();
  for (const state of ["ok", "failed", "cancelled", "crashed", "unknown"] as const) {
    const timeline = buildTimeline(withStatus(scaffolding, state));
    expect(timeline.running).toBe(false);
    seen.add(timeline.terminal.headline);
  }
  expect(seen.size).toBe(5);
  expect(buildTimeline(withStatus(scaffolding, "running")).running).toBe(true);

  const failed = buildTimeline(withStatus(scaffolding, "failed"));
  expect(scaffolding.error).toBeTruthy();
  expect(failed.terminal.detail).toContain(scaffolding.error!);

  // A run whose manifest and process disagree reports both, unedited.
  const detached = buildTimeline({
    ...scaffolding,
    status: "unknown",
    engineStatus: "running",
    processState: "missing",
  });
  expect(detached.terminal.detail.some((line) => line.includes("running"))).toBe(true);
  expect(detached.terminal.detail.some((line) => line.includes("missing"))).toBe(true);
});

test("a non-zero exit on a non-agent step stays visible with its tool", () => {
  const record = multigroup.steps.find(
    (step) => step.kind === "tool" && step.exitCode !== null && step.exitCode !== 0,
  )!;
  const node = allSteps(multigroup).find(
    (step) =>
      step.id === record.id && step.group === record.group && step.iteration === record.iteration,
  )!;
  expect(node.exitCode).toBe(record.exitCode);
  expect(node.tool).toBe(record.tool);
  expect(node.toolKind).toBe("tool");
  expect(node.status).toBe(record.status);
  expect(node.error).toBe(record.error);
  // A multi-hour run is not shown as four digits of seconds.
  expect(buildTimeline(multigroup).durationText).toContain("h ");

  const skipped = multigroup.steps.find((step) => step.status === "skipped")!;
  const skippedNode = allSteps(multigroup).find((step) => step.id === skipped.id)!;
  expect(skippedNode.status).toBe("skipped");
  expect(skippedNode.exitCode).toBeNull();
});

test("a live run marks exactly one step current, wherever it sits", () => {
  for (const [name, status] of fixtures) {
    const partial = status.steps.slice(0, status.steps.length - 1);
    const last = partial[partial.length - 1];
    const live: RunStatus = {
      ...status,
      status: "running",
      steps: partial,
      currentStep: { id: last.id, group: last.group, iteration: last.iteration },
    };
    const timeline = buildTimeline(live);
    const current = allSteps(live).filter((step) => step.current);
    expect(current.length, name).toBe(1);
    expect(current[0].id).toBe(last.id);
    expect(current[0].group).toBe(last.group);
    expect(current[0].iteration).toBe(last.iteration);
    if (last.group !== null) {
      const group = timeline.nodes.find((node) => node.kind === "group" && node.id === last.group)!;
      expect(group.kind === "group" && group.current).toBe(true);
      expect(
        group.kind === "group" && group.iterationRows.some((row) => row.current),
      ).toBe(true);
    }
    expect(timeline.completedSteps).toBeLessThanOrEqual(timeline.totalSteps);
    expect(timeline.totalSteps).toBe(partial.length);
  }
});

test("artifacts pass through, grouped by where they were found", () => {
  for (const [name, status] of fixtures) {
    const timeline = buildTimeline(status);
    const flattened = timeline.artifactGroups.flatMap((group) => group.artifacts);
    expect(timeline.artifactCount, name).toBe(status.artifacts.length);
    expect(flattened.length).toBe(status.artifacts.length);
    expect(flattened.map((artifact) => artifact.uri)).toEqual(
      status.artifacts.map((artifact) => artifact.uri),
    );
    for (const group of timeline.artifactGroups) {
      for (const artifact of group.artifacts) {
        expect(artifact.source).toBe(group.source);
        expect(artifact.stepId).toBe(group.stepId);
        expect(artifact.iteration).toBe(group.iteration);
      }
    }
    // Every source the run reported survives; nothing is filtered by extension.
    expect(new Set(timeline.artifactGroups.map((group) => group.source))).toEqual(
      new Set(status.artifacts.map((artifact) => artifact.source)),
    );
  }
});
