// The canvas graph on disk. The renderer never touches the filesystem, so the
// main process writes the graph out as hipDNN JSON under the OS temp directory
// and hands the resulting absolute path to whatever is going to read it.
//
// Two callers with two lifetimes:
//
//   command bridge  one deterministic path per scope. The child reads the file
//                   and exits, so the next run may overwrite it.
//
//   flow bridge     a per-launch `subdir`. An agent run records this path in
//                   its inputs and re-reads it for as long as the run lasts,
//                   which is hours, so a second launch must not land on it.
//                   Those directories are swept by age instead.

"use strict";

const { app } = require("electron");
const fs = require("node:fs/promises");
const path = require("node:path");

const ROOT_DIR_NAME = "hipdnn-graph-studio";
const GRAPH_SUFFIX = ".hipdnn.json";
//: Launch graphs live here, under the orchestrator's run root rather than in temp.
const LAUNCH_GRAPH_DIR = "_launch-graphs";
const RUN_SUFFIX = ".run";

/** One path segment, safe on every platform; never empty, never a separator. */
const sanitizeName = (name) =>
  String(name ?? "").replace(/[^\w.-]+/g, "_").replace(/^[._]+|[._]+$/g, "") || "graph";

// `subdir` is sanitized as its own segment and joined after the scope, so a
// caller can nest without the separator being flattened into the name.
function scopeDir(scope, subdir) {
  const dir = path.join(app.getPath("temp"), ROOT_DIR_NAME, sanitizeName(scope));
  return subdir === undefined || subdir === null ? dir : path.join(dir, sanitizeName(subdir));
}

async function writeGraphFile(scope, graphName, graphJson, subdir) {
  const dir = scopeDir(scope, subdir);
  await fs.mkdir(dir, { recursive: true });
  const file = path.join(dir, `${sanitizeName(graphName)}${GRAPH_SUFFIX}`);
  await fs.writeFile(file, graphJson ?? "", "utf8");
  return file;
}

// A flow run's graph does not belong in the temp directory, whatever its lifetime.
// The run records the path in its inputs and keeps reading it, agents are pointed at
// it, and the run directory is the evidence someone opens days later -- yet the graph
// was the one input living somewhere the OS is entitled to delete. It already cost a
// run: stage 1 had succeeded, and it could not be resumed because the graph it was
// launched with had been swept and nothing had copied it anywhere durable.
//
// So a launch writes beside the runs instead: same tree as the evidence, same
// lifetime, and a run stays reproducible for as long as its directory exists.
async function writeLaunchGraphFile(runRoot, graphName, graphJson, subdir) {
  const dir = path.join(runRoot, LAUNCH_GRAPH_DIR, sanitizeName(subdir));
  await fs.mkdir(dir, { recursive: true });
  const file = path.join(dir, `${sanitizeName(graphName)}${GRAPH_SUFFIX}`);
  await fs.writeFile(file, graphJson ?? "", "utf8");
  return file;
}

// The command bridge hands the child one directory to write its whole run into:
// report, tensor captures, and profiling artifacts together. Keeping it beside
// the graph means one scope directory holds a run's input and output, the next
// run replaces them together, and every path the report records stays inside a
// tree the reader is allowed to follow.
function runDirFor(graphPath) {
  const base = graphPath.endsWith(GRAPH_SUFFIX)
    ? graphPath.slice(0, -GRAPH_SUFFIX.length)
    : graphPath;
  return base + RUN_SUFFIX;
}

/**
 * Drop the per-launch directories of a scope that nothing can still be reading.
 * Best effort: a directory in use by another instance, or already gone, is
 * skipped rather than reported.
 */
async function sweepLaunchDirs(scope, maxAgeMs) {
  const root = scopeDir(scope);
  const cutoff = Date.now() - maxAgeMs;
  let entries;
  try {
    entries = await fs.readdir(root, { withFileTypes: true });
  } catch {
    return 0;
  }
  let removed = 0;
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const dir = path.join(root, entry.name);
    try {
      const stats = await fs.stat(dir);
      if (stats.mtimeMs >= cutoff) continue;
      await fs.rm(dir, { recursive: true, force: true });
      removed += 1;
    } catch {
      // Still in use, or removed underneath us. Either way, not ours to report.
    }
  }
  return removed;
}

module.exports = {
  sanitizeName,
  writeGraphFile,
  writeLaunchGraphFile,
  runDirFor,
  sweepLaunchDirs,
};
