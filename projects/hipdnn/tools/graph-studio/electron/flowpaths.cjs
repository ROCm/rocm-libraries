// Where the flow orchestrator lives, and which Python runs it.
//
// The orchestrator is a sibling tool in this repository with its own
// virtualenv. Everything the server needs is passed to it explicitly -- the
// tool registry, the flow directory and the run root -- because the server
// otherwise derives the run root from the working directory it happened to be
// started in, and the working directory of a desktop app is not a decision
// anyone made.
//
// There is deliberately no fallback to a `python` on PATH. A wrong interpreter
// fails deep inside an import with a confusing message; a missing virtualenv
// is reported here, in one sentence, with the path that was expected.

"use strict";

const fs = require("node:fs");
const path = require("node:path");

/** The server is invoked as a module, from the orchestrator directory. */
const SERVER_MODULE = "flowmcp.server";

// graph-studio/electron -> graph-studio -> tools -> tools/orchestrator
const IN_REPO_DIR = path.resolve(__dirname, "..", "..", "orchestrator");

const isWindows = process.platform === "win32";

// Match paths.cjs: forward slashes everywhere, so argv and diagnostics read the
// same on every platform.
const slashes = (value) => value.replace(/\\/g, "/");

const isDir = (target) => {
  try {
    return fs.statSync(target).isDirectory();
  } catch {
    return false;
  }
};

const isFile = (target) => {
  try {
    return fs.statSync(target).isFile();
  } catch {
    return false;
  }
};

function venvPython(orchestratorDir) {
  return isWindows
    ? path.join(orchestratorDir, ".venv", "Scripts", "python.exe")
    : path.join(orchestratorDir, ".venv", "bin", "python");
}

/**
 * Resolve the orchestrator, in this order:
 *
 *   1. HIPDNN_ORCHESTRATOR_DIR, when set -- an out-of-tree checkout or a
 *      deployed copy;
 *   2. the in-repo sibling directory.
 *
 * Returns the full set of paths plus `ok`. When `ok` is false, `reason` is
 * displayable prose naming the path that was expected, and the fields resolved
 * so far are still filled in.
 */
function resolve(env = process.env) {
  const configured = env.HIPDNN_ORCHESTRATOR_DIR;
  const orchestratorDir = slashes(
    configured && configured.trim() !== "" ? path.resolve(configured.trim()) : IN_REPO_DIR
  );

  const result = {
    ok: false,
    orchestratorDir,
    python: slashes(venvPython(orchestratorDir)),
    module: SERVER_MODULE,
    toolsPath: "",
    flowsDir: slashes(path.join(orchestratorDir, "configs", "flows")),
    runRoot: slashes(path.join(orchestratorDir, "runs")),
    reason: "",
  };

  if (!isDir(orchestratorDir)) {
    result.reason = configured
      ? `HIPDNN_ORCHESTRATOR_DIR points at ${orchestratorDir}, which is not a directory.`
      : `The flow orchestrator was not found at ${orchestratorDir}. Set HIPDNN_ORCHESTRATOR_DIR to point at it.`;
    return result;
  }

  const serverFile = path.join(orchestratorDir, ...SERVER_MODULE.split("."));
  if (!isFile(`${serverFile}.py`)) {
    result.reason = `${orchestratorDir} has no ${SERVER_MODULE} module (expected ${slashes(serverFile)}.py).`;
    return result;
  }

  if (!isFile(result.python)) {
    result.reason =
      `The orchestrator Python was not found at ${result.python}. ` +
      `Create the virtualenv there; Graph Studio never falls back to a python on PATH.`;
    return result;
  }

  if (!isDir(result.flowsDir)) {
    result.reason = `The orchestrator has no flow directory at ${result.flowsDir}.`;
    return result;
  }

  // A local registry overrides the checked-in one: it is where a developer
  // records the executables their own machine actually has.
  const configsDir = path.join(orchestratorDir, "configs");
  const localTools = slashes(path.join(configsDir, "tools.local.yaml"));
  const sharedTools = slashes(path.join(configsDir, "tools.yaml"));
  if (isFile(localTools)) {
    result.toolsPath = localTools;
  } else if (isFile(sharedTools)) {
    result.toolsPath = sharedTools;
  } else {
    result.reason = `The orchestrator has no tool registry at ${localTools} or ${sharedTools}.`;
    return result;
  }

  result.ok = true;
  return result;
}

/** Argv for the server child, given a resolved set of paths. */
function serverArgs(resolved) {
  return [
    "-m",
    resolved.module,
    "--tools",
    resolved.toolsPath,
    "--flows-dir",
    resolved.flowsDir,
    "--run-root",
    resolved.runRoot,
  ];
}

module.exports = { resolve, serverArgs, SERVER_MODULE };
