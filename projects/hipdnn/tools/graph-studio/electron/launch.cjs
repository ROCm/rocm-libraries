// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { spawn } = require("node:child_process");
const studioPaths = require("./paths.cjs");

try {
  const [mode, ...args] = process.argv.slice(2);
  if (mode !== "studio" && mode !== "benchmark") {
    throw new Error("Expected launcher mode studio or benchmark.");
  }
  const configFile = path.join(__dirname, "..", "runtime-config.json");
  const paths = studioPaths.resolve({ HIPDNN_BUILD_CONFIG: configFile });
  const env = studioPaths.runtimeEnvironment(paths, {
    ...process.env,
    HIPDNN_BUILD_CONFIG: configFile,
  });
  delete env.ELECTRON_RUN_AS_NODE;
  const executable = mode === "studio" ? paths.electronExecutable : paths.pythonExecutable;
  if (!fs.existsSync(executable)) throw new Error(`Installed executable is missing: ${executable}`);
  const childArgs = mode === "studio" ? [paths.appDir, ...args] : ["-m", "dnn_benchmarking", ...args];
  const child = spawn(executable, childArgs, { env, stdio: "inherit" });
  for (const signal of ["SIGINT", "SIGTERM"]) {
    process.on(signal, () => child.kill(signal));
  }
  child.on("error", (error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
  child.on("close", (code, signal) => {
    process.exitCode = code ?? (signal === "SIGINT" ? 130 : 1);
  });
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
