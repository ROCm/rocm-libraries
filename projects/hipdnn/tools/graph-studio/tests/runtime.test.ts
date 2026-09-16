import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, renameSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, delimiter } from "node:path";
import { createRequire } from "node:module";
import { spawnSync } from "node:child_process";

const require = createRequire(import.meta.url);
const { resolve, runtimeEnvironment } = require("../electron/paths.cjs");
const temporary: string[] = [];
afterEach(() => temporary.splice(0).forEach((dir) => rmSync(dir, { recursive: true, force: true })));

// resolve() reports every path with forward slashes, so build expectations in
// that shape; node's join() uses backslashes on Windows.
const at = (...segments: string[]) => join(...segments).replace(/\\/g, "/");

function installation() {
  const directory = at(mkdtempSync(join(tmpdir(), "studio installed ")));
  temporary.push(directory);
  const prefix = at(directory, "application");
  const app = at(prefix, "share", "hipdnn", "graph-studio");
  mkdirSync(app, { recursive: true });
  const config = {
    mode: "installed",
    root: "../../..",
    appDir: ".",
    distDir: "dist",
    addonPath: "native/hipdnn_engine.node",
    backendLibrary: "../../../lib/libhipdnn_backend.so",
    pluginDir: "../../../lib/hipdnn_plugins/engines",
    pythonExecutable: at(directory, "python environment", "bin", "python"),
    electronExecutable: "../../../libexec/hipdnn-graph-studio/electron",
    runtimeDirs: ["../../../lib", at(directory, "wheel", "lib")],
    binDirs: ["../../../bin", at(directory, "python environment", "bin")],
  };
  const configFile = at(app, "runtime-config.json");
  writeFileSync(configFile, JSON.stringify(config));
  return { directory, prefix, configFile };
}

describe("installed runtime selection", () => {
  test("rebases application artifacts after moving the install prefix", () => {
    const { directory, prefix } = installation();
    const moved = at(directory, "moved application");
    renameSync(prefix, moved);
    const paths = resolve({
      HIPDNN_BUILD_CONFIG: at(moved, "share/hipdnn/graph-studio/runtime-config.json"),
      GRAPH_STUDIO_OUT_DIR: at(directory, "obsolete build"),
    });
    expect(paths.root).toBe(moved);
    expect(paths.addonPath).toBe(at(moved, "share/hipdnn/graph-studio/native/hipdnn_engine.node"));
    expect(paths.pluginDir).toBe(at(moved, "lib/hipdnn_plugins/engines"));
    expect(paths.pythonExecutable).toBe(at(directory, "python environment/bin/python"));
  });

  test.skipIf(process.platform === "win32")("installed commands beat inherited SDK executables", () => {
    const { directory, prefix, configFile } = installation();
    const oldBin = at(directory, "old sdk", "bin");
    for (const [bin, label] of [[at(prefix, "bin"), "installed"], [oldBin, "old-sdk"]]) {
      mkdirSync(bin, { recursive: true });
      writeFileSync(at(bin, "selected-tool"), `#!/bin/sh\nprintf '${label}'\n`, { mode: 0o755 });
    }
    const paths = resolve({ HIPDNN_BUILD_CONFIG: configFile });
    const environment = runtimeEnvironment(paths, {
      PATH: [oldBin, "/usr/bin", "/bin"].join(delimiter),
      LD_LIBRARY_PATH: at(directory, "old sdk", "lib"),
      HIPDNN_SDK: at(directory, "old sdk"),
      VITE_DEV_SERVER_URL: "http://obsolete-development-server",
    });
    const child = spawnSync("selected-tool", [], { env: environment, cwd: directory, encoding: "utf8" });
    expect(child.status).toBe(0);
    expect(child.stdout).toBe("installed");
    expect(environment.LD_LIBRARY_PATH.split(delimiter)[0]).toBe(at(prefix, "lib"));
    expect(environment.HIPDNN_SDK).toBe(prefix);
    expect(environment.VITE_DEV_SERVER_URL).toBeUndefined();
  });

  test("an incomplete installed configuration cannot fall back to another SDK", () => {
    const { directory, configFile } = installation();
    writeFileSync(configFile, JSON.stringify({ mode: "installed", root: "../../.." }));
    expect(() => resolve({ HIPDNN_BUILD_CONFIG: configFile, HIPDNN_SDK: directory }))
      .toThrow("missing installed path");
  });
});
