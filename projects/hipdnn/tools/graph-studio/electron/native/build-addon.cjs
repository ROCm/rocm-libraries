// Builds the N-API add-on with node-gyp.
//
// node-gyp always reads binding.gyp from its cwd and writes its tree to
// <cwd>/build, so building outside the source directory means staging
// binding.gyp -- plus the hipdnn-config.gypi it includes -- into the target
// directory and running node-gyp there. binding.gyp reaches its sources through
// hipdnn_native_src_dir, so it works from either location.

"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

const paths = require("../paths.cjs");

const resolved = paths.resolve();
const gypDir = resolved.gypDir;

fs.mkdirSync(gypDir, { recursive: true });

const config = {
  variables: {
    hipdnn_native_src_dir: resolved.nativeSrcDir,
    hipdnn_napi_include_dir: paths.napiIncludeDir(),
    hipdnn_include_dirs: resolved.includeDirs,
    hipdnn_defines: resolved.defines,
    hipdnn_libraries: resolved.libraries,
  },
};
fs.writeFileSync(
  path.join(gypDir, "hipdnn-config.gypi"),
  `${JSON.stringify(config, null, 2)}\n`,
  "utf8",
);

if (path.resolve(gypDir) !== path.resolve(paths.NATIVE_DIR)) {
  fs.copyFileSync(paths.BINDING_GYP, path.join(gypDir, "binding.gyp"));
}

process.stderr.write(`hipDNN: ${resolved.mode} build at ${resolved.root}\n`);
process.stderr.write(`add-on: ${gypDir}\n`);

const nodeGyp = require.resolve("node-gyp/bin/node-gyp.js", { paths: [paths.NATIVE_DIR] });
const result = spawnSync(process.execPath, [nodeGyp, "rebuild"], { cwd: gypDir, stdio: "inherit" });
process.exit(result.status ?? 1);
