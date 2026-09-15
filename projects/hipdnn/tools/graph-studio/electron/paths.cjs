// Where everything lives: the hipDNN the add-on is built and loaded against,
// and the Studio's own build outputs.
//
// hipDNN has two layouts:
//
//   in-tree  A rocm-libraries build tree. The superbuild component
//            `hipdnn-graph-studio` writes graph-studio/build-config.json into
//            the build directory with the include dirs, compile definitions,
//            link libraries and runtime directories taken straight from the
//            CMake targets, so nothing here has to guess the tree's layout.
//
//   sdk      An installed ROCm/hipDNN SDK (the ROCm devel wheel, or /opt/rocm),
//            whose layout is fixed and spelled out in sdkPaths() below.
//
// Outputs (the bundled web assets and the node-gyp tree) default to the source
// directory and move under <build>/graph-studio when a build tree is in play.

"use strict";

const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");

const isWindows = process.platform === "win32";

// Written by the graph-studio superbuild component, relative to the build dir.
const BUILD_CONFIG_RELPATH = path.join("graph-studio", "build-config.json");

// Fallback when nothing else identifies a hipDNN: the ROCm devel wheel layout
// used by the Windows developer setup.
const DEFAULT_SDK = "D:/develop/latest_wheels_nightly/Lib/site-packages/_rocm_sdk_devel";

const STUDIO_DIR = path.resolve(__dirname, "..");
const NATIVE_DIR = path.join(STUDIO_DIR, "electron", "native");
const NATIVE_SRC_DIR = path.join(NATIVE_DIR, "src");
const BINDING_GYP = path.join(NATIVE_DIR, "binding.gyp");

// graph-studio -> tools -> hipdnn -> projects -> repository root
const REPO_ROOT = path.resolve(STUDIO_DIR, "..", "..", "..", "..");

/** Drop the empty entries CMake generator expressions leave behind. */
function compact(list) {
  return (Array.isArray(list) ? list : []).filter((item) => typeof item === "string" && item !== "");
}

// gyp and CMake both emit forward slashes on Windows; keep node's path.join
// output in the same shape so the generated files are uniform.
function slashes(value) {
  return value.replace(/\\/g, "/");
}

function compactPaths(list) {
  return compact(list).map(slashes);
}

/** Where the add-on's node-gyp tree and the web bundle are written. */
function outputPaths(outDir) {
  const root = outDir ?? STUDIO_DIR;
  const gypDir = outDir ? path.join(outDir, "native") : NATIVE_DIR;
  return {
    outDir: slashes(root),
    distDir: slashes(path.join(root, "dist")),
    gypDir: slashes(gypDir),
    addonPath: slashes(path.join(gypDir, "build", "Release", "hipdnn_engine.node")),
  };
}

function buildTreePaths(configFile) {
  const config = JSON.parse(fs.readFileSync(configFile, "utf8"));
  const base = path.dirname(path.resolve(configFile));
  const resolvePath = (value) => typeof value === "string" && value
    ? slashes(path.resolve(base, value))
    : undefined;
  if (config.mode === "installed") {
    const result = { mode: "installed", source: slashes(path.resolve(configFile)) };
    for (const key of ["root", "appDir", "distDir", "addonPath", "backendLibrary",
      "pluginDir", "pythonExecutable", "electronExecutable"]) {
      result[key] = resolvePath(config[key]);
      if (!result[key]) throw new Error(`${configFile}: missing installed path ${key}`);
    }
    result.runtimeDirs = compact(config.runtimeDirs).map(resolvePath);
    result.binDirs = compact(config.binDirs).map(resolvePath);
    return result;
  }
  return {
    mode: "in-tree",
    root: slashes(config.buildDir ?? path.dirname(configFile)),
    source: slashes(configFile),
    outDir: config.outDir,
    includeDirs: compactPaths(config.includeDirs),
    defines: compact(config.defines),
    libraries: compactPaths(config.libraries),
    runtimeDirs: compactPaths(config.runtimeDirs),
    pluginDir: slashes(config.pluginDir ?? ""),
    backendLibrary: resolvePath(config.backendLibrary),
    pythonExecutable: resolvePath(config.pythonExecutable),
    binDirs: compact(config.binDirs).map(resolvePath),
  };
}

function sdkPaths(sdk) {
  const lib = path.join(sdk, "lib");
  const runtimeDir = path.join(sdk, isWindows ? "bin" : "lib");
  const includeDirs = [
    path.join(sdk, "include"),
    path.join(sdk, "include", "hipdnn", "frontend"),
    path.join(sdk, "include", "hipdnn", "backend"),
    path.join(sdk, "include", "hipdnn", "data_sdk"),
    path.join(sdk, "include", "hipdnn", "flatbuffers_sdk"),
    path.join(sdk, "include", "hipdnn", "plugin_sdk"),
  ];

  let libraries;
  if (isWindows) {
    libraries = [path.join(lib, "hipdnn_backend.lib"), path.join(lib, "amdhip64.lib")];
  } else {
    const rocm = process.env.ROCM_PATH ?? "/opt/rocm";
    includeDirs.push(path.join(rocm, "include"));
    libraries = [
      `-L${lib}`,
      `-L${path.join(rocm, "lib")}`,
      "-lhipdnn_backend",
      "-lamdhip64",
      `-Wl,-rpath,${lib}`,
    ];
  }

  return {
    mode: "sdk",
    root: slashes(sdk),
    source: slashes(sdk),
    outDir: undefined,
    includeDirs: compactPaths(includeDirs),
    // The installed public headers gate their nlohmann/json API behind this
    // macro; the add-on uses the string-based serialize/deserialize instead.
    defines: ["HIPDNN_FRONTEND_SKIP_JSON_LIB"],
    libraries: compactPaths(libraries),
    runtimeDirs: compactPaths([runtimeDir]),
    pluginDir: slashes(path.join(runtimeDir, "hipdnn_plugins", "engines")),
  };
}

function hipdnnPaths(env) {
  if (env.HIPDNN_BUILD_CONFIG) {
    return buildTreePaths(env.HIPDNN_BUILD_CONFIG);
  }

  const installed = path.join(STUDIO_DIR, "runtime-config.json");
  if (fs.existsSync(installed)) {
    return buildTreePaths(installed);
  }

  if (env.HIPDNN_BUILD_DIR) {
    const configFile = path.join(env.HIPDNN_BUILD_DIR, BUILD_CONFIG_RELPATH);
    if (!fs.existsSync(configFile)) {
      throw new Error(
        `HIPDNN_BUILD_DIR=${env.HIPDNN_BUILD_DIR} has no ${BUILD_CONFIG_RELPATH}. ` +
          "Configure the build with the 'hipdnn-graph-studio' preset and build it.",
      );
    }
    return buildTreePaths(configFile);
  }

  if (env.HIPDNN_SDK) {
    return sdkPaths(env.HIPDNN_SDK);
  }

  const inTree = path.join(REPO_ROOT, "build", BUILD_CONFIG_RELPATH);
  if (fs.existsSync(inTree)) {
    return buildTreePaths(inTree);
  }

  return sdkPaths(DEFAULT_SDK);
}

/**
 * Resolve hipDNN, in this order:
 *   1. HIPDNN_BUILD_CONFIG  - explicit build-config.json
 *   2. HIPDNN_BUILD_DIR     - a build tree that must contain one
 *   3. HIPDNN_SDK           - an installed SDK
 *   4. <repo>/build         - the conventional in-tree superbuild
 *   5. DEFAULT_SDK
 *
 * Output locations come from GRAPH_STUDIO_OUT_DIR, else the build tree's
 * outDir, else the source directory.
 */
function resolve(env = process.env) {
  const hipdnn = hipdnnPaths(env);
  if (hipdnn.mode === "installed") return hipdnn;
  return {
    ...hipdnn,
    ...outputPaths(env.GRAPH_STUDIO_OUT_DIR ?? hipdnn.outDir ?? null),
    nativeSrcDir: slashes(NATIVE_SRC_DIR),
  };
}

/** Establish loader paths before launching Electron or the benchmark interpreter. */
function runtimeEnvironment(paths, env = process.env) {
  const result = { ...env };
  const prepend = (entries, current) => [...new Set([
    ...compact(entries),
    ...(current ? current.split(path.delimiter).filter(Boolean) : []),
  ])].join(path.delimiter);
  result.PATH = prepend([
    path.join(paths.root, "bin"),
    ...(paths.binDirs ?? []),
    ...(paths.pythonExecutable ? [path.dirname(paths.pythonExecutable)] : []),
  ], result.PATH);
  if (isWindows) {
    result.PATH = prepend(paths.runtimeDirs, result.PATH);
  } else {
    result.LD_LIBRARY_PATH = prepend(paths.runtimeDirs, result.LD_LIBRARY_PATH);
  }
  result.HIPDNN_SDK = paths.root;
  result.HIPDNN_PLUGIN_DIR = paths.pluginDir;
  if (!result.DNN_BENCH_WORKSPACE) {
    const cache = result.XDG_CACHE_HOME
      || (isWindows ? result.LOCALAPPDATA : null)
      || path.join(os.homedir(), ".cache");
    result.DNN_BENCH_WORKSPACE = path.join(cache, "hipdnn-graph-studio");
  }
  if (paths.mode === "installed") delete result.VITE_DEV_SERVER_URL;
  return result;
}

/** Directory holding napi.h, for the add-on's include path. */
function napiIncludeDir() {
  return slashes(path.dirname(require.resolve("node-addon-api", { paths: [NATIVE_DIR] })));
}

module.exports = {
  resolve,
  runtimeEnvironment,
  napiIncludeDir,
  BUILD_CONFIG_RELPATH,
  STUDIO_DIR,
  NATIVE_DIR,
  BINDING_GYP,
};
