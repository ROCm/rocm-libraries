// Electron main process. Owns the BrowserWindow, native file dialogs, and the
// bridge to the native hipDNN addon. The renderer reaches all of this only
// through IPC channels registered here and surfaced by preload.js.

const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const fs = require("node:fs/promises");
const { spawn } = require("node:child_process");
const { existsSync } = require("node:fs");
const path = require("node:path");

const studioPaths = require("./paths.cjs");

// Dev mode = a Vite dev server URL was provided (set by the electron:dev
// script). electron:start builds first and loads the bundled files instead.
const DEV_URL = process.env.VITE_DEV_SERVER_URL;
const isDev = Boolean(DEV_URL);

// The native addon is optional: if it hasn't been built (or hipDNN isn't
// installed) the app still runs, and the engine reports itself unavailable.
//
// The addon needs hipdnn_backend + the HIP runtime at load time, and hipDNN
// needs HIPDNN_PLUGIN_DIR to discover the engine plugins. Both come from
// wherever the addon was built against -- an in-tree build tree or an installed
// SDK -- so resolve that here rather than making the user set up a shell.
let paths = null;
let nativeEngine = null;
let nativeLoadError = "";
try {
  paths = studioPaths.resolve();
} catch (err) {
  nativeLoadError = err instanceof Error ? err.message : String(err);
}

if (paths) {
  console.log(`hipDNN: ${paths.mode} build at ${paths.root}`);
  try {
    for (const dir of paths.runtimeDirs) {
      process.env.PATH = `${dir}${path.delimiter}${process.env.PATH ?? ""}`;
      if (process.platform === "win32" && typeof process.addDllDirectory === "function" && existsSync(dir)) {
        process.addDllDirectory(dir);
      }
    }
    if (paths.pluginDir && !process.env.HIPDNN_PLUGIN_DIR) {
      process.env.HIPDNN_PLUGIN_DIR = paths.pluginDir;
    }
    nativeEngine = require(paths.addonPath);
  } catch (err) {
    nativeLoadError = err instanceof Error ? err.message : String(err);
  }
}

// Falling back to the in-source bundle keeps the window loadable even when
// hipDNN resolution failed outright.
const distDir = paths?.distDir ?? path.join(__dirname, "..", "dist");

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    backgroundColor: "#0d1117",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  if (isDev) {
    void win.loadURL(DEV_URL);
    win.webContents.openDevTools({ mode: "detach" });
  } else {
    void win.loadFile(path.join(distDir, "index.html"));
  }
  return win;
}

// ── File I/O (platform bridge) ─────────────────────────────────────────
ipcMain.handle("platform:openTextFile", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openFile"],
    filters: [{ name: "Graph JSON", extensions: ["json"] }],
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  const filePath = result.filePaths[0];
  const contents = await fs.readFile(filePath, "utf8");
  return { path: filePath, name: path.basename(filePath), contents };
});

ipcMain.handle("platform:saveTextFile", async (_event, contents, options) => {
  let filePath = options?.path;
  if (!filePath) {
    const result = await dialog.showSaveDialog({
      defaultPath: options?.suggestedName ?? "graph.json",
      filters: [{ name: "Graph JSON", extensions: ["json"] }],
    });
    if (result.canceled || !result.filePath) return null;
    filePath = result.filePath;
  }
  await fs.writeFile(filePath, contents, "utf8");
  return { path: filePath, name: path.basename(filePath) };
});

// ── Key/value store, persisted as JSON in userData ─────────────────────
const storePath = () => path.join(app.getPath("userData"), "studio-store.json");

async function readStore() {
  try {
    return JSON.parse(await fs.readFile(storePath(), "utf8"));
  } catch {
    return {};
  }
}
async function writeStore(data) {
  await fs.writeFile(storePath(), JSON.stringify(data), "utf8");
}

ipcMain.handle("store:get", async (_event, key) => {
  const data = await readStore();
  return Object.prototype.hasOwnProperty.call(data, key) ? data[key] : null;
});
ipcMain.handle("store:set", async (_event, key, value) => {
  const data = await readStore();
  data[key] = value;
  await writeStore(data);
});
ipcMain.handle("store:remove", async (_event, key) => {
  const data = await readStore();
  delete data[key];
  await writeStore(data);
});

// ── hipDNN engine (native addon) ───────────────────────────────────────
function engineUnavailable(message) {
  return {
    ok: false,
    error: { code: "ENGINE_UNAVAILABLE", message },
    log: [message],
  };
}

ipcMain.handle("engine:info", async () => {
  if (!nativeEngine) {
    return {
      available: false,
      backend: `unavailable (native addon not loaded: ${nativeLoadError})`,
    };
  }
  return nativeEngine.info();
});

ipcMain.handle("engine:build", async (_event, graphJson, options) => {
  if (!nativeEngine) return engineUnavailable("Native hipDNN addon is not loaded.");
  return nativeEngine.build(graphJson, options ?? {});
});

ipcMain.handle("engine:buildHipdnnJson", async (_event, hipdnnJson, options) => {
  if (!nativeEngine) return engineUnavailable("Native hipDNN addon is not loaded.");
  return nativeEngine.buildHipdnnJson(hipdnnJson, options ?? {});
});

ipcMain.handle("engine:serializeGraph", async (_event, graphJson) => {
  if (!nativeEngine) return engineUnavailable("Native hipDNN addon is not loaded.");
  return nativeEngine.serializeGraph(graphJson);
});

ipcMain.handle("engine:listEngines", async (_event, graphJson) => {
  if (!nativeEngine) {
    return { ...engineUnavailable("Native hipDNN addon is not loaded."), engines: [] };
  }
  return nativeEngine.listEngines(graphJson);
});

ipcMain.handle("engine:execute", async (_event, handle, options) => {
  if (!nativeEngine) return engineUnavailable("Native hipDNN addon is not loaded.");
  return nativeEngine.execute(handle, options);
});

ipcMain.handle("engine:release", async (_event, handle) => {
  if (nativeEngine) nativeEngine.release(handle);
});

ipcMain.handle("engine:setLogLevel", async (_event, level) => {
  if (nativeEngine) nativeEngine.setLogLevel(level);
});

// ── External commands (command bridge) ─────────────────────────────────
// Each run writes the graph out as hipDNN JSON under the OS temp directory,
// substitutes that path for ${current_graph}, and streams the child's output
// back to the renderer as it arrives.

const GRAPH_PLACEHOLDER = "${current_graph}";

/** Running children by request id, so the renderer can cancel them. */
const runningCommands = new Map();

const errorText = (err) => (err instanceof Error ? err.message : String(err));

const sanitizeName = (name) =>
  String(name ?? "").replace(/[^\w.-]+/g, "_").replace(/^[._]+|[._]+$/g, "") || "graph";

// The path is quoted unless the placeholder already sits between quotes, so a
// temp directory containing spaces survives the shell either way.
function substituteGraphPath(command, graphPath) {
  let out = "";
  let from = 0;
  for (;;) {
    const at = command.indexOf(GRAPH_PLACEHOLDER, from);
    if (at < 0) return out + command.slice(from);
    const preQuoted =
      command[at - 1] === '"' && command[at + GRAPH_PLACEHOLDER.length] === '"';
    out += command.slice(from, at) + (preQuoted ? graphPath : `"${graphPath}"`);
    from = at + GRAPH_PLACEHOLDER.length;
  }
}

async function writeGraphFile(scope, graphName, graphJson) {
  const dir = path.join(app.getPath("temp"), "hipdnn-graph-studio", sanitizeName(scope));
  await fs.mkdir(dir, { recursive: true });
  const file = path.join(dir, `${sanitizeName(graphName)}.hipdnn.json`);
  await fs.writeFile(file, graphJson ?? "", "utf8");
  return file;
}

ipcMain.handle("command:execute", async (event, request) => {
  const { id, command, graphJson, graphName, scope } = request ?? {};
  if (typeof command !== "string" || command.trim() === "") {
    return { ok: false, error: "Command is empty." };
  }
  if (runningCommands.has(id)) {
    return { ok: false, error: "A command is already running in this tab." };
  }

  let graphPath;
  try {
    graphPath = await writeGraphFile(scope, graphName, graphJson);
  } catch (err) {
    return { ok: false, error: `Failed to write the graph: ${errorText(err)}` };
  }

  const resolvedCommand = substituteGraphPath(command, graphPath);
  const send = (stream, text) => {
    if (!event.sender.isDestroyed()) event.sender.send("command:output", { id, stream, text });
  };

  return new Promise((resolve) => {
    let settled = false;
    const finish = (result) => {
      if (settled) return;
      settled = true;
      runningCommands.delete(id);
      resolve({ ...result, resolvedCommand, graphPath });
    };

    let child;
    try {
      child = spawn(resolvedCommand, { shell: true, windowsHide: true });
    } catch (err) {
      finish({ ok: false, error: errorText(err) });
      return;
    }
    runningCommands.set(id, child);

    child.stdout?.setEncoding("utf8");
    child.stdout?.on("data", (text) => send("stdout", text));
    child.stderr?.setEncoding("utf8");
    child.stderr?.on("data", (text) => send("stderr", text));
    child.on("error", (err) => finish({ ok: false, error: errorText(err) }));
    child.on("close", (code, signal) => finish({ ok: code === 0, exitCode: code, signal }));
  });
});

// shell:true means the child is cmd.exe/sh, so the tool itself is a grandchild:
// kill the tree rather than just the shell.
function killCommand(child) {
  if (process.platform === "win32") {
    spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], { windowsHide: true });
  } else {
    child.kill("SIGTERM");
  }
}

ipcMain.handle("command:cancel", async (_event, id) => {
  const child = runningCommands.get(id);
  if (child) killCommand(child);
});

app.on("before-quit", () => {
  for (const child of runningCommands.values()) killCommand(child);
  runningCommands.clear();
});

app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
