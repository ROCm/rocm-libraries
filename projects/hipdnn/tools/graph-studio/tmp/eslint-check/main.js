// electron/main.js
var { app, BrowserWindow, ipcMain, dialog } = require("electron");
var fs = require("node:fs/promises");
var path = require("node:path");
var isDev = !app.isPackaged;
var DEV_URL = process.env.VITE_DEV_SERVER_URL ?? "http://localhost:5173";
var nativeEngine = null;
var nativeLoadError = "";
try {
  nativeEngine = require("./native/build/Release/hipdnn_engine.node");
} catch (err) {
  nativeLoadError = err instanceof Error ? err.message : String(err);
}
function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    backgroundColor: "#0d1117",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });
  if (isDev) {
    void win.loadURL(DEV_URL);
    win.webContents.openDevTools({ mode: "detach" });
  } else {
    void win.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  }
  return win;
}
ipcMain.handle("platform:openTextFile", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openFile"],
    filters: [{ name: "Graph JSON", extensions: ["json"] }]
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
      filters: [{ name: "Graph JSON", extensions: ["json"] }]
    });
    if (result.canceled || !result.filePath) return null;
    filePath = result.filePath;
  }
  await fs.writeFile(filePath, contents, "utf8");
  return { path: filePath, name: path.basename(filePath) };
});
var storePath = () => path.join(app.getPath("userData"), "studio-store.json");
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
function engineUnavailable(message) {
  return {
    ok: false,
    error: { code: "ENGINE_UNAVAILABLE", message },
    log: [message]
  };
}
ipcMain.handle("engine:info", async () => {
  if (!nativeEngine) {
    return {
      available: false,
      backend: `unavailable (native addon not loaded: ${nativeLoadError})`
    };
  }
  return nativeEngine.info();
});
ipcMain.handle("engine:build", async (_event, graphJson) => {
  if (!nativeEngine) return engineUnavailable("Native hipDNN addon is not loaded.");
  return nativeEngine.build(graphJson);
});
ipcMain.handle("engine:execute", async (_event, handle, options) => {
  if (!nativeEngine) return engineUnavailable("Native hipDNN addon is not loaded.");
  return nativeEngine.execute(handle, options);
});
ipcMain.handle("engine:release", async (_event, handle) => {
  if (nativeEngine) nativeEngine.release(handle);
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
