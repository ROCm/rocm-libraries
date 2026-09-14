// Preload: the ONLY code with access to both Node and the renderer. It exposes
// two frozen, minimal APIs over contextBridge that exactly match the shapes the
// renderer's platform/index.ts and engine/index.ts detect. Everything else in
// the renderer stays sandboxed.

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("hipdnn", {
  openTextFile: (accept) => ipcRenderer.invoke("platform:openTextFile", accept),
  saveTextFile: (contents, options) =>
    ipcRenderer.invoke("platform:saveTextFile", contents, options),
  store: {
    get: (key) => ipcRenderer.invoke("store:get", key),
    set: (key, value) => ipcRenderer.invoke("store:set", key, value),
    remove: (key) => ipcRenderer.invoke("store:remove", key),
  },
});

contextBridge.exposeInMainWorld("hipdnnEngine", {
  info: () => ipcRenderer.invoke("engine:info"),
  build: (graphJson, options) => ipcRenderer.invoke("engine:build", graphJson, options),
  buildHipdnnJson: (hipdnnJson, options) =>
    ipcRenderer.invoke("engine:buildHipdnnJson", hipdnnJson, options),
  listEngines: (graphJson) => ipcRenderer.invoke("engine:listEngines", graphJson),
  execute: (handle, options) => ipcRenderer.invoke("engine:execute", handle, options),
  release: (handle) => ipcRenderer.invoke("engine:release", handle),
  setLogLevel: (level) => ipcRenderer.invoke("engine:setLogLevel", level),
});
