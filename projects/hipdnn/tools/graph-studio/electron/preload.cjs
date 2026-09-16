// Preload: the ONLY code with access to both Node and the renderer. It exposes
// four frozen, minimal APIs over contextBridge that exactly match the shapes
// the renderer's platform/index.ts, engine/index.ts, command/index.ts and
// flow/index.ts detect. Everything else in the renderer stays sandboxed.

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("hipdnn", {
  openTextFile: (accept) => ipcRenderer.invoke("platform:openTextFile", accept),
  saveTextFile: (contents, options) =>
    ipcRenderer.invoke("platform:saveTextFile", contents, options),
  readTensorArtifact: (manifestPath, reportPath) =>
    ipcRenderer.invoke("tensors:read", manifestPath, reportPath),
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
  serializeGraph: (graphJson) => ipcRenderer.invoke("engine:serializeGraph", graphJson),
  listEngines: (graphJson) => ipcRenderer.invoke("engine:listEngines", graphJson),
  execute: (handle, options) => ipcRenderer.invoke("engine:execute", handle, options),
  release: (handle) => ipcRenderer.invoke("engine:release", handle),
  setLogLevel: (level) => ipcRenderer.invoke("engine:setLogLevel", level),
});

contextBridge.exposeInMainWorld("hipdnnCommand", {
  execute: (request) => ipcRenderer.invoke("command:execute", request),
  cancel: (id) => ipcRenderer.invoke("command:cancel", id),
  onOutput: (listener) => {
    const handler = (_event, chunk) => listener(chunk);
    ipcRenderer.on("command:output", handler);
    return () => ipcRenderer.removeListener("command:output", handler);
  },
});

contextBridge.exposeInMainWorld("hipdnnFlow", {
  available: () => ipcRenderer.invoke("flow:available"),
  list: () => ipcRenderer.invoke("flow:list"),
  inputs: (flow) => ipcRenderer.invoke("flow:inputs", { flow }),
  validate: (request) => ipcRenderer.invoke("flow:validate", request),
  launch: (request) => ipcRenderer.invoke("flow:launch", request),
  cancel: (runId) => ipcRenderer.invoke("flow:cancel", { runId }),
  status: (runId, logTail) => ipcRenderer.invoke("flow:status", { runId, logTail }),
  artifact: (uri) => ipcRenderer.invoke("flow:artifact", { uri }),
  revealArtifact: (uri) => ipcRenderer.invoke("flow:revealArtifact", { uri }),
  onEvent: (listener) => {
    const handler = (_event, flowEvent) => listener(flowEvent);
    ipcRenderer.on("flow:event", handler);
    return () => ipcRenderer.removeListener("flow:event", handler);
  },
});
