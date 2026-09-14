// electron/preload.js
var { contextBridge, ipcRenderer } = require("electron");
contextBridge.exposeInMainWorld("hipdnn", {
  openTextFile: (accept) => ipcRenderer.invoke("platform:openTextFile", accept),
  saveTextFile: (contents, options) => ipcRenderer.invoke("platform:saveTextFile", contents, options),
  store: {
    get: (key) => ipcRenderer.invoke("store:get", key),
    set: (key, value) => ipcRenderer.invoke("store:set", key, value),
    remove: (key) => ipcRenderer.invoke("store:remove", key)
  }
});
contextBridge.exposeInMainWorld("hipdnnEngine", {
  info: () => ipcRenderer.invoke("engine:info"),
  build: (graphJson) => ipcRenderer.invoke("engine:build", graphJson),
  execute: (handle, options) => ipcRenderer.invoke("engine:execute", handle, options),
  release: (handle) => ipcRenderer.invoke("engine:release", handle)
});
