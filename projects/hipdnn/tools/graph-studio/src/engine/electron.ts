import type {
  BuildHandle,
  BuildOptions,
  BuildPlanResult,
  EngineBridge,
  EngineInfo,
  ExecuteOptions,
  ExecuteResult,
  ListEnginesResult,
  LogLevel,
  SerializeGraphResult,
} from "./types";

/**
 * Electron engine bridge. Active when the preload has exposed `window.hipdnnEngine`
 * over contextBridge. That object forwards each call over IPC to the main
 * process, which drives the native N-API addon (see electron/native/). The
 * shapes below are the IPC contract; they intentionally match {@link EngineBridge}
 * one-to-one so the preload is a thin `ipcRenderer.invoke` shim.
 */

export interface ElectronEngineApi {
  info(): Promise<EngineInfo>;
  build(graphJson: string, options?: BuildOptions): Promise<BuildPlanResult>;
  buildHipdnnJson(hipdnnJson: string, options?: BuildOptions): Promise<BuildPlanResult>;
  serializeGraph(graphJson: string): Promise<SerializeGraphResult>;
  listEngines(graphJson: string): Promise<ListEnginesResult>;
  execute(handle: BuildHandle, options: ExecuteOptions): Promise<ExecuteResult>;
  release(handle: BuildHandle): Promise<void>;
  setLogLevel(level: LogLevel): Promise<void>;
}

interface EngineWindow {
  hipdnnEngine?: ElectronEngineApi;
}

export function detectElectronEngine(): EngineBridge | null {
  const api = (window as unknown as EngineWindow).hipdnnEngine;
  return api ?? null;
}
