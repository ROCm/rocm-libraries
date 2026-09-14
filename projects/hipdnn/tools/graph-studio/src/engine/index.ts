import { detectElectronEngine } from "./electron";
import type { EngineBridge } from "./types";
import { webEngine } from "./web";

export type {
  BuildHandle,
  BuildOptions,
  BuildPlanResult,
  BuildResult,
  EngineBridge,
  EngineError,
  EngineErrorCode,
  EngineInfo,
  EngineOption,
  ExecuteOptions,
  ExecuteResult,
  ListEnginesResult,
  LogEntry,
  LogLevel,
  LogSeverity,
} from "./types";

/**
 * The active engine bridge. Native hipDNN under Electron; an "unavailable" stub
 * in a plain browser. Chosen once at load, mirroring the platform selector.
 */
export const engine: EngineBridge = detectElectronEngine() ?? webEngine;
