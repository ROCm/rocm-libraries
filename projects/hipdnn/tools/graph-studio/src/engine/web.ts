import type {
  BuildPlanResult,
  EngineBridge,
  EngineInfo,
  ExecuteResult,
  ListEnginesResult,
  SerializeGraphResult,
} from "./types";

/**
 * Web engine: no native hipDNN in a browser. Every operation resolves with a
 * clear "unavailable" result so the UI can present the JSON translation and
 * disable build/execute without special-casing the platform.
 */

const UNAVAILABLE = {
  code: "ENGINE_UNAVAILABLE",
  message:
    "The hipDNN runtime is only available in the Electron desktop build. " +
    "Run the desktop app to build and execute graphs on a GPU.",
} as const;

export const webEngine: EngineBridge = {
  async info(): Promise<EngineInfo> {
    return { available: false, backend: "unavailable (web build)" };
  },
  async build(): Promise<BuildPlanResult> {
    return { ok: false, error: UNAVAILABLE, log: [UNAVAILABLE.message] };
  },
  async buildHipdnnJson(): Promise<BuildPlanResult> {
    return { ok: false, error: UNAVAILABLE, log: [UNAVAILABLE.message] };
  },
  async serializeGraph(): Promise<SerializeGraphResult> {
    return { ok: false, error: UNAVAILABLE, log: [UNAVAILABLE.message] };
  },
  async listEngines(): Promise<ListEnginesResult> {
    return { ok: false, error: UNAVAILABLE, engines: [], log: [UNAVAILABLE.message] };
  },
  async execute(): Promise<ExecuteResult> {
    return { ok: false, error: UNAVAILABLE, log: [UNAVAILABLE.message] };
  },
  async release(): Promise<void> {
    /* nothing to release */
  },
  async setLogLevel(): Promise<void> {
    /* no engine to configure in the browser */
  },
};
