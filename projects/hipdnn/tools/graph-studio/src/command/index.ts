import { detectElectronCommandRunner } from "./electron";
import type { CommandRunner } from "./types";
import { webCommandRunner } from "./web";

export { GRAPH_PLACEHOLDER } from "./types";
export type { CommandChunk, CommandRequest, CommandResult, CommandRunner } from "./types";

/**
 * The active command runner. Native process spawning under Electron; an
 * "unavailable" stub in a plain browser. Chosen once at load, mirroring the
 * platform and engine selectors.
 */
export const commandRunner: CommandRunner =
  detectElectronCommandRunner() ?? webCommandRunner;
