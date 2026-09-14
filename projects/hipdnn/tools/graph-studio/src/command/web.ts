import type { CommandResult, CommandRunner } from "./types";

/**
 * Web command runner: a browser cannot spawn processes, so every run fails with
 * a clear message and the UI keeps its controls disabled.
 */

const UNAVAILABLE =
  "Running commands is only available in the Electron desktop build.";

export const webCommandRunner: CommandRunner = {
  available: false,
  async execute(): Promise<CommandResult> {
    return { ok: false, error: UNAVAILABLE };
  },
  async cancel(): Promise<void> {
    /* nothing can be running */
  },
  onOutput(): () => void {
    return () => {};
  },
};
