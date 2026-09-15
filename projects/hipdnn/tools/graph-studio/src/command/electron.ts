import type { CommandChunk, CommandRequest, CommandResult, CommandRunner } from "./types";

/**
 * Electron command runner. Active when the preload has exposed
 * `window.hipdnnCommand`. The shape below is the IPC contract the preload must
 * satisfy; the main process owns the temp file, the spawn and the output pump.
 */

export interface ElectronCommandApi {
  execute(request: CommandRequest): Promise<CommandResult>;
  cancel(id: string): Promise<void>;
  onOutput(listener: (chunk: CommandChunk) => void): () => void;
}

interface CommandWindow {
  hipdnnCommand?: ElectronCommandApi;
}

export function detectElectronCommandRunner(): CommandRunner | null {
  const api = (window as unknown as CommandWindow).hipdnnCommand;
  if (!api) return null;
  return { available: true, ...api };
}
