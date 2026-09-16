import type {
  DirectoryRef,
  FileHandleRef,
  OpenResult,
  PlatformBridge,
  ReadBase,
  SaveOptions,
} from "./types";

/**
 * Electron platform. Active when a preload script has exposed `window.hipdnn`
 * via contextBridge. Until the Electron shell exists this file is dormant —
 * `detectElectronPlatform()` returns null and the app uses the web bridge — but
 * the contract below is what the eventual preload must satisfy, so porting is
 * additive rather than a rewrite.
 */

// The main process needs to know which directory a relative path resolves
// against: the report file's own directory, or a folder granted outright.
// `ReadBase`'s two variants are structurally identical (`{ name, token }`), so
// that discriminator has to be attached here, in the renderer, rather than
// guessed from the token's shape in main.
type IpcReadBase = { kind: "file" | "directory"; path: string };

// Shape the preload must expose over contextBridge. Mirrors PlatformBridge but
// with plain serializable payloads suitable for IPC.
export interface ElectronApi {
  openTextFile(accept?: string): Promise<{ path: string; name: string; contents: string } | null>;
  saveTextFile(
    contents: string,
    options?: { suggestedName?: string; path?: string },
  ): Promise<{ path: string; name: string } | null>;
  readRelated(base: IpcReadBase, relativePath: string): Promise<Uint8Array>;
  openDirectory(): Promise<{ path: string; name: string } | null>;
  store: {
    get(key: string): Promise<string | null>;
    set(key: string, value: string): Promise<void>;
    remove(key: string): Promise<void>;
  };
}

interface ElectronWindow {
  hipdnn?: ElectronApi;
}

// `openTextFile`'s handle carries a bare file path string as its token (see
// `saveTextFile` below, which already relies on that). `openDirectory` below
// wraps its path in an object so the two remain distinguishable at runtime.
function toIpcBase(base: ReadBase | null | undefined): IpcReadBase | null {
  if (!base) return null;
  if (typeof base.token === "string") return { kind: "file", path: base.token };
  if (
    base.token !== null &&
    typeof base.token === "object" &&
    "path" in base.token &&
    typeof base.token.path === "string"
  ) {
    return { kind: "directory", path: base.token.path };
  }
  return null;
}

export function detectElectronPlatform(): PlatformBridge | null {
  const api = (window as unknown as ElectronWindow).hipdnn;
  if (!api) return null;

  return {
    kind: "electron",
    store: api.store,
    async openTextFile(accept?: string): Promise<OpenResult | null> {
      const result = await api.openTextFile(accept);
      if (!result) return null;
      return {
        handle: { name: result.name, token: result.path },
        contents: result.contents,
      };
    },
    async saveTextFile(
      contents: string,
      options?: SaveOptions,
    ): Promise<FileHandleRef | null> {
      const path = typeof options?.handle?.token === "string" ? options.handle.token : undefined;
      const result = await api.saveTextFile(contents, {
        suggestedName: options?.suggestedName,
        path,
      });
      if (!result) return null;
      return { name: result.name, token: result.path };
    },
    canReadRelated(base) {
      return toIpcBase(base) !== null;
    },
    async readRelated(base: ReadBase, relativePath: string): Promise<Uint8Array | null> {
      const ipcBase = toIpcBase(base);
      if (!ipcBase) return null;
      return api.readRelated(ipcBase, relativePath);
    },
    async openDirectory(): Promise<DirectoryRef | null> {
      const result = await api.openDirectory();
      if (!result) return null;
      return { name: result.name, token: { path: result.path } };
    },
  };
}
