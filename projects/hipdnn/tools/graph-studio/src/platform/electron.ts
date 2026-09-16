import type {
  FileHandleRef,
  OpenResult,
  PlatformBridge,
  SaveOptions,
  TensorArtifact,
} from "./types";

/**
 * Electron platform. Active when a preload script has exposed `window.hipdnn`
 * via contextBridge. Until the Electron shell exists this file is dormant —
 * `detectElectronPlatform()` returns null and the app uses the web bridge — but
 * the contract below is what the eventual preload must satisfy, so porting is
 * additive rather than a rewrite.
 */

// Shape the preload must expose over contextBridge. Mirrors PlatformBridge but
// with plain serializable payloads suitable for IPC.
export interface ElectronApi {
  openTextFile(accept?: string): Promise<{ path: string; name: string; contents: string } | null>;
  saveTextFile(
    contents: string,
    options?: { suggestedName?: string; path?: string },
  ): Promise<{ path: string; name: string } | null>;
  readTensorArtifact(
    manifestPath: string,
    reportPath: string,
  ): Promise<
    | { ok: true; manifest: string; files: Record<string, Uint8Array> }
    | { ok: false; error: string }
  >;
  store: {
    get(key: string): Promise<string | null>;
    set(key: string, value: string): Promise<void>;
    remove(key: string): Promise<void>;
  };
}

interface ElectronWindow {
  hipdnn?: ElectronApi;
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
    async readTensorArtifact(manifestPath: string, reportPath: string): Promise<TensorArtifact> {
      const result = await api.readTensorArtifact(manifestPath, reportPath);
      if (!result.ok) throw new Error(result.error);
      return { manifest: result.manifest, files: result.files };
    },
  };
}
