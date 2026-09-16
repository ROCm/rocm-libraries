/**
 * Platform bridge: the single seam between the app and its host environment.
 *
 * The web build talks to browser APIs (File System Access / download fallback,
 * localStorage). An Electron build swaps in an implementation backed by IPC to
 * the main process (native dialogs, fs). App code depends ONLY on this
 * interface, never on `window`, `electron`, or `fs` directly — that keeps the
 * port to Electron a matter of adding one implementation file.
 */

export interface FileHandleRef {
  /** Human-facing name, e.g. "resnet.graph.json". */
  readonly name: string;
  /** Opaque token the platform uses to write back to the same location. */
  readonly token: unknown;
}

export interface OpenResult {
  readonly handle: FileHandleRef;
  readonly contents: string;
}

export interface SaveOptions {
  readonly suggestedName?: string;
  /** Reuse a prior handle to save without a dialog ("Save" vs "Save As"). */
  readonly handle?: FileHandleRef;
}

export type PlatformKind = "web" | "electron";

/** One tensor artifact directory: its manifest text and every sibling `.bin`. */
export interface TensorArtifact {
  readonly manifest: string;
  readonly files: Readonly<Record<string, Uint8Array>>;
}

export interface PlatformBridge {
  readonly kind: PlatformKind;

  /** Prompt for and read a single text file. Resolves null if cancelled. */
  openTextFile(accept?: string): Promise<OpenResult | null>;

  /**
   * Persist text. Returns the handle actually written to (may be freshly
   * created), or null if the user cancelled the dialog.
   */
  saveTextFile(contents: string, options?: SaveOptions): Promise<FileHandleRef | null>;

  /**
   * Read a tensor capture by the manifest path a report recorded, resolved
   * against that report's own location. Present only where the host has a
   * filesystem; absent means captures must be picked as files instead.
   * Rejects when the path is not readable.
   */
  readTensorArtifact?(manifestPath: string, reportPath: string): Promise<TensorArtifact>;

  /** Best-effort key/value persistence for app state and preferences. */
  readonly store: KeyValueStore;
}

export interface KeyValueStore {
  get(key: string): Promise<string | null>;
  set(key: string, value: string): Promise<void>;
  remove(key: string): Promise<void>;
}
