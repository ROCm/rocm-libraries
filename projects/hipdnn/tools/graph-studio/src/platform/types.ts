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

/** A folder the user granted, against which relative paths resolve. */
export interface DirectoryRef {
  readonly name: string;
  readonly token: unknown;
}

/** Where a report's relative paths resolve from: the report file, or a granted folder. */
export type ReadBase = FileHandleRef | DirectoryRef;

export type PlatformKind = "web" | "electron";

export interface PlatformBridge {
  readonly kind: PlatformKind;

  /** Prompt for and read a single text file. Resolves null if cancelled. */
  openTextFile(accept?: string): Promise<OpenResult | null>;

  /**
   * Persist text. Returns the handle actually written to (may be freshly
   * created), or null if the user cancelled the dialog.
   */
  saveTextFile(contents: string, options?: SaveOptions): Promise<FileHandleRef | null>;

  /** Best-effort key/value persistence for app state and preferences. */
  readonly store: KeyValueStore;

  /**
   * Whether `readRelated` can resolve paths against this base with no further
   * prompting. Electron knows the report's own directory; the web build only
   * knows a folder the user granted through `openDirectory`.
   */
  canReadRelated(base: ReadBase | null | undefined): boolean;

  /**
   * Read bytes a report refers to, resolved against `base`. Resolves null when
   * the host cannot read by path, and rejects when the file is missing or the
   * path escapes the base directory — a report is untrusted input and must not
   * be able to name an arbitrary file on the machine.
   */
  readRelated(base: ReadBase, relativePath: string): Promise<Uint8Array | null>;

  /**
   * Prompt for a folder whose files may then be read by relative path.
   * Resolves null when cancelled or when the host has no such capability.
   */
  openDirectory(): Promise<DirectoryRef | null>;

  /**
   * Whether `openDirectory` can do anything here. False in a browser without
   * the File System Access API, where no folder grant is possible at all.
   */
  canGrantDirectory(): boolean;

  /**
   * Names of the files directly inside a granted folder, so a caller can find
   * the report in a run directory without a second pick. Subdirectories are
   * not descended into and are not listed.
   */
  listFiles(base: DirectoryRef): Promise<readonly string[]>;
}

export interface KeyValueStore {
  get(key: string): Promise<string | null>;
  set(key: string, value: string): Promise<void>;
  remove(key: string): Promise<void>;
}
