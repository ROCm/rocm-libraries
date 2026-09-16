import type {
  DirectoryRef,
  FileHandleRef,
  KeyValueStore,
  OpenResult,
  PlatformBridge,
  SaveOptions,
} from "./types";

/**
 * Web platform. Prefers the File System Access API (Chromium) so "Save" can
 * write back to the same file; falls back to <input type=file> + download for
 * browsers without it.
 */

// Minimal structural typings for the File System Access API, which is not yet
// in the DOM lib. We only reference the members we use.
interface FSWritable {
  write(data: string): Promise<void>;
  close(): Promise<void>;
}
interface FSFileHandle {
  readonly name: string;
  getFile(): Promise<File>;
  createWritable(): Promise<FSWritable>;
}

// A granted folder (via `showDirectoryPicker`). Kept separate from
// `FSFileHandle` above: its leaf files never need `createWritable`, and
// keeping the shape minimal is what makes it easy to fake in tests.
export interface FSLeafFileHandle {
  getFile(): Promise<{ arrayBuffer(): Promise<ArrayBuffer> }>;
}
export interface FSDirectoryHandle {
  readonly name: string;
  getDirectoryHandle(name: string): Promise<FSDirectoryHandle>;
  getFileHandle(name: string): Promise<FSLeafFileHandle>;
  entries?: () => AsyncIterableIterator<[string, { kind: string }]>;
}
interface FSWindow {
  showOpenFilePicker?: (opts?: {
    types?: { description: string; accept: Record<string, string[]> }[];
  }) => Promise<FSFileHandle[]>;
  showSaveFilePicker?: (opts?: {
    suggestedName?: string;
    types?: { description: string; accept: Record<string, string[]> }[];
  }) => Promise<FSFileHandle>;
  showDirectoryPicker?: () => Promise<FSDirectoryHandle>;
}

const fsWindow = window as unknown as FSWindow;
const hasFileSystemAccess = typeof fsWindow.showOpenFilePicker === "function";

const localStore: KeyValueStore = {
  async get(key) {
    return localStorage.getItem(key);
  },
  async set(key, value) {
    localStorage.setItem(key, value);
  },
  async remove(key) {
    localStorage.removeItem(key);
  },
};

async function openViaFsApi(): Promise<OpenResult | null> {
  let handles: FSFileHandle[];
  try {
    handles = await fsWindow.showOpenFilePicker!({
      types: [{ description: "Graph JSON", accept: { "application/json": [".json"] } }],
    });
  } catch {
    // AbortError when the user dismisses the picker.
    return null;
  }
  const fsHandle = handles[0];
  const file = await fsHandle.getFile();
  const contents = await file.text();
  return { handle: { name: fsHandle.name, token: fsHandle }, contents };
}

function openViaInput(accept: string): Promise<OpenResult | null> {
  const { promise, resolve, reject } = Promise.withResolvers<OpenResult | null>();
  const input = document.createElement("input");
  input.type = "file";
  input.accept = accept;
  input.style.display = "none";

  let settled = false;
  const finish = (value: OpenResult | null) => {
    if (settled) return;
    settled = true;
    input.remove();
    resolve(value);
  };

  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (!file) {
      finish(null);
      return;
    }
    void file.text().then((contents) => {
      finish({ handle: { name: file.name, token: null }, contents });
    }, (error: unknown) => {
      if (settled) return;
      settled = true;
      input.remove();
      reject(error);
    });
  });
  // Fires when the picker is dismissed on browsers that support it.
  input.addEventListener("cancel", () => finish(null));

  document.body.append(input);
  input.click();
  return promise;
}

async function saveViaFsApi(
  contents: string,
  options: SaveOptions | undefined,
): Promise<FileHandleRef | null> {
  const existing = options?.handle?.token as FSFileHandle | undefined;
  let fsHandle: FSFileHandle;
  if (existing) {
    fsHandle = existing;
  } else {
    try {
      fsHandle = await fsWindow.showSaveFilePicker!({
        suggestedName: options?.suggestedName ?? "graph.json",
        types: [{ description: "Graph JSON", accept: { "application/json": [".json"] } }],
      });
    } catch {
      return null;
    }
  }
  const writable = await fsHandle.createWritable();
  await writable.write(contents);
  await writable.close();
  return { name: fsHandle.name, token: fsHandle };
}

function saveViaDownload(
  contents: string,
  options: SaveOptions | undefined,
): FileHandleRef {
  const name = options?.suggestedName ?? "graph.json";
  const blob = new Blob([contents], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
  // Download can't be written back to, so the handle carries no token.
  return { name, token: null };
}

function isDirectoryHandle(token: unknown): token is FSDirectoryHandle {
  return (
    typeof token === "object" &&
    token !== null &&
    typeof (token as FSDirectoryHandle).getDirectoryHandle === "function" &&
    typeof (token as FSDirectoryHandle).getFileHandle === "function"
  );
}

/**
 * Walks `relativePath` down from `dir` one segment at a time. A report is
 * untrusted input: refusing `..` and a leading `/` here, before any handle
 * lookup, is what stops it from naming a file outside the granted folder.
 */
export async function readRelatedFromDirectory(
  dir: FSDirectoryHandle,
  relativePath: string,
): Promise<Uint8Array> {
  if (relativePath.startsWith("/")) {
    throw new Error(`Absolute path not allowed: ${relativePath}`);
  }
  const segments = relativePath.split("/");
  if (segments.some((segment) => segment === "" || segment === "..")) {
    throw new Error(`Invalid relative path: ${relativePath}`);
  }
  let cursor = dir;
  try {
    for (const segment of segments.slice(0, -1)) {
      cursor = await cursor.getDirectoryHandle(segment);
    }
    const fileHandle = await cursor.getFileHandle(segments[segments.length - 1]);
    const file = await fileHandle.getFile();
    return new Uint8Array(await file.arrayBuffer());
  } catch {
    // The browser's own NotFoundError names neither path nor folder, which
    // hides the usual cause: a folder granted below the one the run wrote from.
    throw new Error(
      `${relativePath} not found under ${dir.name}/ — the report's artifacts must sit beside it`,
    );
  }
}

export const webPlatform: PlatformBridge = {
  kind: "web",
  store: localStore,
  openTextFile(accept = ".json,application/json") {
    return hasFileSystemAccess ? openViaFsApi() : openViaInput(accept);
  },
  async saveTextFile(contents, options) {
    return hasFileSystemAccess
      ? saveViaFsApi(contents, options)
      : saveViaDownload(contents, options);
  },
  canReadRelated(base) {
    return isDirectoryHandle(base?.token);
  },
  async readRelated(base, relativePath) {
    if (!isDirectoryHandle(base.token)) return null;
    return readRelatedFromDirectory(base.token, relativePath);
  },
  async listFiles(base) {
    const dir = base.token;
    if (!isDirectoryHandle(dir) || !dir.entries) return [];
    const names: string[] = [];
    for await (const [name, handle] of dir.entries()) {
      if (handle.kind === "file") names.push(name);
    }
    return names;
  },
  canGrantDirectory(): boolean {
    return typeof fsWindow.showDirectoryPicker === "function";
  },

  async openDirectory(): Promise<DirectoryRef | null> {
    if (typeof fsWindow.showDirectoryPicker !== "function") return null;
    try {
      const handle = await fsWindow.showDirectoryPicker();
      return { name: handle.name, token: handle };
    } catch {
      // AbortError when the user dismisses the picker.
      return null;
    }
  },
};
