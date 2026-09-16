// Reading a file a benchmark report points at. A report is untrusted input, so
// this module is the trust boundary between the paths it names and the
// filesystem: nothing downstream re-checks containment.

const fs = require("node:fs/promises");
const path = require("node:path");

/** Resolves `relativePath` against `baseDir` and refuses to leave it. */
function resolveAndContain(baseDir, relativePath) {
  if (path.isAbsolute(relativePath)) {
    throw new Error(`Path must be relative: ${relativePath}`);
  }
  const resolved = path.resolve(baseDir, relativePath);
  const fromBase = path.relative(baseDir, resolved);
  // A sibling directory shares a prefix with the base, so compare path steps,
  // never strings. `path.isAbsolute` catches a different Windows drive.
  if (fromBase.startsWith("..") || path.isAbsolute(fromBase)) {
    throw new Error(`Path escapes base directory: ${relativePath}`);
  }
  return resolved;
}

/**
 * `base` is `{ kind: "file" | "directory", path }`: a report resolves its
 * paths against its own directory, a granted folder against itself.
 */
async function readRelated(base, relativePath) {
  const baseDir = base.kind === "directory" ? base.path : path.dirname(base.path);
  // A Buffer is a Uint8Array subclass; structured cloning over IPC keeps only
  // the plain view, so hand back exactly what the renderer's contract promises.
  return new Uint8Array(await fs.readFile(resolveAndContain(baseDir, relativePath)));
}

module.exports = { resolveAndContain, readRelated };
