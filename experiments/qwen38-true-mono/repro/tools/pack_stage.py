#!/usr/bin/env python3
"""Pack a tree into a q38 stage tar the way hackathon-prep/tools/sync_pack.py does (same exclusions, CRLF->LF for
text files, mode 755 for *.sh / shebang files) and write the '<sha256>  <relpath>' manifest of the packed bytes.
Differences from sync_pack.py: Windows long paths work (\\\\?\\ prefix), and the tar is written to a file.

  pack_stage.py <src dir> <out.tar> <out.manifest>
"""
import fnmatch
import hashlib
import io
import os
import sys
import tarfile

EXC_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules"}
EXC_DIR_GLOBS = ["build", "build-*", "build_*", ".venv*", "venv*", "*.egg-info"]
EXC_FILE_GLOBS = ["*.pyc", "*.pyo", "*.o", "*.a", "*.so", "*.so.*", "*.obj", "*.dll", "*.lib", "*.exe", "*.pyd"]
TEXT_EXT = {".py", ".sh", ".txt", ".md", ".yaml", ".yml", ".json", ".cmake", ".cpp", ".cc", ".c", ".h", ".hpp", ".hh",
            ".cu", ".hip", ".in", ".toml", ".cfg", ".ini", ".s", ".ll", ".mlir", ".td", ".fbs", ".proto", ".rst", ".csv",
            ".cl"}
TEXT_NAMES = {"CMakeLists.txt", "Makefile", "Dockerfile", "LICENSE"}


def _long(path):
    if os.name == "nt":
        path = os.path.abspath(path)
        if not path.startswith(r"\\?" + "\\"):
            path = r"\\?" + "\\" + path
    return path


def main():
    src, out, man = sys.argv[1:4]
    root = _long(src)
    lines = []
    with tarfile.open(out, "w", format=tarfile.GNU_FORMAT) as tf:
        for d, dirs, files in os.walk(root):
            rel_d = os.path.relpath(d, root).replace("\\", "/")
            dirs[:] = sorted(x for x in dirs if x not in EXC_DIRS and not any(fnmatch.fnmatch(x, g) for g in EXC_DIR_GLOBS))
            for f in sorted(files):
                if any(fnmatch.fnmatch(f, g) for g in EXC_FILE_GLOBS):
                    continue
                full = os.path.join(d, f)
                rel = f if rel_d == "." else rel_d + "/" + f
                with open(full, "rb") as fh:
                    data = fh.read()
                if (os.path.splitext(f)[1].lower() in TEXT_EXT or f in TEXT_NAMES) and b"\r\n" in data:
                    data = data.replace(b"\r\n", b"\n")
                ti = tarfile.TarInfo(rel)
                ti.size = len(data)
                ti.mode = 0o755 if (data.startswith(b"#!") or rel.endswith(".sh")) else 0o644
                ti.mtime = int(os.path.getmtime(full))
                tf.addfile(ti, io.BytesIO(data))
                lines.append(hashlib.sha256(data).hexdigest() + "  " + rel)
    lines.sort(key=lambda l: l.split("  ", 1)[1])
    with open(man, "w", newline="\n") as m:
        m.write("\n".join(lines) + "\n")
    print(f"packed {len(lines)} files -> {out}")


if __name__ == "__main__":
    main()
