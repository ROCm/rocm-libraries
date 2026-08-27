import subprocess
from pathlib import Path

from .errors import HkpPackError
from .variant import _hash_payload


def hip_source_relpath(rel_dir, source):
    """The compiled input's identity: its normalized path relative to the root.

    `source` alone is not an identity. Two descriptors in different child
    folders may both name `PointwiseAdd.cpp` and mean different files; keying on
    the bare name would collapse them onto one compiled .co and ship the first
    one's bytes for both. The root-relative path is what actually distinguishes
    the inputs.

    Flat layouts (`rel_dir == "."`) reduce to `source` unchanged, so their keys
    are byte-identical to the pre-nesting keys.
    """
    return (Path(rel_dir) / source).as_posix().removeprefix("./")


def hip_variant_key(rel_source, build):
    """Stable input hash over (rel_source, build) for a hip variant.

    Drives both the toc_key and the intermediate .co filename. Two hip UKDs
    sharing rel_source+build (differing entry) hash identically and share one
    compiled .co; a different build hashes apart.

    `rel_source` is the root-relative source path from hip_source_relpath, not
    the authored `source` string — see there for why the bare name is unsafe.
    """
    payload = {"source": rel_source, "build": build}
    return _hash_payload(Path(rel_source).stem, payload)


def _hipcc_command(hipcc, source_path, arch, build, out_co):
    cmd = [hipcc, "--genco", f"--offload-arch={arch}"]
    for name, val in (build.get("defines") or {}).items():
        if isinstance(val, bool):
            val = "1" if val else "0"
        cmd.append(f"-D{name}={val}")
    cmd += list(build.get("flags") or [])
    # Pin the compilation-unit id: hipcc otherwise defaults it to random, which
    # perturbs the __hip_cuid_ symbol so identical inputs emit different .co bytes
    # and an unstable sha256/provenance stamp. Appended after the authored build
    # flags so clang's last-flag-wins keeps this value; authored -fuse-cuid flags
    # are rejected in validation as well.
    cmd.append("-fuse-cuid=none")
    cmd += [str(source_path), "-o", str(out_co)]
    return cmd


def compile_hip_variant(hipcc, source_root, rel_dir, source, build, arch, out_dir):
    """Compile one (source, build) variant for one arch into out_dir.

    Resolves `source` **relative to the descriptor that named it** —
    `source_root / rel_dir / source` — and names the .co after hip_variant_key.

    Resolution is descriptor-relative only, with no root-relative fallback. A
    fallback would fire exactly when the descriptor-local file is missing, so a
    typo in `source` would stop being an error and instead bind silently to a
    same-named file elsewhere in the tree. Sharing one .cpp between sibling
    folders stays expressible by saying so: `"../shared/Kernel.cpp"`. The
    resolved path must stay inside the root.

    Missing source -> 'source not found'; a non-zero hipcc -> 'compile failed'.
    Both are hard errors, never skips.
    """
    root = Path(source_root).resolve()
    source_path = (root / rel_dir / source).resolve()
    if not source_path.is_relative_to(root):
        raise HkpPackError(
            f"source escapes the source root: {source} "
            f"(from {Path(rel_dir).as_posix()}, resolved to {source_path})"
        )
    if not source_path.is_file():
        raise HkpPackError(
            f"source not found: {source} (looked for {source_path}, "
            f"resolved relative to descriptor folder {Path(rel_dir).as_posix()})"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_co = (
        out_dir / f"{hip_variant_key(hip_source_relpath(rel_dir, source), build)}.co"
    )

    cmd = _hipcc_command(hipcc, source_path, arch, build, out_co)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_co.is_file():
        raise HkpPackError(
            f"compile failed for {source} @ {arch} (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return out_co
