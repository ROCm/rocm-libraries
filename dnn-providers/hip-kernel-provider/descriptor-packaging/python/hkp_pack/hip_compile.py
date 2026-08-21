import subprocess
from pathlib import Path

from .errors import HkpPackError
from .variant import _hash_payload


def hip_variant_key(source, build, origin_index=0):
    """Stable input hash over (source, build, origin_index) for a hip variant.

    Drives both the toc_key and the intermediate .co filename. Two hip UKDs
    sharing source+build+origin_index (differing entry) hash identically and
    share one compiled .co; a different build hashes apart. origin_index is a
    positional discriminator ("root{n}") for the source root the UKD loaded
    from, so two roots carrying an identically named source at the same relative
    path but different bytes get distinct keys. It is positional, not
    filesystem-derived: reordering paths provided with --source-root changes keys.
    """
    payload = {"source": source, "build": build, "origin_index": f"root{origin_index}"}
    return _hash_payload(Path(source).stem, payload)


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


def compile_hip_variant(
    hipcc, source_root, source, build, arch, out_dir, key_origin_index=0
):
    """Compile one (source, build) variant for one arch into out_dir.

    Resolves the source against source_root and names the .co after
    hip_variant_key. key_origin_index is folded into that key so the .co
    filename matches the toc_key the pipeline computes for the same UKD. Missing
    source -> 'source not found'; a non-zero hipcc -> 'compile failed'. Both are
    hard errors, never skips.
    """
    source_path = Path(source_root) / source
    if not source_path.is_file():
        raise HkpPackError(f"source not found: {source} (looked in {source_root})")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_co = out_dir / f"{hip_variant_key(source, build, key_origin_index)}.co"

    cmd = _hipcc_command(hipcc, source_path, arch, build, out_co)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_co.is_file():
        raise HkpPackError(
            f"compile failed for {source} @ {arch} (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return out_co
