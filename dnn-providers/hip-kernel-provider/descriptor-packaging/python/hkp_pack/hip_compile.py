import subprocess
from pathlib import Path

from .errors import HkpPackError
from .variant import variant_key


def _hipcc_command(hipcc, source_path, arch, build, out_co):
    # -fuse-cuid=none: hipcc defaults to a random compilation unit id, which
    # perturbs the __hip_cuid_ symbol so identical inputs produce different .co
    # bytes. Pinning it keeps the sha256/provenance stamped on each UKD a stable
    # traceability record across builds. In the fixed prefix so build flags
    # cannot displace it.
    cmd = [hipcc, "--genco", f"--offload-arch={arch}", "-fuse-cuid=none"]
    for name, val in (build.get("defines") or {}).items():
        if isinstance(val, bool):
            val = "1" if val else "0"
        cmd.append(f"-D{name}={val}")
    cmd += list(build.get("flags") or [])
    cmd += [str(source_path), "-o", str(out_co)]
    return cmd


def compile_hip_variant(hipcc, source_root, source, build, arch, out_dir):
    """Compile one (source, build) variant for one arch into out_dir.

    Returns the produced .co Path (named <variant_key>.co). Missing source ->
    'source not found'; a non-zero hipcc -> 'compile failed'. Both are hard
    errors, never skips.
    """
    source_path = Path(source_root) / source
    if not source_path.is_file():
        raise HkpPackError(f"source not found: {source} (looked in {source_root})")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_co = out_dir / f"{variant_key(source, build)}.co"

    cmd = _hipcc_command(hipcc, source_path, arch, build, out_co)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_co.is_file():
        raise HkpPackError(
            f"compile failed for {source} @ {arch} (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return out_co
