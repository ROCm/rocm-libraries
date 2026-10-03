"""A prebuilt code object named by a descriptor: `kernel_source` kind `hsaco`.

No producer runs. The authored file is resolved like a hip source, keyed on its
resolved root-relative path, and packed byte-for-byte at pack time.
"""

from pathlib import Path

from .hip_compile import resolve_descriptor_file
from .variant import _hash_payload


def hsaco_file_identity(root, path):
    """The authored code object's identity: its resolved path relative to the root.

    `root` and `path` are both resolved, so every spelling of one file (absolute,
    `..`-re-entering, through an in-root symlink, or via a symlinked source root)
    names one identity, and the identity does not depend on where the root lives.
    """
    return path.relative_to(root).as_posix()


def hsaco_variant_key(rel_file):
    """Stable input hash over the root-relative file path for an hsaco variant.

    The toc_key the authored bytes pack under. One file serving several symbols
    hashes once, so its bytes enter the archive once.
    """
    return _hash_payload(Path(rel_file).stem, {"file": rel_file})


def resolve_hsaco_file(source_root, rel_dir, file, where):
    """The authored code object on disk, via the hip resolver.

    Relative to the descriptor that named it, contained in the root once
    symlinks are resolved, with no fallback.
    """
    return resolve_descriptor_file(source_root, rel_dir, file, "hsaco file", where)
