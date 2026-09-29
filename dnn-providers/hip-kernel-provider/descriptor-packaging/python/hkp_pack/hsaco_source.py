"""A prebuilt code object named by a descriptor: `kernel_source` kind `hsaco`.

No producer runs. The authored file is resolved like a hip source, keyed on its
normalized root-relative path, and packed byte-for-byte at pack time.
"""

import posixpath
from pathlib import Path

from .hip_compile import resolve_descriptor_file
from .variant import _hash_payload


def hsaco_file_relpath(rel_dir, file):
    """The authored code object's identity: its normalized root-relative path.

    Lexical rather than resolved, so the key depends on the authored tree and
    not on where a symlink happens to point. `a/../b/K.co` and `b/K.co` name one
    file and share one identity.
    """
    return posixpath.normpath((Path(rel_dir) / file).as_posix())


def hsaco_variant_key(rel_file):
    """Stable input hash over the root-relative file path for an hsaco variant.

    The toc_key the authored bytes pack under. One file serving several symbols
    hashes once, so its bytes enter the archive once.
    """
    return _hash_payload(Path(rel_file).stem, {"file": rel_file})


def resolve_hsaco_file(source_root, rel_dir, file, where):
    """The authored code object on disk, via the hip resolver.

    Relative to the descriptor that named it, contained in the root, with no
    fallback. In-root symlinks are accepted.
    """
    return resolve_descriptor_file(source_root, rel_dir, file, "hsaco file", where)
