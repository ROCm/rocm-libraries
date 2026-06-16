################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Artifact compression and extraction helpers.

These utilities are pytest-independent and can be used from standalone scripts.
"""

import os
import tarfile
import warnings


_TESTS_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


def artifact_name_for_config(config_path):
    """Derive a deterministic artifact name from a YAML config path.

    Stable across machines because the relative path under Tensile/Tests/
    is identical in every checkout. E.g. common/gemm/fp16_tn.yaml ->
    common__gemm__fp16_tn.
    """
    relpath = os.path.relpath(config_path, _TESTS_ROOT_DIR)
    name = os.path.splitext(relpath)[0]
    return name.replace("/", "__").replace("\\", "__")


def _is_use_cache_file(relpath):
    """Return True if relpath (relative to output_dir) is needed for --use-cache.

    Whitelist:
    - caches/<hash>/cache.yaml
    - caches/<hash>/source/library/* (.co, .hsaco, .yaml)
    - Data/*.yaml

    Everything else is build-only intermediate output.
    """
    parts = relpath.replace(os.sep, "/").split("/")

    if parts[-1] == "cache.yaml" and "caches" in parts:
        return True

    try:
        source_idx = parts.index("source")
        if (source_idx + 1 < len(parts)
                and parts[source_idx + 1] == "library"
                and "caches" in parts):
            return True
    except ValueError:
        pass

    if len(parts) >= 2 and parts[-2] == "Data" and parts[-1].endswith(".yaml"):
        return True

    return False


def compress_output(output_dir, dest_dir=None, name=None):
    """Compress only the files needed for --use-cache into a .tar.gz artifact.

    Excludes build-only intermediates (assembly .s/.o files, headers, build_tmp)
    which typically account for ~88% of the output directory size.

    Args:
        output_dir: The directory to compress.
        dest_dir: Where to write the artifact. Defaults to the parent of output_dir.
        name: Base name for the artifact file (without extension).
            Defaults to the basename of output_dir.

    Returns:
        Path to the created .tar.gz artifact.
    """
    if dest_dir is None:
        dest_dir = os.path.dirname(output_dir)
    if name is None:
        name = os.path.basename(output_dir)

    os.makedirs(dest_dir, exist_ok=True)
    artifact_path = os.path.join(dest_dir, name + ".tar.gz")
    file_count = 0
    with tarfile.open(artifact_path, "w:gz") as tar:
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                full_path = os.path.join(root, f)
                relpath = os.path.relpath(full_path, output_dir)
                if _is_use_cache_file(relpath):
                    arcname = os.path.join(name, relpath)
                    tar.add(full_path, arcname=arcname)
                    file_count += 1

    if file_count == 0:
        warnings.warn(
            f"compress_output produced an empty archive for '{output_dir}'. "
            f"No files matched the use-cache whitelist.",
            stacklevel=2,
        )

    return artifact_path


def _safe_extractall(tar, dest):
    """Extract with tarfile's data_filter when available, manual validation otherwise.

    Defense-in-depth — archives are self-generated.
    """
    if hasattr(tarfile, "data_filter"):
        tar.extractall(path=dest, filter="data")
    else:
        dest_real = os.path.realpath(dest)
        validated_members = []
        for member in tar.getmembers():
            member_path = os.path.realpath(os.path.join(dest, member.name))
            if not member_path.startswith(dest_real + os.sep) and member_path != dest_real:
                raise ValueError(
                    f"Path traversal detected in tar member: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(
                    f"Symlinks/hardlinks not permitted in artifacts: {member.name}")
            validated_members.append(member)
        tar.extractall(path=dest, members=validated_members)


def extract_artifact(artifact_path, dest_dir):
    """Extract a .tar.gz artifact whose single top-level directory becomes dest_dir."""
    parent = os.path.dirname(dest_dir)
    os.makedirs(parent, exist_ok=True)
    with tarfile.open(artifact_path, "r:gz") as tar:
        _safe_extractall(tar, parent)
    if not os.path.isdir(dest_dir):
        actual = os.listdir(parent)
        raise FileNotFoundError(
            f"Expected directory '{os.path.basename(dest_dir)}' after "
            f"extracting '{artifact_path}', but found: {actual}")

    for dirpath, _, filenames in os.walk(dest_dir):
        for f in filenames:
            relpath = os.path.relpath(os.path.join(dirpath, f), dest_dir)
            if not _is_use_cache_file(relpath):
                raise ValueError(
                    f"Unexpected file in artifact '{artifact_path}': {relpath}")

    code_object_exts = (".co", ".hsaco")
    has_code_objects = any(
        f.endswith(code_object_exts)
        for dirpath, _, filenames in os.walk(dest_dir)
        for f in filenames
    )
    if not has_code_objects:
        warnings.warn(
            f"Extracted artifact '{artifact_path}' contains no code object "
            f"files ({', '.join(code_object_exts)}). The _is_use_cache_file "
            f"whitelist may be out of sync with the Tensile output layout.",
            stacklevel=2,
        )

    return dest_dir
