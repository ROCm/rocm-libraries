# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every generator failure carries a TRITONGEN-* code, and both entry points print
one coded line and return 1 instead of a traceback. Needs no triton."""

import os
import shutil
import struct
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

TRITON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRITON_DIR))

import gen_descriptors  # noqa: E402
from generators import _codeobject  # noqa: E402
from generators import _contract as C  # noqa: E402
from generators import _triton_env as env  # noqa: E402
from generators import gen_rmsnorm  # noqa: E402
from generators._codeobject import (  # noqa: E402
    GeneratorError,
    describe,
    disassemble,
    hkp_pack_module,
    llvm_commit,
)

COMMITTED = env.CONTENT_DIR
OBJECT = (
    COMMITTED / C.OP / "gfx942" / C.hsaco_file(C.instances()[0].name)
).read_bytes()


def elf64(sections: dict[str, bytes]) -> bytes:
    """A minimal little-endian ELF64 holding ``sections`` and a section-name table."""
    entries = [*sections.items(), (".shstrtab", b"")]
    strtab = b"\0"
    name_offsets = []
    for name, _ in entries:
        name_offsets.append(len(strtab))
        strtab += name.encode() + b"\0"
    blobs = [data for _, data in entries[:-1]] + [strtab]
    body, offsets = b"", []
    for blob in blobs:
        offsets.append(64 + len(body))
        body += blob
    headers = bytes(64)
    for name_offset, offset, blob in zip(name_offsets, offsets, blobs):
        headers += struct.pack(
            "<IIQQQQIIQQ", name_offset, 1, 0, 0, offset, len(blob), 0, 0, 1, 0
        )
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", header, 0x28, 64 + len(body))
    struct.pack_into("<HHH", header, 0x3A, 64, len(entries) + 1, len(entries))
    return bytes(header) + body + headers


def run_main(main, argv) -> tuple[int, str]:
    err = StringIO()
    with redirect_stdout(StringIO()), redirect_stderr(err):
        code = main(argv)
    return code, err.getvalue()


class CodedRaiseSiteTest(unittest.TestCase):
    def assert_code(self, code: str, call, *args, **kwargs):
        with self.assertRaises(GeneratorError) as raised:
            call(*args, **kwargs)
        self.assertTrue(
            str(raised.exception).startswith(f"{code}: "), str(raised.exception)
        )
        return str(raised.exception)

    def test_triton_not_importable_is_env(self):
        with mock.patch.dict(sys.modules, {"triton": None}), mock.patch.dict(
            os.environ
        ), mock.patch.object(env, "_triton", None):
            message = self.assert_code(C.ERR_ENV, env.import_triton)
        self.assertIn(sys.executable, message)

    def test_missing_packaging_tree_is_env(self):
        with mock.patch.object(_codeobject, "PACKAGING_DIR", TRITON_DIR / "absent"):
            self.assert_code(C.ERR_ENV, hkp_pack_module)

    def test_missing_msgpack_is_env(self):
        with mock.patch.dict(sys.modules, {"msgpack": None}):
            self.assert_code(C.ERR_ENV, describe, OBJECT, "t")

    def test_malformed_object_is_abi(self):
        self.assert_code(C.ERR_ABI, describe, b"not an object", "t")

    def test_missing_symbol_is_abi(self):
        self.assert_code(C.ERR_ABI, describe, OBJECT, "t", "other_kernel")

    def test_kernel_count_is_abi(self):
        hkp = hkp_pack_module()
        with mock.patch.object(
            hkp, "_metadata_document", return_value={"amdhsa.kernels": []}
        ):
            self.assert_code(C.ERR_ABI, describe, OBJECT, "t")

    def test_not_elf64_is_abi(self):
        self.assert_code(C.ERR_ABI, llvm_commit, b"\x7fELF\x01\x01" + bytes(58), "t")

    def test_no_comment_section_is_env(self):
        self.assert_code(C.ERR_ENV, llvm_commit, elf64({}), "t")

    def test_no_llvm_commit_is_env(self):
        blob = elf64({".comment": b"clang version 20.0.0\0"})
        self.assert_code(C.ERR_ENV, llvm_commit, blob, "t")

    def test_failing_objdump_is_env(self):
        false = shutil.which("false")
        with mock.patch.object(_codeobject, "objdump_path", return_value=false):
            self.assert_code(C.ERR_ENV, disassemble, OBJECT, "gfx942", "t")

    def test_triton_compile_failure_is_compile(self):
        def fail(*args, **kwargs):
            raise RuntimeError("injected")

        triton = types.SimpleNamespace(compile=fail)
        backends_compiler = types.ModuleType("triton.backends.compiler")
        backends_compiler.GPUTarget = mock.Mock()
        compiler = types.ModuleType("triton.compiler")
        compiler.ASTSource = mock.Mock()
        modules = {
            "triton": types.ModuleType("triton"),
            "triton.backends": types.ModuleType("triton.backends"),
            "triton.backends.compiler": backends_compiler,
            "triton.compiler": compiler,
        }
        with mock.patch.dict(sys.modules, modules), mock.patch.object(
            env, "import_triton", return_value=triton
        ), mock.patch.object(env, "new_cache_dir"):
            message = self.assert_code(
                C.ERR_COMPILE, env.compile_instance, None, {}, {}, {}, "gfx942", 4
            )
        self.assertIn("injected", message)

    def test_mixed_llvm_commits_are_env(self):
        def compiled(fn, arch, instance):
            return env.Compiled(OBJECT, C.KERNEL_SYMBOL, 1, 4, 64, 0, 0)

        commits = iter(["a" * 40, "b" * 40] * len(C.instances()))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ
        ), mock.patch.object(env, "_prepared", False), mock.patch.object(
            env, "import_triton"
        ), mock.patch.object(
            env, "load_kernel"
        ), mock.patch.object(
            env, "assert_options_match_wheel_defaults"
        ), mock.patch.object(
            gen_rmsnorm, "compile_rmsnorm", side_effect=compiled
        ), mock.patch.object(
            gen_rmsnorm, "verify_compiled", return_value=describe(OBJECT, "t")
        ), mock.patch.object(
            gen_rmsnorm, "llvm_commit", side_effect=lambda *a: next(commits)
        ):
            code, err = run_main(
                gen_rmsnorm.main, ["--arch", "gfx942", "--out", str(Path(tmp) / "o")]
            )
        self.assertEqual(code, 1)
        self.assertTrue(err.startswith(f"error: {C.ERR_ENV}: gfx942: "), err)


class EmitInputTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.root = self.tmp / "root"
        shutil.copytree(COMMITTED, self.root)

    def assert_input_refused(self, *fragments: str):
        with self.assertRaises(GeneratorError) as raised, redirect_stdout(StringIO()):
            gen_descriptors.emit(self.root, self.root, ["gfx942"])
        message = str(raised.exception)
        self.assertTrue(message.startswith(f"{C.ERR_INPUT}: "), message)
        for fragment in fragments:
            self.assertIn(fragment, message)

    def test_missing_source_md_is_input(self):
        (self.root / C.OP / "gfx942.SOURCE.md").unlink()
        self.assert_input_refused("gfx942.SOURCE.md")

    def test_invalid_staged_tree_is_input_and_output_untouched(self):
        planted = self.root / C.OP / "gfx950" / "bad.ukd.json"
        planted.write_text("{}")
        self.assert_input_refused("descriptor validation failed")
        planted.unlink()
        files = {
            root: {
                p.relative_to(root): p.read_bytes()
                for p in root.rglob("*")
                if p.is_file()
            }
            for root in (self.root, COMMITTED)
        }
        self.assertEqual(files[self.root], files[COMMITTED])


class EntryPointTest(unittest.TestCase):
    def test_gen_rmsnorm_main_codes_an_unexpected_exception(self):
        with mock.patch.object(gen_rmsnorm, "generate", side_effect=ValueError("x")):
            code, err = run_main(gen_rmsnorm.main, ["--arch", "gfx942"])
        self.assertEqual(code, 1)
        self.assertEqual(err, f"error: {C.ERR_INTERNAL}: ValueError: x\n")

    def test_gen_descriptors_main_codes_an_unexpected_exception(self):
        with mock.patch.object(gen_descriptors, "emit", side_effect=KeyError("k")):
            code, err = run_main(gen_descriptors.main, [])
        self.assertEqual(code, 1)
        self.assertEqual(err, f"error: {C.ERR_INTERNAL}: KeyError: 'k'\n")

    def test_gen_descriptors_main_missing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, err = run_main(
                gen_descriptors.main,
                ["--in", tmp, "--out", str(Path(tmp) / "o"), "--arch", "gfx942"],
            )
        self.assertEqual(code, 1)
        self.assertTrue(err.startswith(f"error: {C.ERR_INPUT}: "), err)
        self.assertIn(C.MANIFEST_FILE, err)
        self.assertEqual(err.count("\n"), 1, err)

    def test_gen_descriptors_main_missing_packaging_tree(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            _codeobject, "PACKAGING_DIR", Path(tmp) / "absent"
        ):
            code, err = run_main(
                gen_descriptors.main,
                ["--in", str(COMMITTED), "--out", str(Path(tmp) / "o")],
            )
        self.assertEqual(code, 1)
        self.assertTrue(err.startswith(f"error: {C.ERR_ENV}: "), err)
        self.assertEqual(err.count("\n"), 1, err)


if __name__ == "__main__":
    unittest.main()
