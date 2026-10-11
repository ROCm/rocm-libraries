# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Launch geometry and manifest records are checked against the contract and the
object, the one-command regeneration publishes each arch whole or not at all, and the
committed objects disassemble. Needs no triton: compilation is stubbed with the
committed objects and their manifest metadata."""

import dataclasses
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

TRITON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRITON_DIR))

import gen_descriptors  # noqa: E402
from generators import _contract as C  # noqa: E402
from generators import _triton_env as env  # noqa: E402
from generators import gen_rmsnorm  # noqa: E402
from generators._codeobject import (  # noqa: E402
    GeneratorError,
    disassemble,
    isa_counts,
)
from generators._manifest import read_manifest  # noqa: E402

COMMITTED = env.CONTENT_DIR


def committed_record(arch: str, name: str) -> dict:
    manifest = read_manifest(COMMITTED / C.OP / arch / C.MANIFEST_FILE)
    return next(r for r in manifest["instances"] if r["name"] == name)


def committed_compiled(arch: str, instance: C.Instance) -> env.Compiled:
    """The committed object for ``instance`` as ``compile_instance`` would return it."""
    record = committed_record(arch, instance.name)
    return env.Compiled(
        hsaco=(COMMITTED / C.OP / arch / record["file"]).read_bytes(),
        name=record["symbol"],
        shared=record["shared"],
        num_warps=record["num_warps"],
        warp_size=record["warp_size"],
        global_scratch_size=record["global_scratch_size"],
        profile_scratch_size=record["profile_scratch_size"],
    )


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def leftover_stages(root: Path) -> list[str]:
    return [
        str(path)
        for parent in (root, root / C.OP)
        for path in parent.glob(".tritongen-*")
    ]


INSTANCE_BN4096 = next(
    i for i in C.instances() if i.dtype == "bf16" and i.block_n == 4096
)


class VerifyCompiledGeometryTest(unittest.TestCase):
    def assert_abi_refused(self, compiled, instance):
        with self.assertRaises(GeneratorError) as raised:
            gen_rmsnorm.verify_compiled(compiled, instance, "gfx942")
        message = str(raised.exception)
        self.assertTrue(message.startswith(C.ERR_ABI), message)
        self.assertIn(instance.name, message)

    def test_committed_object_accepted(self):
        compiled = committed_compiled("gfx942", INSTANCE_BN4096)
        described = gen_rmsnorm.verify_compiled(compiled, INSTANCE_BN4096, "gfx942")
        self.assertEqual(
            described["max_flat_workgroup_size"], INSTANCE_BN4096.block_threads
        )

    def test_num_warps_differing_from_the_object_refused(self):
        compiled = dataclasses.replace(
            committed_compiled("gfx942", INSTANCE_BN4096), num_warps=4
        )
        self.assert_abi_refused(compiled, INSTANCE_BN4096)

    def test_block_threads_differing_from_num_warps_refused(self):
        config = C.BlockConfig(4096, 8, 256, 200)
        mutant = dataclasses.replace(
            INSTANCE_BN4096,
            num_warps=config.num_warps,
            block_threads=config.block_threads,
            priority=config.priority,
        )
        self.assert_abi_refused(committed_compiled("gfx942", INSTANCE_BN4096), mutant)

    def test_launch_geometry_differing_from_the_object_refused(self):
        # Contract and num_warps*warp_size agree on 256; only the object says 512.
        mutant = dataclasses.replace(INSTANCE_BN4096, num_warps=4, block_threads=256)
        compiled = dataclasses.replace(
            committed_compiled("gfx942", INSTANCE_BN4096), num_warps=4
        )
        with self.assertRaises(GeneratorError) as raised:
            gen_rmsnorm.verify_compiled(compiled, mutant, "gfx942")
        message = str(raised.exception)
        self.assertTrue(message.startswith(C.ERR_ABI), message)
        self.assertIn(".max_flat_workgroup_size=512", message)


class CollectDriftTest(unittest.TestCase):
    ARCH = "gfx942"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "root"
        shutil.copytree(COMMITTED, self.root)
        self.manifest_path = self.root / C.OP / self.ARCH / C.MANIFEST_FILE

    def mutate(self, change):
        manifest = read_manifest(self.manifest_path)
        change(manifest)
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def assert_refused(self, *fragments: str):
        with self.assertRaises(GeneratorError) as raised:
            gen_descriptors.collect(self.root, self.ARCH)
        message = str(raised.exception)
        self.assertTrue(message.startswith(C.ERR_INPUT), message)
        for fragment in fragments:
            self.assertIn(fragment, message)

    def first_record(self, manifest) -> dict:
        return manifest["instances"][0]

    def test_committed_manifest_accepted(self):
        records = gen_descriptors.collect(self.root, self.ARCH)
        self.assertEqual({r["name"] for r in records}, {i.name for i in C.instances()})

    def test_dtype_drift_refused(self):
        self.mutate(lambda m: self.first_record(m).update(dtype="f16"))
        self.assert_refused("rmsnorm_bf16_bn1024", "dtype='f16'", "dtype='bf16'")

    def test_block_n_drift_refused(self):
        self.mutate(lambda m: self.first_record(m).update(block_n=2048))
        self.assert_refused("rmsnorm_bf16_bn1024", "block_n=2048", "block_n=1024")

    def test_block_drift_refused(self):
        self.mutate(lambda m: self.first_record(m).update(block=[512, 1, 1]))
        self.assert_refused("rmsnorm_bf16_bn1024", "block[0]=512", "=256")

    def test_zero_shared_refused(self):
        self.mutate(lambda m: self.first_record(m).update(shared=0))
        self.assert_refused("rmsnorm_bf16_bn1024", "shared=0")

    def test_arch_mismatch_refused(self):
        self.mutate(lambda m: m.update(arch="gfx950"))
        self.assert_refused("gfx950/rmsnorm")

    def test_op_mismatch_refused(self):
        self.mutate(lambda m: m.update(op="layernorm"))
        self.assert_refused("layernorm")

    def test_missing_instance_refused(self):
        self.mutate(lambda m: m["instances"].pop())
        self.assert_refused("instance names")

    def test_priority_drift_refused(self):
        self.mutate(lambda m: self.first_record(m).update(priority=1))
        self.assert_refused("rmsnorm_bf16_bn1024", "priority=1", "priority=400")

    def test_block_differing_from_the_object_refused(self):
        # Manifest and contract agree; only the object's workgroup size differs.
        real_describe = gen_descriptors.describe

        def describe(*args, **kwargs):
            described = dict(real_describe(*args, **kwargs))
            described["max_flat_workgroup_size"] *= 2
            return described

        with mock.patch.object(gen_descriptors, "describe", describe):
            self.assert_refused("rmsnorm_bf16_bn1024", ".max_flat_workgroup_size=512")


class OneCommandRegenerationTest(unittest.TestCase):
    """``gen_rmsnorm.main`` publishes objects and descriptors one arch at a time."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "root"
        shutil.copytree(COMMITTED, self.root)
        self.committed = tree(COMMITTED)
        environ = mock.patch.dict(os.environ)
        environ.start()
        self.addCleanup(environ.stop)
        os.environ.pop("TRITON_F32_DEFAULT", None)
        for target, value in (
            ("_prepared", False),
            ("import_triton", mock.Mock()),
            ("load_kernel", mock.Mock()),
            ("assert_options_match_wheel_defaults", mock.Mock()),
        ):
            patcher = mock.patch.object(env, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def drop_descriptors(self, arch: str) -> None:
        for path in (self.root / C.OP / arch).glob("*.json"):
            if path.name != C.MANIFEST_FILE:
                path.unlink()

    def run_main(self, compile_stub) -> tuple[int, str]:
        err = StringIO()
        with mock.patch.object(
            gen_rmsnorm, "compile_rmsnorm", side_effect=compile_stub
        ), redirect_stdout(StringIO()), redirect_stderr(err):
            code = gen_rmsnorm.main(
                ["--arch", "gfx942", "--arch", "gfx950", "--out", str(self.root)]
            )
        return code, err.getvalue()

    def subtree(self, snapshot: dict[str, bytes], arch: str) -> dict[str, bytes]:
        prefix = f"{C.OP}/{arch}"
        return {k: v for k, v in snapshot.items() if k.startswith(prefix)}

    def test_regenerates_objects_and_descriptors(self):
        for path in (self.root / C.OP).glob("*.json"):
            path.unlink()
        self.drop_descriptors("gfx942")
        self.drop_descriptors("gfx950")
        code, err = self.run_main(
            lambda fn, arch, instance: committed_compiled(arch, instance)
        )
        self.assertEqual(code, 0, err)
        self.assertEqual(tree(self.root), self.committed)
        self.assertEqual(leftover_stages(self.root), [])

    def test_compile_failure_keeps_earlier_arches_and_leaves_the_failing_one(self):
        self.drop_descriptors("gfx942")
        gfx950 = self.root / C.OP / "gfx950"
        inode = gfx950.stat().st_ino

        def stub(fn, arch, instance):
            if arch == "gfx950":
                raise GeneratorError(f"{C.ERR_COMPILE}: {arch}: injected")
            return committed_compiled(arch, instance)

        code, err = self.run_main(stub)
        self.assertEqual(code, 1)
        self.assertTrue(err.startswith(f"error: {C.ERR_COMPILE}: "), err)
        after = tree(self.root)
        self.assertEqual(
            self.subtree(after, "gfx942"), self.subtree(self.committed, "gfx942")
        )
        self.assertEqual(
            self.subtree(after, "gfx950"), self.subtree(self.committed, "gfx950")
        )
        self.assertEqual(gfx950.stat().st_ino, inode)
        self.assertEqual(leftover_stages(self.root), [])

    def test_descriptor_step_failure_leaves_the_arch_as_it_was(self):
        victim = C.instances()[-1].name

        def stub(fn, arch, instance):
            compiled = committed_compiled(arch, instance)
            if arch == "gfx950" and instance.name == victim:
                return dataclasses.replace(compiled, shared=0)
            return compiled

        code, err = self.run_main(stub)
        self.assertEqual(code, 1)
        self.assertTrue(err.startswith(f"error: {C.ERR_INPUT}: "), err)
        self.assertIn(victim, err)
        self.assertEqual(tree(self.root), self.committed)
        self.assertEqual(leftover_stages(self.root), [])


class CommittedObjectDisassemblyTest(unittest.TestCase):
    def test_every_committed_object_disassembles_with_packed_f32(self):
        for arch in C.ARCHES:
            manifest = read_manifest(COMMITTED / C.OP / arch / C.MANIFEST_FILE)
            self.assertEqual(len(manifest["instances"]), len(C.instances()))
            for record in manifest["instances"]:
                where = f"{C.OP}/{arch}/{record['file']}"
                with self.subTest(where=where):
                    blob = (COMMITTED / C.OP / arch / record["file"]).read_bytes()
                    text = disassemble(blob, arch, where)
                    self.assertTrue(text)
                    self.assertIn(C.KERNEL_SYMBOL, text)
                    self.assertGreater(isa_counts(text)[1], 0)


if __name__ == "__main__":
    unittest.main()
