# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Launch geometry and manifest records are checked against the op contract and the
object, the one-command regeneration publishes each arch whole or not at all, and the
committed objects disassemble. Needs no triton: compilation is stubbed with the
committed objects and their manifest metadata. Every case runs for every op in OPS."""

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
from tritongen import _family as F  # noqa: E402
from tritongen import _triton_env as env  # noqa: E402
from tritongen import gen  # noqa: E402
from tritongen._codeobject import disassemble, isa_counts  # noqa: E402
from tritongen._family import GeneratorError  # noqa: E402
from tritongen._manifest import read_manifest  # noqa: E402
from tritongen.ops import OPS  # noqa: E402

COMMITTED = F.CONTENT_DIR


def committed_manifest(op, arch: str) -> dict:
    return read_manifest(COMMITTED / op.OP / arch / F.MANIFEST_FILE)


def committed_record(op, arch: str, name: str) -> dict:
    return next(
        r for r in committed_manifest(op, arch)["instances"] if r["name"] == name
    )


def committed_compiled(op, arch: str, instance) -> env.Compiled:
    """The committed object for ``instance`` as ``compile_instance`` would return it."""
    record = committed_record(op, arch, instance.name)
    return env.Compiled(
        hsaco=(COMMITTED / op.OP / arch / record["file"]).read_bytes(),
        name=record["symbol"],
        dynamic_lds_bytes=record["dynamic_lds_bytes"],
        num_warps=record["knobs"]["block_threads"] // F.WARP_SIZE,
        warp_size=F.WARP_SIZE,
        global_scratch_size=record["global_scratch_size"],
        profile_scratch_size=record["profile_scratch_size"],
    )


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def leftover_stages(op, root: Path) -> list[str]:
    return [
        str(path)
        for parent in (root, root / op.OP)
        for path in parent.glob(".tritongen-*")
    ]


def widest(op):
    """The instance with the most threads (halving it keeps a whole number of warps)."""
    return max(op.instances(), key=lambda i: i.knobs["block_threads"])


class VerifyCompiledGeometryTest(unittest.TestCase):
    def assert_abi_refused(self, op, compiled, instance, *fragments):
        arch = op.ARCHES[0]
        with self.assertRaises(GeneratorError) as raised:
            gen.verify_compiled(op, compiled, instance, arch)
        message = str(raised.exception)
        self.assertTrue(message.startswith(F.ERR_ABI), message)
        self.assertIn(instance.name, message)
        for fragment in fragments:
            self.assertIn(fragment, message)

    def test_committed_objects_accepted(self):
        for op in OPS.values():
            for arch in op.ARCHES:
                for instance in op.instances():
                    with self.subTest(op=op.OP, arch=arch, instance=instance.name):
                        compiled = committed_compiled(op, arch, instance)
                        described = gen.verify_compiled(op, compiled, instance, arch)
                        self.assertEqual(
                            described["max_flat_workgroup_size"],
                            instance.knobs["block_threads"],
                        )

    def test_num_warps_differing_from_the_object_refused(self):
        for op in OPS.values():
            with self.subTest(op=op.OP):
                instance = widest(op)
                compiled = committed_compiled(op, op.ARCHES[0], instance)
                compiled = dataclasses.replace(
                    compiled, num_warps=compiled.num_warps // 2
                )
                self.assert_abi_refused(op, compiled, instance)

    def test_block_threads_differing_from_num_warps_refused(self):
        for op in OPS.values():
            with self.subTest(op=op.OP):
                instance = widest(op)
                mutant = dataclasses.replace(
                    instance, block_threads=instance.block_threads // 2
                )
                compiled = committed_compiled(op, op.ARCHES[0], instance)
                self.assert_abi_refused(op, compiled, mutant)

    def test_launch_geometry_differing_from_the_object_refused(self):
        # Contract and num_warps*warp_size agree on half; only the object says whole.
        for op in OPS.values():
            with self.subTest(op=op.OP):
                instance = widest(op)
                whole = instance.block_threads
                mutant = dataclasses.replace(instance, block_threads=whole // 2)
                compiled = committed_compiled(op, op.ARCHES[0], instance)
                compiled = dataclasses.replace(
                    compiled, num_warps=compiled.num_warps // 2
                )
                self.assert_abi_refused(
                    op, compiled, mutant, f".max_flat_workgroup_size={whole}"
                )


class CollectDriftTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "root"
        shutil.copytree(COMMITTED, self.root)

    def manifest_path(self, op) -> Path:
        return self.root / op.OP / op.ARCHES[0] / F.MANIFEST_FILE

    def mutate(self, op, change):
        manifest = read_manifest(self.manifest_path(op))
        change(manifest)
        self.manifest_path(op).write_text(json.dumps(manifest), encoding="utf-8")

    def assert_refused(self, op, *fragments: str):
        with self.assertRaises(GeneratorError) as raised:
            gen_descriptors.collect(op, self.root, op.ARCHES[0])
        message = str(raised.exception)
        self.assertTrue(message.startswith(F.ERR_INPUT), message)
        for fragment in fragments:
            self.assertIn(fragment, message)

    def each_op(self, change, *fragments):
        """Apply ``change(op, first_record)`` to each op's first manifest and expect a
        refusal naming the first instance and ``fragments(op, record)``."""
        for op in OPS.values():
            with self.subTest(op=op.OP):
                self.setUp()
                first = committed_manifest(op, op.ARCHES[0])["instances"][0]
                self.mutate(op, lambda m: change(op, m["instances"][0]))
                self.assert_refused(op, first["name"], *fragments)

    def test_committed_manifests_accepted(self):
        for op in OPS.values():
            for arch in op.ARCHES:
                with self.subTest(op=op.OP, arch=arch):
                    records = gen_descriptors.collect(op, self.root, arch)
                    self.assertEqual(
                        sorted(r["name"] for r in records),
                        sorted(i.name for i in op.instances()),
                    )

    def test_dtype_drift_refused(self):
        self.each_op(lambda op, r: r.update(dtype="x"), "dtype='x'")

    def test_knob_drift_refused(self):
        self.each_op(lambda op, r: r["knobs"].update(block_n=3), "'block_n': 3")

    def test_block_threads_drift_refused(self):
        self.each_op(
            lambda op, r: r["knobs"].update(block_threads=1),
            "'block_threads': 1",
        )

    def test_priority_drift_refused(self):
        self.each_op(lambda op, r: r.update(priority=-1), "priority=-1")

    def test_kernarg_segment_size_drift_refused(self):
        self.each_op(
            lambda op, r: r.update(kernarg_segment_size=304), "kernarg_segment_size=304"
        )

    def test_arg_record_drift_refused(self):
        self.each_op(lambda op, r: r["args"][0].update(name="renamed"), "renamed")

    def test_nonzero_scratch_refused(self):
        for op in OPS.values():
            with self.subTest(op=op.OP):
                self.setUp()
                self.mutate(
                    op, lambda m: m["instances"][0].update(global_scratch_size=64)
                )
                with self.assertRaises(GeneratorError) as raised:
                    gen_descriptors.collect(op, self.root, op.ARCHES[0])
                self.assertTrue(str(raised.exception).startswith(F.ERR_SCRATCH))

    def test_arch_mismatch_refused(self):
        for op in OPS.values():
            with self.subTest(op=op.OP):
                self.setUp()
                self.mutate(op, lambda m: m.update(arch="gfx000"))
                self.assert_refused(op, "gfx000")

    def test_op_mismatch_refused(self):
        for op in OPS.values():
            with self.subTest(op=op.OP):
                self.setUp()
                self.mutate(op, lambda m: m.update(op="otherop"))
                self.assert_refused(op, "otherop")

    def test_missing_instance_refused(self):
        for op in OPS.values():
            with self.subTest(op=op.OP):
                self.setUp()
                last = op.instances()[-1].name
                self.mutate(op, lambda m: m["instances"].pop())
                (self.root / op.OP / op.ARCHES[0] / F.hsaco_file(last)).unlink()
                self.assert_refused(op, "instance names")

    def test_workgroup_differing_from_the_object_refused(self):
        # Manifest and contract agree; only the object's workgroup size differs.
        real_describe = gen_descriptors.describe

        def describe(*args, **kwargs):
            described = dict(real_describe(*args, **kwargs))
            described["max_flat_workgroup_size"] *= 2
            return described

        for op in OPS.values():
            with self.subTest(op=op.OP), mock.patch.object(
                gen_descriptors, "describe", describe
            ):
                first = committed_manifest(op, op.ARCHES[0])["instances"][0]
                doubled = 2 * first["knobs"]["block_threads"]
                self.assert_refused(
                    op, first["name"], f".max_flat_workgroup_size={doubled}"
                )


class OneCommandRegenerationTest(unittest.TestCase):
    """``tritongen.gen.main`` publishes objects and descriptors one arch at a time."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "root"
        shutil.copytree(COMMITTED, self.root)
        self.committed = tree(COMMITTED)
        environ = mock.patch.dict(os.environ)
        environ.start()
        self.addCleanup(environ.stop)
        for target, value in (
            ("_prepared", False),
            ("import_triton", mock.Mock()),
            ("load_kernel", mock.Mock()),
            ("assert_options_match_wheel_defaults", mock.Mock()),
        ):
            patcher = mock.patch.object(env, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def drop_descriptors(self, op, arch: str) -> None:
        for path in (self.root / op.OP / arch).glob("*.json"):
            if path.name != F.MANIFEST_FILE:
                path.unlink()

    def run_main(self, op, compile_stub) -> tuple[int, str]:
        measured = committed_manifest(op, op.ARCHES[0])["toolchain"]["libtriton_sha256"]
        argv = ["--op", op.OP, "--out", str(self.root)]
        for arch in op.ARCHES:
            argv += ["--arch", arch]
        err = StringIO()
        with mock.patch.object(
            gen, "compile_one", side_effect=compile_stub
        ), mock.patch.object(
            env, "libtriton_sha256", return_value=measured
        ), redirect_stdout(
            StringIO()
        ), redirect_stderr(
            err
        ):
            code = gen.main(argv)
        return code, err.getvalue()

    def subtree(self, op, snapshot: dict[str, bytes], arch: str) -> dict[str, bytes]:
        prefix = f"{op.OP}/{arch}"
        return {k: v for k, v in snapshot.items() if k.startswith(prefix)}

    def test_regenerates_objects_and_descriptors(self):
        for op in OPS.values():
            with self.subTest(op=op.OP):
                for path in (self.root / op.OP).glob("*.json"):
                    path.unlink()
                for arch in op.ARCHES:
                    self.drop_descriptors(op, arch)
                code, err = self.run_main(
                    op,
                    lambda o, fn, arch, instance: committed_compiled(o, arch, instance),
                )
                self.assertEqual(code, 0, err)
                self.assertEqual(tree(self.root), self.committed)
                self.assertEqual(leftover_stages(op, self.root), [])

    def test_compile_failure_keeps_earlier_arches_and_leaves_the_failing_one(self):
        for op in OPS.values():
            if len(op.ARCHES) < 2:
                continue
            with self.subTest(op=op.OP):
                first, failing = op.ARCHES[0], op.ARCHES[1]
                self.drop_descriptors(op, first)
                failing_dir = self.root / op.OP / failing
                inode = failing_dir.stat().st_ino

                def stub(o, fn, arch, instance):
                    if arch == failing:
                        raise GeneratorError(f"{F.ERR_COMPILE}: {arch}: injected")
                    return committed_compiled(o, arch, instance)

                code, err = self.run_main(op, stub)
                self.assertEqual(code, 1)
                self.assertTrue(err.startswith(f"error: {F.ERR_COMPILE}: "), err)
                after = tree(self.root)
                for arch in (first, failing):
                    self.assertEqual(
                        self.subtree(op, after, arch),
                        self.subtree(op, self.committed, arch),
                    )
                self.assertEqual(failing_dir.stat().st_ino, inode)
                self.assertEqual(leftover_stages(op, self.root), [])

    def test_descriptor_step_failure_leaves_the_arch_as_it_was(self):
        for op in OPS.values():
            with self.subTest(op=op.OP):
                victim = op.instances()[-1].name
                last = op.ARCHES[-1]

                def stub(o, fn, arch, instance):
                    compiled = committed_compiled(o, arch, instance)
                    if arch == last and instance.name == victim:
                        return dataclasses.replace(compiled, dynamic_lds_bytes=0)
                    return compiled

                code, err = self.run_main(op, stub)
                self.assertEqual(code, 1)
                self.assertTrue(err.startswith(f"error: {F.ERR_INPUT}: "), err)
                self.assertIn(victim, err)
                self.assertEqual(tree(self.root), self.committed)
                self.assertEqual(leftover_stages(op, self.root), [])


class CommittedObjectDisassemblyTest(unittest.TestCase):
    def test_every_committed_object_disassembles_with_packed_f32(self):
        for op in OPS.values():
            for arch in op.ARCHES:
                manifest = committed_manifest(op, arch)
                self.assertEqual(len(manifest["instances"]), len(op.instances()))
                for record in manifest["instances"]:
                    where = f"{op.OP}/{arch}/{record['file']}"
                    with self.subTest(where=where):
                        blob = (COMMITTED / op.OP / arch / record["file"]).read_bytes()
                        text = disassemble(blob, arch, where)
                        self.assertIn(op.KERNEL_SYMBOL, text)
                        self.assertGreater(isa_counts(text)[1], 0)


if __name__ == "__main__":
    unittest.main()
