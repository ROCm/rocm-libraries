#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import struct
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "verify_golden_bundles.py"


class TestVerifyGoldenBundlesCli(unittest.TestCase):
    def run_verifier(
        self,
        *roots: Path,
        default_tier: str | None = None,
        require_data: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(SCRIPT_PATH)]
        if default_tier is not None:
            command.extend(["--default-tier", default_tier])
        if require_data:
            command.append("--require-data")
        command.extend(str(root) for root in roots)
        return subprocess.run(command, capture_output=True, text=True, check=False)

    def default_bytes(self, data_type: str) -> bytes:
        dtype_key = data_type.lower()
        if dtype_key in {"float", "float32", "fp32"}:
            return struct.pack("<f", 1.0)
        if dtype_key in {"half", "float16", "fp16"}:
            return struct.pack("<H", 0x3C00)
        if dtype_key in {"bfloat16", "bf16", "bfp16"}:
            return struct.pack("<H", 0x3F80)
        raise ValueError(f"Unsupported test dtype {data_type}")

    def filled_bytes(self, data_type: str, elements: int) -> bytes:
        return self.default_bytes(data_type) * elements

    def write_bundle(
        self,
        root: Path,
        relative_dir: Path,
        *,
        input_dtype: str = "float",
        output_dtype: str = "float",
        input_dims: tuple[int, ...] = (1,),
        output_dims: tuple[int, ...] = (1,),
        input_bytes: bytes | None = None,
        output_bytes: bytes | None = None,
        write_input_tensor: bool = True,
        write_output_tensor: bool = True,
        write_tensor_manifest: bool = False,
        metadata: dict[str, object] | None = None,
        nodes: list[dict[str, object]] | None = None,
    ) -> Path:
        bundle_dir = root / relative_dir
        bundle_dir.mkdir(parents=True)
        name = bundle_dir.name
        graph_path = bundle_dir / f"{name}.json"
        input_payload = (
            input_bytes if input_bytes is not None else self.default_bytes(input_dtype)
        )
        output_payload = (
            output_bytes
            if output_bytes is not None
            else self.default_bytes(output_dtype)
        )

        graph = {
            "nodes": nodes if nodes is not None else [{"outputs": {"y_tensor_uid": 1}}],
            "tensors": [
                {
                    "uid": 0,
                    "dims": list(input_dims),
                    "strides": [1] * len(input_dims),
                    "data_type": input_dtype,
                    "virtual": False,
                },
                {
                    "uid": 1,
                    "dims": list(output_dims),
                    "strides": [1] * len(output_dims),
                    "data_type": output_dtype,
                    "virtual": False,
                },
            ],
            "io_data_type": output_dtype,
            "compute_data_type": output_dtype,
            "intermediate_data_type": output_dtype,
            "name": "",
        }
        graph_path.write_text(json.dumps(graph))

        if write_input_tensor:
            (bundle_dir / f"{name}.tensor0.bin").write_bytes(input_payload)
        if write_output_tensor:
            (bundle_dir / f"{name}.tensor1.bin").write_bytes(output_payload)
        if write_tensor_manifest:
            (bundle_dir / f"{name}.tensors.dvc").write_text(
                "\n".join(
                    [
                        "outs:",
                        f"- path: {name}.tensor0.bin",
                        f"  size: {len(input_payload)}",
                        f"- path: {name}.tensor1.bin",
                        f"  size: {len(output_payload)}",
                    ]
                )
                + "\n"
            )
        if metadata is not None:
            (bundle_dir / f"{name}.meta.json").write_text(json.dumps(metadata))

        return graph_path

    def write_sweep_bundle(
        self, root: Path, relative_dir: Path, cases: list[dict[str, object]]
    ) -> tuple[Path, Path]:
        bundle_dir = root / relative_dir
        bundle_dir.mkdir(parents=True)
        template = {
            "nodes": [
                {
                    "inputs": {"x_tensor_uid": 0},
                    "outputs": {"y_tensor_uid": 1},
                    "type": "TestOpAttributes",
                    "name": "",
                }
            ],
            "tensors": [
                {
                    "uid": 0,
                    "name": "",
                    "dims": "${case.dims}",
                    "strides": "${case.strides}",
                    "data_type": "${case.data_type}",
                    "virtual": False,
                },
                {
                    "uid": 1,
                    "name": "",
                    "dims": "${case.dims}",
                    "strides": "${case.strides}",
                    "data_type": "${case.data_type}",
                    "virtual": False,
                },
            ],
            "io_data_type": "${case.io_data_type}",
            "compute_data_type": "float",
            "intermediate_data_type": "float",
            "name": "",
        }
        template_path = bundle_dir / "graph.template.json"
        template_path.write_text(json.dumps(template))

        sweep_path = bundle_dir / "sweep.json"
        sweep_path.write_text(json.dumps({"version": 1, "cases": cases}))

        return template_path, sweep_path

    def write_sweep_golden_case(
        self,
        bundle_dir: Path,
        case_id: str,
        *,
        input_bytes: bytes,
        output_bytes: bytes,
        write_input_tensor: bool = True,
        write_output_tensor: bool = True,
        write_tensor_manifest: bool = True,
    ) -> Path:
        case_dir = bundle_dir / "golden" / case_id
        case_dir.mkdir(parents=True)
        if write_input_tensor:
            (case_dir / "tensor0.bin").write_bytes(input_bytes)
        if write_output_tensor:
            (case_dir / "tensor1.bin").write_bytes(output_bytes)
        if write_tensor_manifest:
            (case_dir / "tensors.dvc").write_text(
                "\n".join(
                    [
                        "outs:",
                        "- path: tensor0.bin",
                        f"  size: {len(input_bytes)}",
                        "- path: tensor1.bin",
                        f"  size: {len(output_bytes)}",
                    ]
                )
                + "\n"
            )
        return case_dir

    def sweep_case(
        self,
        case_id: str,
        *,
        dims: tuple[int, ...] = (1,),
        data_type: str = "float",
        values_overrides: dict[str, object] | None = None,
        tensor_uids: tuple[int, ...] = (0, 1),
        metadata: dict[str, object] | None = None,
        golden_path: str | None = "",
    ) -> dict[str, object]:
        values: dict[str, object] = {
            "io_data_type": data_type,
            "tensors": [
                {
                    "uid": uid,
                    "data_type": data_type,
                    "dims": list(dims),
                    "strides": [1] * len(dims),
                }
                for uid in tensor_uids
            ],
        }
        if values_overrides:
            values.update(values_overrides)

        case: dict[str, object] = {
            "id": case_id,
            "values": values,
            "metadata": (
                metadata
                if metadata is not None
                else {"generator": "manual", "reference_source": "manual"}
            ),
        }
        if golden_path == "":
            golden_path = f"golden/{case_id}/tensors.dvc"
        if golden_path is not None:
            case["golden"] = {"path": golden_path}
        return case

    def test_valid_bundle_prints_canonical_path_and_full_test_name(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_bundle(
                root,
                Path("quick/BatchnormFwdInference/nchw/fp32/Small"),
                metadata={"generator": "manual", "reference_source": "manual"},
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "canonical_path: quick/BatchnormFwdInference/nchw/fp32/Small/",
                completed.stdout,
            )
            self.assertIn(
                "full_test_name: quick_BatchnormFwdInference_nchw_fp32_Small.Small",
                completed.stdout,
            )

    def test_nan_output_tensor_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_bundle(
                root,
                Path("quick/BatchnormFwdInference/nchw/fp32/Small"),
                output_bytes=struct.pack("<f", float("nan")),
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("tensor uid 1", completed.stderr)
            self.assertIn("output tensor contains NaN/Inf", completed.stderr)

    def test_nan_input_tensor_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_bundle(
                root,
                Path("quick/BatchnormFwdInference/nchw/fp32/Small"),
                input_bytes=struct.pack("<f", float("nan")),
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("tensor uid 0", completed.stderr)
            self.assertIn("input tensor contains NaN/Inf", completed.stderr)

    def test_flat_output_uid_node_type_is_valid(self) -> None:
        # ReductionAttributes puts "out_tensor_uid" directly on the node instead
        # of nesting it under an "outputs" object like every other node type.
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_bundle(
                root,
                Path("quick/Reduction/nchw/fp32/Small"),
                nodes=[
                    {
                        "type": "ReductionAttributes",
                        "in_tensor_uid": 0,
                        "out_tensor_uid": 1,
                    }
                ],
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertNotIn("outputs is required", completed.stderr)

    def test_flat_output_uid_node_type_nan_output_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_bundle(
                root,
                Path("quick/Reduction/nchw/fp32/Small"),
                nodes=[
                    {
                        "type": "ReductionAttributes",
                        "in_tensor_uid": 0,
                        "out_tensor_uid": 1,
                    }
                ],
                output_bytes=struct.pack("<f", float("nan")),
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("tensor uid 1", completed.stderr)
            self.assertIn("output tensor contains NaN/Inf", completed.stderr)

    def test_truncated_tensor_file_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_bundle(
                root,
                Path("quick/BatchnormFwdInference/nchw/fp32/Small"),
                output_dims=(2,),
                output_bytes=struct.pack("<f", 1.0),
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("tensor uid 1", completed.stderr)
            self.assertIn("file has", completed.stderr)
            self.assertIn("graph expects", completed.stderr)

    def test_bundle_over_1mib_warns(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_elements = 1024 * 1024 // 4
            self.write_bundle(
                root,
                Path("quick/BatchnormFwdInference/nchw/fp32/LargeWarn"),
                output_dims=(output_elements,),
                output_bytes=self.filled_bytes("float", output_elements),
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("WARNING", completed.stderr)
            self.assertIn("bundle totals", completed.stderr)
            self.assertIn(
                "keep bundles at or below 1 MiB when possible", completed.stderr
            )

    def test_bundle_over_2mib_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_elements = 2 * 1024 * 1024 // 4
            self.write_bundle(
                root,
                Path("quick/BatchnormFwdInference/nchw/fp32/LargeFail"),
                output_dims=(output_elements,),
                output_bytes=self.filled_bytes("float", output_elements),
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("bundle totals", completed.stderr)
            self.assertIn(
                "cannot have bundles larger than 2 MiB because they would quickly explode our test artifact sizes",
                completed.stderr,
            )

    def test_missing_output_tensor_file_is_optional_without_tensor_manifest(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_bundle(
                root,
                Path("quick/BatchnormFwdInference/nchw/fp32/Small"),
                write_output_tensor=False,
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertNotIn("missing tensor file", completed.stderr)

    def test_missing_output_tensor_file_warns_without_require_data(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_bundle(
                root,
                Path("quick/BatchnormFwdInference/nchw/fp32/Small"),
                write_output_tensor=False,
                write_tensor_manifest=True,
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("tensor uid 1", completed.stderr)
            self.assertIn("tensor data not pulled locally", completed.stderr)

    def test_missing_output_tensor_file_fails_with_tensor_manifest(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_bundle(
                root,
                Path("quick/BatchnormFwdInference/nchw/fp32/Small"),
                write_output_tensor=False,
                write_tensor_manifest=True,
            )

            completed = self.run_verifier(root, require_data=True)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("tensor uid 1", completed.stderr)
            self.assertIn("missing tensor file", completed.stderr)

    def test_metadata_missing_reference_source_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_bundle(
                root,
                Path("quick/BatchnormFwdInference/nchw/fp32/Small"),
                metadata={"generator": "manual"},
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("Small.meta.json", completed.stderr)
            self.assertIn("reference_source", completed.stderr)

    def test_non_graph_json_warns_and_is_ignored(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "quick").mkdir()
            (root / "quick/README.json").write_text(json.dumps({"note": "ignored"}))

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 0)
            self.assertIn("WARNING", completed.stderr)
            self.assertIn("non-graph JSON ignored", completed.stderr)

    def test_unexpected_top_level_directory_warns(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "quick").mkdir()
            (root / "quik").mkdir()

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 0)
            self.assertIn("WARNING", completed.stderr)
            self.assertIn("unexpected top-level directory", completed.stderr)

    def test_fp16_and_bf16_nonfinite_detection(self) -> None:
        cases = [
            ("half", Path("quick/TestOp/nchw/fp16/Small"), 0x7C00, 0x3C00),
            ("bfloat16", Path("quick/TestOp/nchw/bf16/Small"), 0x7F80, 0x3F80),
        ]

        for data_type, relative_dir, nonfinite_word, finite_word in cases:
            with self.subTest(data_type=data_type, kind="nonfinite"):
                with TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    self.write_bundle(
                        root,
                        relative_dir,
                        input_dtype=data_type,
                        output_dtype=data_type,
                        input_bytes=struct.pack("<H", finite_word),
                        output_bytes=struct.pack("<H", nonfinite_word),
                    )

                    completed = self.run_verifier(root)

                    self.assertEqual(completed.returncode, 1)
                    self.assertIn("tensor uid 1", completed.stderr)
                    self.assertIn("output tensor contains NaN/Inf", completed.stderr)

            with self.subTest(data_type=data_type, kind="finite"):
                with TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    self.write_bundle(
                        root,
                        relative_dir,
                        input_dtype=data_type,
                        output_dtype=data_type,
                        input_bytes=struct.pack("<H", finite_word),
                        output_bytes=struct.pack("<H", finite_word),
                    )

                    completed = self.run_verifier(root)

                    self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_valid_sweep_bundle_prints_advisories(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_path, sweep_path = self.write_sweep_bundle(
                root,
                Path("quick/TestOp/Topology"),
                [self.sweep_case("small_fp32")],
            )
            self.write_sweep_golden_case(
                sweep_path.parent,
                "small_fp32",
                input_bytes=self.default_bytes("float"),
                output_bytes=self.default_bytes("float"),
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "canonical_path: quick/TestOp/Topology/sweep.json", completed.stdout
            )
            self.assertIn(
                "full_test_name: quick_TestOp_Topology.small_fp32", completed.stdout
            )

    def test_sweep_duplicate_case_id_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_sweep_bundle(
                root,
                Path("quick/TestOp/Topology"),
                [
                    self.sweep_case("small_fp32", golden_path=None),
                    self.sweep_case("small_fp32", golden_path=None),
                ],
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("duplicate case id 'small_fp32'", completed.stderr)

    def test_sweep_case_id_must_be_snake_case(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_sweep_bundle(
                root,
                Path("quick/TestOp/Topology"),
                [self.sweep_case("Small-FP32", golden_path=None)],
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("must be lowercase_snake_case", completed.stderr)

    def test_sweep_missing_tensor_uid_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_sweep_bundle(
                root,
                Path("quick/TestOp/Topology"),
                [self.sweep_case("small_fp32", tensor_uids=(0,), golden_path=None)],
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "is missing tensor uid(s) [1] present in template graph",
                completed.stderr,
            )

    def test_sweep_tensor_uid_not_in_template_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_sweep_bundle(
                root,
                Path("quick/TestOp/Topology"),
                [
                    self.sweep_case(
                        "small_fp32", tensor_uids=(0, 1, 99), golden_path=None
                    )
                ],
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "tensor uid is not present in template graph", completed.stderr
            )

    def test_sweep_missing_scalar_placeholder_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_path, sweep_path = self.write_sweep_bundle(
                root, Path("quick/TestOp/Topology"), []
            )
            case = self.sweep_case("small_fp32", golden_path=None)
            del case["values"]["io_data_type"]
            sweep_path.write_text(json.dumps({"version": 1, "cases": [case]}))

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "is missing placeholder value for 'io_data_type'", completed.stderr
            )

    def test_sweep_unused_values_entry_warns(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_sweep_bundle(
                root,
                Path("quick/TestOp/Topology"),
                [
                    self.sweep_case(
                        "small_fp32",
                        values_overrides={"layout": "nchw"},
                        golden_path=None,
                    )
                ],
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("has unused values entry 'layout'", completed.stderr)

    def test_sweep_golden_path_must_reference_tensors_dvc(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_sweep_bundle(
                root,
                Path("quick/TestOp/Topology"),
                [
                    self.sweep_case(
                        "small_fp32", golden_path="golden/small_fp32/notes.txt"
                    )
                ],
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "golden.path must reference a tensors.dvc file", completed.stderr
            )

    def test_sweep_invalid_golden_path_does_not_validate_unrelated_directory(
        self,
    ) -> None:
        # A case whose golden.path fails the tensors.dvc-name check must not
        # have golden_dir derived from that bad path and fed into
        # tensor-payload validation — otherwise an unrelated (but real)
        # golden directory belonging to another case gets validated a
        # second time, duplicating its diagnostics under the wrong case.
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_path, sweep_path = self.write_sweep_bundle(
                root,
                Path("quick/TestOp/Topology"),
                [
                    self.sweep_case("wrong_target"),
                    self.sweep_case(
                        "small_fp32", golden_path="golden/wrong_target/notes.txt"
                    ),
                ],
            )
            self.write_sweep_golden_case(
                sweep_path.parent,
                "wrong_target",
                input_bytes=self.default_bytes("float"),
                output_bytes=struct.pack("<f", float("nan")),
            )

            completed = self.run_verifier(root, require_data=True)

            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "golden.path must reference a tensors.dvc file", completed.stderr
            )
            self.assertEqual(completed.stderr.count("NaN/Inf"), 1)

    def test_sweep_missing_tensor_file_fails_with_manifest(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_path, sweep_path = self.write_sweep_bundle(
                root,
                Path("quick/TestOp/Topology"),
                [self.sweep_case("small_fp32")],
            )
            self.write_sweep_golden_case(
                sweep_path.parent,
                "small_fp32",
                input_bytes=self.default_bytes("float"),
                output_bytes=self.default_bytes("float"),
                write_output_tensor=False,
            )

            completed = self.run_verifier(root, require_data=True)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("tensor uid 1", completed.stderr)
            self.assertIn("missing tensor file", completed.stderr)

    def test_sweep_nan_output_tensor_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_path, sweep_path = self.write_sweep_bundle(
                root,
                Path("quick/TestOp/Topology"),
                [self.sweep_case("small_fp32")],
            )
            self.write_sweep_golden_case(
                sweep_path.parent,
                "small_fp32",
                input_bytes=self.default_bytes("float"),
                output_bytes=struct.pack("<f", float("nan")),
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("tensor uid 1", completed.stderr)
            self.assertIn("output tensor contains NaN/Inf", completed.stderr)

    def test_sweep_nan_input_tensor_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_path, sweep_path = self.write_sweep_bundle(
                root,
                Path("quick/TestOp/Topology"),
                [self.sweep_case("small_fp32")],
            )
            self.write_sweep_golden_case(
                sweep_path.parent,
                "small_fp32",
                input_bytes=struct.pack("<f", float("nan")),
                output_bytes=self.default_bytes("float"),
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("tensor uid 0", completed.stderr)
            self.assertIn("input tensor contains NaN/Inf", completed.stderr)

    def test_sweep_case_metadata_missing_reference_source_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_sweep_bundle(
                root,
                Path("quick/TestOp/Topology"),
                [
                    self.sweep_case(
                        "small_fp32",
                        metadata={"generator": "manual"},
                        golden_path=None,
                    )
                ],
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "case 'small_fp32' metadata reference_source", completed.stderr
            )

    def test_sweep_bundle_missing_sweep_json_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_dir = root / "quick/TestOp/Topology"
            bundle_dir.mkdir(parents=True)
            (bundle_dir / "graph.template.json").write_text(json.dumps({"nodes": []}))

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("missing sweep.json", completed.stderr)

    def test_sweep_bundle_missing_template_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_dir = root / "quick/TestOp/Topology"
            bundle_dir.mkdir(parents=True)
            (bundle_dir / "sweep.json").write_text(
                json.dumps({"version": 1, "cases": []})
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("missing graph.template.json", completed.stderr)

    def test_sweep_case_over_2mib_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_elements = 2 * 1024 * 1024 // 4
            template_path, sweep_path = self.write_sweep_bundle(
                root,
                Path("quick/TestOp/Topology"),
                [self.sweep_case("large_fp32", dims=(output_elements,))],
            )
            self.write_sweep_golden_case(
                sweep_path.parent,
                "large_fp32",
                input_bytes=self.filled_bytes("float", output_elements),
                output_bytes=self.filled_bytes("float", output_elements),
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("bundle totals", completed.stderr)
            self.assertIn(
                "cannot have bundles larger than 2 MiB because they would quickly explode our test artifact sizes",
                completed.stderr,
            )


if __name__ == "__main__":
    unittest.main()
