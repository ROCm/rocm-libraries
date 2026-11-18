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

sys.path.insert(0, str(SCRIPT_PATH.parent))
import verify_golden_bundles  # noqa: E402


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

    def test_variant_directory_bundle_keeps_variant_in_canonical_path_and_suite(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_bundle(
                root,
                Path("quick/SdpaFwd/bshd/bf16/hd192_nomask_ragged/Small"),
                metadata={"generator": "manual", "reference_source": "manual"},
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "canonical_path: quick/SdpaFwd/bshd/bf16/hd192_nomask_ragged/Small/",
                completed.stdout,
            )
            self.assertIn(
                "full_test_name: "
                "quick_SdpaFwd_bshd_bf16_hd192_nomask_ragged_Small.Small",
                completed.stdout,
            )

    def test_tier_named_ancestor_directory_is_excluded_from_suite(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "quick" / "checkout"
            self.write_bundle(
                root,
                Path("quick/SdpaFwd/bshd/bf16/hd192_nomask_ragged/Small"),
                metadata={"generator": "manual", "reference_source": "manual"},
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "full_test_name: "
                "quick_SdpaFwd_bshd_bf16_hd192_nomask_ragged_Small.Small",
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

    # A padded SDPA batch: the dead row's LSE is -inf by contract (SEM-010).
    SDPA_FWD_NODES = [
        {
            "type": "SdpaAttributes",
            "inputs": {"q_tensor_uid": 0},
            "outputs": {"o_tensor_uid": None, "stats_tensor_uid": 1},
        }
    ]
    SDPA_BWD_NODES = [
        {
            "type": "SdpaBackwardAttributes",
            "inputs": {"q_tensor_uid": None, "stats_tensor_uid": 0},
            "outputs": {"dq_tensor_uid": 1},
        }
    ]

    def padded_lse(self, dead_value: float) -> bytes:
        return struct.pack("<3f", 0.5, dead_value, 1.25)

    def test_sdpa_padded_stats_neg_inf_passes(self) -> None:
        for label, nodes, padded in (
            ("forward stats output", self.SDPA_FWD_NODES, "output"),
            ("backward stats input", self.SDPA_BWD_NODES, "input"),
        ):
            with self.subTest(label=label):
                with TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    payload = {
                        "input_bytes": self.filled_bytes("float", 3),
                        "output_bytes": self.filled_bytes("float", 3),
                        f"{padded}_bytes": self.padded_lse(float("-inf")),
                    }
                    self.write_bundle(
                        root,
                        Path("quick/SdpaFwd/bhsd/fp32/padneq/Small"),
                        input_dims=(3,),
                        output_dims=(3,),
                        nodes=nodes,
                        **payload,
                    )

                    completed = self.run_verifier(root, require_data=True)

                    self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_sdpa_stats_pos_inf_or_nan_fails(self) -> None:
        for value in (float("inf"), float("nan")):
            with self.subTest(value=value):
                with TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    self.write_bundle(
                        root,
                        Path("quick/SdpaFwd/bhsd/fp32/padneq/Small"),
                        output_dims=(3,),
                        output_bytes=self.padded_lse(value),
                        nodes=self.SDPA_FWD_NODES,
                    )

                    completed = self.run_verifier(root)

                    self.assertEqual(completed.returncode, 1)
                    self.assertIn("tensor uid 1", completed.stderr)
                    self.assertIn(
                        "SDPA stats tensor contains NaN/+Inf at element index 1",
                        completed.stderr,
                    )

    def test_sdpa_neg_inf_outside_stats_fails(self) -> None:
        # Only the stats tensor may hold -inf; the same value in O is a defect.
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_bundle(
                root,
                Path("quick/SdpaFwd/bhsd/fp32/padneq/Small"),
                output_dims=(3,),
                output_bytes=self.padded_lse(float("-inf")),
                nodes=[
                    {
                        "type": "SdpaAttributes",
                        "inputs": {"q_tensor_uid": 0},
                        "outputs": {"o_tensor_uid": 1, "stats_tensor_uid": None},
                    }
                ],
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "output tensor contains NaN/Inf at element index 1", completed.stderr
            )

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

    def run_single_graph_with_metadata(
        self, metadata: dict[str, object], **bundle_options: object
    ) -> subprocess.CompletedProcess[str]:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_bundle(
                root,
                Path("quick/BatchnormFwdInference/nchw/fp32/Small"),
                metadata=metadata,
                **bundle_options,
            )
            return self.run_verifier(root)

    def test_reference_flag_must_be_boolean(self) -> None:
        for flag in ("no_cpu_reference", "no_gpu_reference"):
            for bad_value in ("yes", 1, None):
                with self.subTest(flag=flag, value=bad_value):
                    completed = self.run_single_graph_with_metadata(
                        {
                            "generator": "manual",
                            "reference_source": "manual",
                            flag: bad_value,
                        }
                    )

                    self.assertEqual(completed.returncode, 1, completed.stdout)
                    self.assertIn("Small.meta.json", completed.stderr)
                    self.assertIn(f"{flag} must be a boolean", completed.stderr)

    def test_boolean_reference_flags_pass(self) -> None:
        completed = self.run_single_graph_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "no_cpu_reference": False,
                "no_gpu_reference": True,
            }
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_disabling_both_references_without_golden_fails(self) -> None:
        completed = self.run_single_graph_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "no_cpu_reference": True,
                "no_gpu_reference": True,
            },
            write_input_tensor=False,
            write_output_tensor=False,
        )

        self.assertEqual(completed.returncode, 1, completed.stdout)
        self.assertIn("Small.meta.json", completed.stderr)
        self.assertIn("disables both references", completed.stderr)
        self.assertIn("declares no golden data", completed.stderr)

    def test_disabling_both_references_with_tensor_manifest_only_passes(self) -> None:
        completed = self.run_single_graph_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "no_cpu_reference": True,
                "no_gpu_reference": True,
            },
            write_input_tensor=False,
            write_output_tensor=False,
            write_tensor_manifest=True,
        )

        self.assertNotIn("disables both references", completed.stderr)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_disabling_both_references_with_output_blob_only_passes(self) -> None:
        completed = self.run_single_graph_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "no_cpu_reference": True,
                "no_gpu_reference": True,
            },
            write_input_tensor=False,
            write_output_tensor=True,
        )

        self.assertNotIn("disables both references", completed.stderr)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_disabling_one_reference_without_golden_passes(self) -> None:
        for flag in ("no_cpu_reference", "no_gpu_reference"):
            with self.subTest(flag=flag):
                completed = self.run_single_graph_with_metadata(
                    {"generator": "manual", "reference_source": "manual", flag: True},
                    write_input_tensor=False,
                    write_output_tensor=False,
                )

                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_non_boolean_reference_flag_does_not_double_report(self) -> None:
        completed = self.run_single_graph_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "no_cpu_reference": True,
                "no_gpu_reference": "yes",
            },
            write_input_tensor=False,
            write_output_tensor=False,
        )

        self.assertEqual(completed.returncode, 1, completed.stdout)
        self.assertIn("no_gpu_reference must be a boolean", completed.stderr)
        self.assertNotIn("disables both references", completed.stderr)

    def test_expected_rejection_must_be_boolean(self) -> None:
        for bad_value in ("yes", 1, None):
            with self.subTest(value=bad_value):
                completed = self.run_single_graph_with_metadata(
                    {
                        "generator": "manual",
                        "reference_source": "manual",
                        "expected_rejection": bad_value,
                    },
                    write_input_tensor=False,
                    write_output_tensor=False,
                )

                self.assertEqual(completed.returncode, 1, completed.stdout)
                self.assertIn("Small.meta.json", completed.stderr)
                self.assertIn("expected_rejection must be a boolean", completed.stderr)

    def test_non_boolean_expected_rejection_does_not_double_report(self) -> None:
        completed = self.run_single_graph_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "expected_rejection": "yes",
            },
            write_input_tensor=True,
            write_output_tensor=True,
            write_tensor_manifest=True,
        )

        self.assertEqual(completed.returncode, 1, completed.stdout)
        self.assertIn("expected_rejection must be a boolean", completed.stderr)
        self.assertNotIn("declares golden data", completed.stderr)

    def test_expected_rejection_without_golden_passes(self) -> None:
        completed = self.run_single_graph_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "expected_rejection": True,
            },
            write_input_tensor=False,
            write_output_tensor=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_expected_rejection_with_tensor_manifest_fails(self) -> None:
        completed = self.run_single_graph_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "expected_rejection": True,
            },
            write_input_tensor=False,
            write_output_tensor=False,
            write_tensor_manifest=True,
        )

        self.assertEqual(completed.returncode, 1, completed.stdout)
        self.assertIn("Small.meta.json", completed.stderr)
        self.assertIn(
            "is an expected_rejection bundle but declares golden data", completed.stderr
        )

    def test_expected_rejection_with_output_blobs_fails(self) -> None:
        completed = self.run_single_graph_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "expected_rejection": True,
            },
            write_input_tensor=False,
            write_output_tensor=True,
        )

        self.assertEqual(completed.returncode, 1, completed.stdout)
        self.assertIn(
            "is an expected_rejection bundle but declares golden data",
            completed.stderr,
        )

    # Input blobs alone are not golden data (the loader needs every output's blob),
    # so they neither trip the expected-rejection rule nor satisfy the both-off one.
    def test_expected_rejection_with_input_blobs_only_passes(self) -> None:
        completed = self.run_single_graph_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "expected_rejection": True,
            },
            write_input_tensor=True,
            write_output_tensor=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_both_references_disabled_with_input_blobs_only_fails(self) -> None:
        completed = self.run_single_graph_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "no_cpu_reference": True,
                "no_gpu_reference": True,
            },
            write_input_tensor=True,
            write_output_tensor=False,
        )

        self.assertEqual(completed.returncode, 1, completed.stdout)
        self.assertIn("disables both references", completed.stderr)

    # A virtual intermediate output has no blob, and the loader does not ask for one:
    # golden data is declared once every non-virtual output has its blob.
    def test_both_references_disabled_with_blob_for_real_output_only_passes(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_dir = root / "quick/BatchnormFwdInference/nchw/fp32/Small"
            bundle_dir.mkdir(parents=True)
            tensors = [
                {"uid": uid, "dims": [1], "strides": [1], "data_type": "float"}
                | {"virtual": uid == 1}
                for uid in (0, 1, 2)
            ]
            graph = {
                "nodes": [
                    {"outputs": {"y_tensor_uid": 1}},
                    {"outputs": {"y_tensor_uid": 2}},
                ],
                "tensors": tensors,
                "io_data_type": "float",
                "compute_data_type": "float",
                "intermediate_data_type": "float",
                "name": "",
            }
            (bundle_dir / "Small.json").write_text(json.dumps(graph))
            (bundle_dir / "Small.tensor2.bin").write_bytes(self.default_bytes("float"))
            (bundle_dir / "Small.meta.json").write_text(
                json.dumps(
                    {
                        "generator": "manual",
                        "reference_source": "manual",
                        "no_cpu_reference": True,
                        "no_gpu_reference": True,
                    }
                )
            )

            completed = self.run_verifier(root)

        self.assertNotIn("disables both references", completed.stderr)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    # The reference flags are ignored on an expected rejection, so setting both
    # without golden data is not the both-off error.
    def test_expected_rejection_with_both_references_disabled_passes(self) -> None:
        completed = self.run_single_graph_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "expected_rejection": True,
                "no_cpu_reference": True,
                "no_gpu_reference": True,
            },
            write_input_tensor=False,
            write_output_tensor=False,
        )

        self.assertNotIn("disables both references", completed.stderr)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_expected_rejection_false_with_golden_passes(self) -> None:
        completed = self.run_single_graph_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "expected_rejection": False,
            }
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_checked_in_expected_rejection_example_passes(self) -> None:
        bundle_dir = (
            SCRIPT_PATH.parent.parent
            / "integration-test-bundles/quick/Matmul/ncl/fp32"
            / "ExpectedRejectionMatmulBatchMismatch"
        )
        meta = json.loads(
            (bundle_dir / "ExpectedRejectionMatmulBatchMismatch.meta.json").read_text()
        )
        self.assertIs(meta["expected_rejection"], True)

        completed = self.run_verifier(bundle_dir)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn("ERROR", completed.stderr)

    # RFC 0011 4.1 and 4.2: a backward bundle must carry a .meta.json naming the
    # reference strategy and the forward pass whose O and LSE its gradients were
    # computed from. Only backward bundles are held to this; forward bundles
    # migrated from the old test system may have no metadata at all.

    BACKWARD_METADATA = {
        "generator": "generate_sdpa_bwd_golden.py",
        "reference_source": "PyTorch 2.14.1",
        "reference_strategy": "matched_precision",
        "direction": "backward",
        "forward_source": {
            "method": "pytorch_math_backend",
            "version": "2.14.1",
            "generated_at": "2026-10-20T00:00:00Z",
        },
    }

    def write_backward_bundle(self, root: Path, **overrides: object) -> Path:
        metadata = {**self.BACKWARD_METADATA, **overrides}
        metadata = {key: value for key, value in metadata.items() if value is not None}
        return self.write_bundle(
            root,
            Path("quick/SdpaBackward/bhsd/bf16/Small"),
            metadata=metadata,
        )

    def test_well_formed_backward_metadata_passes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_backward_bundle(root)

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_backward_metadata_without_forward_source_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_backward_bundle(root, forward_source=None)

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "forward_source is required on backward bundles", completed.stderr
            )

    def test_backward_metadata_without_reference_strategy_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_backward_bundle(root, reference_strategy=None)

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "reference_strategy is required on backward bundles", completed.stderr
            )

    def test_forward_source_must_name_method_version_and_timestamp(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_backward_bundle(
                root, forward_source={"method": "pytorch_math_backend"}
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("forward_source.version is required", completed.stderr)
            self.assertIn("forward_source.generated_at is required", completed.stderr)

    def test_forward_source_must_be_an_object(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_backward_bundle(root, forward_source="pytorch")

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("forward_source must be an object", completed.stderr)

    def test_forward_metadata_does_not_require_forward_source(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_bundle(
                root,
                Path("quick/SdpaFwd/bhsd/bf16/Small"),
                metadata={
                    "generator": "generate_sdpa_fwd_golden.py",
                    "reference_source": "PyTorch 2.14.1",
                    "direction": "forward",
                },
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_unknown_direction_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_backward_bundle(root, direction="sideways")

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("direction must be 'forward' or 'backward'", completed.stderr)

    def _write_pair(self, root: Path, **forward_source_overrides: object):
        """A backward bundle alongside the forward bundle its metadata names."""
        self.write_bundle(
            root,
            Path("quick/SdpaBackward/bhsd/bf16/Small_fwd_pair"),
            metadata={
                "generator": "generate_sdpa_bwd_golden.py",
                "reference_source": "PyTorch 2.14.1",
                "reference_strategy": "matched_precision",
                "direction": "forward",
                "reference_backend": "pytorch_math_backend",
                "generated_at": "2026-10-20T00:00:00Z",
            },
        )
        forward_source = {
            **self.BACKWARD_METADATA["forward_source"],
            "bundle": "../Small_fwd_pair/Small_fwd_pair.json",
            **forward_source_overrides,
        }
        self.write_backward_bundle(root, forward_source=forward_source)

    def test_matching_forward_source_pairing_passes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_pair(root)

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_stale_forward_source_timestamp_fails(self) -> None:
        """The failure a forward bundle regenerated on its own would produce."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_pair(root, generated_at="2026-01-01T00:00:00Z")

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "the forward bundle was regenerated without its backward partner",
                completed.stderr,
            )

    def test_mismatched_forward_source_method_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_pair(root, method="cudnn")

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("forward_source.method is 'cudnn'", completed.stderr)

    def test_forward_source_naming_an_absent_bundle_is_not_an_error(self) -> None:
        """Verifying a subtree without the forward partner pulled must still pass."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_backward_bundle(
                root,
                forward_source={
                    **self.BACKWARD_METADATA["forward_source"],
                    "bundle": "../NotPulled/NotPulled.json",
                },
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)

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

    def test_sweep_tensor_field_placeholders_resolve_by_first_token(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_path, sweep_path = self.write_sweep_bundle(
                root, Path("quick/TestOp/Topology"), []
            )
            template = json.loads(template_path.read_text())
            template["tensors"][0]["alignment"] = "${case.alignment.0}"
            template["tensors"][1]["alignment"] = "${case.alignment.1}"
            template["tensors"][1]["value"] = "${case.value}"
            template_path.write_text(json.dumps(template))
            used = self.sweep_case(
                "used",
                values_overrides={"alignment": {"0": 16, "1": 2}},
                golden_path=None,
            )
            used["values"]["tensors"][1]["value"] = 0.5
            missing = self.sweep_case("missing", golden_path=None)
            sweep_path.write_text(json.dumps({"version": 1, "cases": [used, missing]}))

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertNotIn("case 'used' has unused values entry", completed.stderr)
            self.assertNotIn("case 'used' is missing", completed.stderr)
            self.assertIn(
                "case 'missing' is missing placeholder value for 'alignment.0'",
                completed.stderr,
            )
            self.assertIn(
                "case 'missing' is missing placeholder value for 'value'",
                completed.stderr,
            )

    def test_sweep_dotted_placeholder_nested_in_a_tensor_entry_now_resolves(
        self,
    ) -> None:
        # The C++ loader's resolvePlaceholder()/lookupJsonPath() walk a (possibly
        # dotted) placeholder path against a tensor's own values.tensors[] entry
        # with no restriction on field name, descending through nested dicts. A
        # dotted placeholder like alignment.0 therefore DOES resolve when the
        # tensor's own entry holds the matching nested structure -- this used to
        # be rejected because the verifier only ever checked a flat whitelist of
        # field names against a single dotted token, not a real recursive walk.
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_path, sweep_path = self.write_sweep_bundle(
                root, Path("quick/TestOp/Topology"), []
            )
            template = json.loads(template_path.read_text())
            template["tensors"][0]["alignment"] = "${case.alignment.0}"
            template_path.write_text(json.dumps(template))
            case = self.sweep_case("nested", golden_path=None)
            case["values"]["tensors"][0]["alignment"] = {"0": 16}
            sweep_path.write_text(json.dumps({"version": 1, "cases": [case]}))

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_sweep_dotted_placeholder_over_a_tensor_scalar_is_still_missing(
        self,
    ) -> None:
        # The real false-negative this family of tests originally caught: a
        # dotted placeholder (alignment.0) whose tensor-scoped value is a plain
        # scalar, not the nested object the dotted path needs to walk through.
        # The recursive walk can't descend into a non-dict 16 to find key "0",
        # so this must still be reported as missing -- distinct from the
        # previous test, which nests the value correctly.
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_path, sweep_path = self.write_sweep_bundle(
                root, Path("quick/TestOp/Topology"), []
            )
            template = json.loads(template_path.read_text())
            template["tensors"][0]["alignment"] = "${case.alignment.0}"
            template_path.write_text(json.dumps(template))
            case = self.sweep_case("scalar_misplaced", golden_path=None)
            case["values"]["tensors"][0]["alignment"] = 16
            sweep_path.write_text(json.dumps({"version": 1, "cases": [case]}))

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "case 'scalar_misplaced' is missing placeholder value for 'alignment.0'",
                completed.stderr,
            )

    def test_sweep_plain_tensor_scoped_placeholder_resolves_from_its_own_entry(
        self,
    ) -> None:
        # A non-dotted, non-whitelisted field name (e.g. alignment, as used by
        # develop's SdpaBackward bundles) placed directly in a tensor's own
        # values.tensors[] entry must resolve -- the C++ loader has no
        # whitelist on which per-tensor field names are allowed.
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_path, sweep_path = self.write_sweep_bundle(
                root, Path("quick/TestOp/Topology"), []
            )
            template = json.loads(template_path.read_text())
            template["tensors"][0]["alignment"] = "${case.alignment}"
            template_path.write_text(json.dumps(template))
            case = self.sweep_case("plain", golden_path=None)
            case["values"]["tensors"][0]["alignment"] = 16
            sweep_path.write_text(json.dumps({"version": 1, "cases": [case]}))

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)

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

    def run_sweep_with_metadata(
        self, metadata: dict[str, object], *, with_golden: bool
    ) -> subprocess.CompletedProcess[str]:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, sweep_path = self.write_sweep_bundle(
                root,
                Path("quick/TestOp/Topology"),
                [
                    self.sweep_case(
                        "small_fp32",
                        metadata=metadata,
                        golden_path="" if with_golden else None,
                    )
                ],
            )
            if with_golden:
                self.write_sweep_golden_case(
                    sweep_path.parent,
                    "small_fp32",
                    input_bytes=self.default_bytes("float"),
                    output_bytes=self.default_bytes("float"),
                )
            return self.run_verifier(root)

    def test_sweep_case_reference_flag_must_be_boolean(self) -> None:
        for flag in ("no_cpu_reference", "no_gpu_reference"):
            for bad_value in ("yes", 1, None):
                with self.subTest(flag=flag, value=bad_value):
                    completed = self.run_sweep_with_metadata(
                        {
                            "generator": "manual",
                            "reference_source": "manual",
                            flag: bad_value,
                        },
                        with_golden=True,
                    )

                    self.assertEqual(completed.returncode, 1, completed.stdout)
                    self.assertIn(
                        f"case 'small_fp32' metadata {flag} must be a boolean",
                        completed.stderr,
                    )

    def test_sweep_case_boolean_reference_flags_pass(self) -> None:
        completed = self.run_sweep_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "no_cpu_reference": True,
                "no_gpu_reference": False,
            },
            with_golden=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_sweep_case_disabling_both_references_requires_golden(self) -> None:
        completed = self.run_sweep_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "no_cpu_reference": True,
                "no_gpu_reference": True,
            },
            with_golden=False,
        )

        self.assertEqual(completed.returncode, 1, completed.stdout)
        self.assertIn("case 'small_fp32' disables both references", completed.stderr)
        self.assertIn("declares no golden data", completed.stderr)

    def test_sweep_case_disabling_both_references_with_golden_path_passes(
        self,
    ) -> None:
        completed = self.run_sweep_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "no_cpu_reference": True,
                "no_gpu_reference": True,
            },
            with_golden=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_sweep_case_disabling_both_references_with_golden_path_only_passes(
        self,
    ) -> None:
        # golden.path alone declares golden data, even if the file is not pulled.
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_sweep_bundle(
                root,
                Path("quick/TestOp/Topology"),
                [
                    self.sweep_case(
                        "small_fp32",
                        metadata={
                            "generator": "manual",
                            "reference_source": "manual",
                            "no_cpu_reference": True,
                            "no_gpu_reference": True,
                        },
                    )
                ],
            )

            completed = self.run_verifier(root)

            self.assertNotIn("disables both references", completed.stderr)
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_sweep_case_disabling_one_reference_without_golden_passes(self) -> None:
        for flag in ("no_cpu_reference", "no_gpu_reference"):
            with self.subTest(flag=flag):
                completed = self.run_sweep_with_metadata(
                    {"generator": "manual", "reference_source": "manual", flag: True},
                    with_golden=False,
                )

                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_sweep_case_non_boolean_flag_does_not_double_report(self) -> None:
        completed = self.run_sweep_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "no_cpu_reference": True,
                "no_gpu_reference": "yes",
            },
            with_golden=False,
        )

        self.assertEqual(completed.returncode, 1, completed.stdout)
        self.assertIn("no_gpu_reference must be a boolean", completed.stderr)
        self.assertNotIn("disables both references", completed.stderr)

    def test_sweep_case_expected_rejection_must_be_boolean(self) -> None:
        for bad_value in ("yes", 1, None):
            with self.subTest(value=bad_value):
                completed = self.run_sweep_with_metadata(
                    {
                        "generator": "manual",
                        "reference_source": "manual",
                        "expected_rejection": bad_value,
                    },
                    with_golden=False,
                )

                self.assertEqual(completed.returncode, 1, completed.stdout)
                self.assertIn(
                    "case 'small_fp32' metadata expected_rejection must be a boolean",
                    completed.stderr,
                )

    def test_sweep_case_non_boolean_expected_rejection_does_not_double_report(
        self,
    ) -> None:
        completed = self.run_sweep_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "expected_rejection": "yes",
            },
            with_golden=True,
        )

        self.assertEqual(completed.returncode, 1, completed.stdout)
        self.assertIn("expected_rejection must be a boolean", completed.stderr)
        self.assertNotIn("declares golden data", completed.stderr)

    def test_sweep_case_expected_rejection_without_golden_passes(self) -> None:
        completed = self.run_sweep_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "expected_rejection": True,
            },
            with_golden=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_sweep_case_expected_rejection_with_golden_path_fails(self) -> None:
        completed = self.run_sweep_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "expected_rejection": True,
            },
            with_golden=True,
        )

        self.assertEqual(completed.returncode, 1, completed.stdout)
        self.assertIn(
            "case 'small_fp32' is an expected_rejection bundle but declares golden data",
            completed.stderr,
        )

    def test_sweep_case_expected_rejection_false_with_golden_passes(self) -> None:
        completed = self.run_sweep_with_metadata(
            {
                "generator": "manual",
                "reference_source": "manual",
                "expected_rejection": False,
            },
            with_golden=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

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

    def write_superset_sweep_bundle(
        self, root: Path, absent_case: dict[str, object], version: int = 2
    ) -> Path:
        """A template with an optional sink tensor 2, one case that has it and one
        that marks it absent."""
        template_path, sweep_path = self.write_sweep_bundle(
            root, Path("quick/TestOp/Topology"), []
        )
        template = json.loads(template_path.read_text())
        template["nodes"][0]["inputs"]["sink_tensor_uid"] = 2
        template["tensors"].append(
            {
                "uid": 2,
                "name": "sink",
                "dims": "${case.dims}",
                "strides": "${case.strides}",
                "data_type": "${case.data_type}",
                "virtual": False,
            }
        )
        template_path.write_text(json.dumps(template))
        full = self.sweep_case(
            "with_sink",
            tensor_uids=(0, 1, 2),
            golden_path=None,
        )
        sweep_path.write_text(
            json.dumps({"version": version, "cases": [full, absent_case]})
        )
        return sweep_path

    def absent_sink_case(self, **overrides: object) -> dict[str, object]:
        case = self.sweep_case(
            "no_sink",
            tensor_uids=(0, 1),
            metadata={
                "format_version": 2,
                "generator": "manual",
                "reference_source": "manual",
            },
            golden_path=None,
        )
        case["absent_tensors"] = [2]
        case.update(overrides)
        return case

    def test_sweep_case_with_absent_tensor_needs_no_values_for_it(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_superset_sweep_bundle(root, self.absent_sink_case())

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("0 error(s), 0 warning(s)", completed.stderr)

    def test_absent_tensor_with_a_tensor_scoped_placeholder_needs_no_value(
        self,
    ) -> None:
        """A placeholder in a non-shape field of an absent tensor (here its
        runtime `value`) is never expanded, so the absent case needs no value
        for it; the case that keeps the tensor still does."""
        for keep_value, returncode in ((True, 0), (False, 1)):
            with self.subTest(keep_value=keep_value), TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                sweep_path = self.write_superset_sweep_bundle(
                    root, self.absent_sink_case()
                )
                template_path = sweep_path.parent / "graph.template.json"
                template = json.loads(template_path.read_text())
                template["tensors"][-1]["value"] = "${case.sink_value}"
                template_path.write_text(json.dumps(template))
                sweep = json.loads(sweep_path.read_text())
                if keep_value:
                    sweep["cases"][0]["values"]["sink_value"] = 0.5
                sweep_path.write_text(json.dumps(sweep))

                completed = self.run_verifier(root)

                self.assertEqual(completed.returncode, returncode, completed.stderr)
                self.assertNotIn("case 'no_sink' is missing", completed.stderr)

    def test_sweep_case_with_absent_tensor_the_loader_refuses_fails(self) -> None:
        for version, overrides, message in (
            (1, {}, "absent_tensors needs sweep version 2, not 1"),
            (2, {"absent_tensors": [7]}, "7 is not a tensor of the template"),
            (
                2,
                {"golden": {"path": "golden/no_sink/tensors.dvc"}},
                "absent_tensors and golden data do not mix",
            ),
        ):
            with self.subTest(message=message), TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                self.write_superset_sweep_bundle(
                    root, self.absent_sink_case(**overrides), version
                )

                completed = self.run_verifier(root)

                self.assertEqual(completed.returncode, 1)
                self.assertIn(
                    f"case 'no_sink' absent_tensors: {message}", completed.stderr
                )

    def test_sweep_values_entry_for_an_absent_tensor_is_one_error(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case = self.absent_sink_case()
            case["values"]["tensors"].append({"uid": 2})
            self.write_superset_sweep_bundle(root, case)

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "case 'no_sink' absent_tensors: 2 has a values.tensors entry",
                completed.stderr,
            )
            self.assertIn("1 error(s), 0 warning(s)", completed.stderr)

    def test_sweep_version_must_be_1_or_2(self) -> None:
        for version, returncode in ((1, 0), (2, 0), (3, 1), (0, 1), (2.0, 1)):
            with self.subTest(version=version), TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                _, sweep_path = self.write_sweep_bundle(
                    root, Path("quick/TestOp/Topology"), []
                )
                case = self.sweep_case("small_fp32", golden_path=None)
                sweep_path.write_text(json.dumps({"version": version, "cases": [case]}))

                completed = self.run_verifier(root)

                self.assertEqual(completed.returncode, returncode, completed.stderr)
                if returncode:
                    self.assertIn(
                        "version is required and must be 1 or 2", completed.stderr
                    )

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

    RAGGED_BUNDLE_DIR = Path("quick/SdpaFwd/bshd/fp32/hd2_ragged/Small")
    RAGGED_LOGICAL_DIMS = (2, 1, 4, 2)
    RAGGED_BSHD_STRIDES = (8, 2, 2, 1)
    NON_UNIFORM_OFFSETS = (0, 3, 5)

    def write_ragged_bundle(
        self,
        root: Path,
        *,
        output_bytes: bytes,
        output_dtype: str = "float",
        offsets: tuple[int, ...] | None = None,
        offset_dtype: str = "int32",
        offset_uid_in_graph: int = 10,
        multiplier: int | None = 2,
        dense_output_bytes: bytes | None = None,
        write_offset_tensor: bool = True,
        write_tensor_manifest: bool = False,
    ) -> Path:
        """Writes x (uid 0) -> ragged y (uid 1) + dense stats (uid 2), offsets uid 10."""
        bundle_dir = root / self.RAGGED_BUNDLE_DIR
        bundle_dir.mkdir(parents=True)
        name = bundle_dir.name
        offsets = offsets if offsets is not None else self.NON_UNIFORM_OFFSETS

        ragged_output = {
            "uid": 1,
            "dims": list(self.RAGGED_LOGICAL_DIMS),
            "strides": list(self.RAGGED_BSHD_STRIDES),
            "data_type": output_dtype,
            "virtual": False,
            "ragged_offset_tensor_uid": offset_uid_in_graph,
        }
        if multiplier is not None:
            ragged_output["ragged_offset_multiplier"] = multiplier

        graph = {
            "nodes": [{"outputs": {"y_tensor_uid": 1, "stats_tensor_uid": 2}}],
            "tensors": [
                {
                    "uid": 0,
                    "dims": [1],
                    "strides": [1],
                    "data_type": "float",
                    "virtual": False,
                },
                ragged_output,
                {
                    "uid": 2,
                    "dims": [1],
                    "strides": [1],
                    "data_type": "float",
                    "virtual": False,
                },
                {
                    "uid": 10,
                    "dims": [len(offsets), 1, 1, 1],
                    "strides": [1, 1, 1, 1],
                    "data_type": offset_dtype,
                    "virtual": False,
                },
            ],
            "io_data_type": output_dtype,
            "compute_data_type": "float",
            "intermediate_data_type": "float",
            "name": "",
        }
        graph_path = bundle_dir / f"{name}.json"
        graph_path.write_text(json.dumps(graph))

        offset_format = "<i" if offset_dtype == "int32" else "<q"
        payloads = {
            0: self.default_bytes("float"),
            1: output_bytes,
            2: (
                dense_output_bytes
                if dense_output_bytes is not None
                else self.default_bytes("float")
            ),
        }
        if write_offset_tensor:
            payloads[10] = b"".join(struct.pack(offset_format, o) for o in offsets)
        for uid, payload in payloads.items():
            (bundle_dir / f"{name}.tensor{uid}.bin").write_bytes(payload)

        if write_tensor_manifest:
            (bundle_dir / f"{name}.tensors.dvc").write_text(
                "outs:\n"
                + "".join(f"- path: {name}.tensor{uid}.bin\n" for uid in (0, 1, 2, 10))
            )
        (bundle_dir / f"{name}.meta.json").write_text(
            json.dumps({"generator": "manual", "reference_source": "manual"})
        )
        return graph_path

    def packed_element_count(self, multiplier: int = 2) -> int:
        return self.NON_UNIFORM_OFFSETS[-1] * multiplier

    def test_ragged_output_sized_by_non_uniform_offsets_passes(self) -> None:
        for offset_dtype in ("int32", "int64"):
            with self.subTest(offset_dtype=offset_dtype):
                with TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    self.write_ragged_bundle(
                        root,
                        output_bytes=self.filled_bytes(
                            "float", self.packed_element_count()
                        ),
                        offset_dtype=offset_dtype,
                    )

                    completed = self.run_verifier(root)

                    self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_ragged_output_with_padded_size_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            padded_elements = verify_golden_bundles.element_space(
                self.RAGGED_LOGICAL_DIMS, self.RAGGED_BSHD_STRIDES
            )
            self.write_ragged_bundle(
                root, output_bytes=self.filled_bytes("float", padded_elements)
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("tensor uid 1", completed.stderr)
            self.assertIn(
                f"graph expects {self.packed_element_count() * 4} bytes",
                completed.stderr,
            )

    def test_ragged_output_missing_multiplier_defaults_to_one(self) -> None:
        # Element offsets 0, 6, 10 are whole tokens of the 2-element sequence stride.
        for elements, expected_returncode in (
            (self.packed_element_count(multiplier=2), 0),
            (2 * self.packed_element_count(multiplier=2), 1),
        ):
            with self.subTest(elements=elements):
                with TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    self.write_ragged_bundle(
                        root,
                        output_bytes=self.filled_bytes("float", elements),
                        offsets=(0, 6, 10),
                        multiplier=None,
                    )

                    completed = self.run_verifier(root)

                    self.assertEqual(
                        completed.returncode, expected_returncode, completed.stderr
                    )

    def test_ragged_output_nan_sentinel_passes(self) -> None:
        cases = [
            ("float", struct.pack("<f", float("nan")), self.default_bytes("float")),
            ("half", struct.pack("<H", 0x7E00), self.default_bytes("half")),
            ("bfloat16", struct.pack("<H", 0x7FC0), self.default_bytes("bfloat16")),
        ]
        for data_type, nan_bytes, finite_bytes in cases:
            with self.subTest(data_type=data_type):
                with TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    self.write_ragged_bundle(
                        root,
                        output_dtype=data_type,
                        output_bytes=nan_bytes
                        + finite_bytes * (self.packed_element_count() - 1),
                    )

                    completed = self.run_verifier(root)

                    self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_ragged_output_inf_fails(self) -> None:
        cases = [
            ("float", struct.pack("<f", float("-inf")), self.default_bytes("float")),
            ("half", struct.pack("<H", 0x7C00), self.default_bytes("half")),
            ("bfloat16", struct.pack("<H", 0xFF80), self.default_bytes("bfloat16")),
        ]
        for data_type, inf_bytes, finite_bytes in cases:
            with self.subTest(data_type=data_type):
                with TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    self.write_ragged_bundle(
                        root,
                        output_dtype=data_type,
                        output_bytes=finite_bytes
                        + inf_bytes
                        + finite_bytes * (self.packed_element_count() - 2),
                    )

                    completed = self.run_verifier(root)

                    self.assertEqual(completed.returncode, 1)
                    self.assertIn("tensor uid 1", completed.stderr)
                    self.assertIn(
                        "ragged output tensor contains Inf at element index 1",
                        completed.stderr,
                    )

    def test_dense_output_nan_fails_beside_ragged_output(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_ragged_bundle(
                root,
                output_bytes=self.filled_bytes("float", self.packed_element_count()),
                dense_output_bytes=struct.pack("<f", float("nan")),
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn("tensor uid 2", completed.stderr)
            self.assertIn("output tensor contains NaN/Inf", completed.stderr)

    def test_ragged_missing_offset_tensor_file(self) -> None:
        for require_data, expected_returncode in ((False, 0), (True, 1)):
            with self.subTest(require_data=require_data):
                with TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    self.write_ragged_bundle(
                        root,
                        output_bytes=self.filled_bytes(
                            "float", self.packed_element_count()
                        ),
                        write_offset_tensor=False,
                        write_tensor_manifest=True,
                    )

                    completed = self.run_verifier(root, require_data=require_data)

                    self.assertEqual(
                        completed.returncode, expected_returncode, completed.stderr
                    )
                    self.assertIn("tensor uid 10", completed.stderr)
                    self.assertIn(".tensor10.bin", completed.stderr)

    def test_ragged_offset_uid_not_declared_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_ragged_bundle(
                root,
                output_bytes=self.filled_bytes("float", self.packed_element_count()),
                offset_uid_in_graph=42,
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "ragged_offset_tensor_uid 42 is not a declared tensor",
                completed.stderr,
            )

    def test_ragged_multiplier_below_one_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_ragged_bundle(
                root,
                output_bytes=self.filled_bytes("float", self.packed_element_count()),
                multiplier=0,
            )

            completed = self.run_verifier(root)

            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "ragged_offset_multiplier must be an integer >= 1", completed.stderr
            )

    SDPA_DIMS = (2, 1, 4, 2)
    SDPA_STRIDES = (8, 2, 2, 1)

    def write_sdpa_ragged_bundle(
        self,
        root: Path,
        *,
        elements: int,
        offsets: tuple[int, ...],
        lengths: tuple[int, ...] | None,
        dims: tuple[int, ...] = SDPA_DIMS,
        strides: tuple[int, ...] = SDPA_STRIDES,
        multiplier: int = 1,
        role_field: str = "q_tensor_uid",
        length_field: str = "seq_len_q_tensor_uid",
        aux_stride: int = 1,
    ) -> Path:
        """Writes a ragged tensor (uid 0) named by `role_field` of an SDPA backward node
        whose `length_field` names the seq_len tensor uid 20 (absent: no seq_len). The
        offset and seq_len tensors have batch stride `aux_stride`, poison between values.
        """
        bundle_dir = root / self.RAGGED_BUNDLE_DIR
        bundle_dir.mkdir(parents=True)
        name = bundle_dir.name

        def plain(
            uid: int, data_type: str, shape: list[int], stride: int = 1
        ) -> dict[str, object]:
            return {
                "uid": uid,
                "dims": shape,
                "strides": [stride] + [1] * (len(shape) - 1),
                "data_type": data_type,
                "virtual": False,
            }

        inputs: dict[str, object] = {role_field: 0}
        tensors = [
            {
                **plain(0, "float", list(dims)),
                "strides": list(strides),
                "ragged_offset_tensor_uid": 10,
                "ragged_offset_multiplier": multiplier,
            },
            plain(1, "float", [1]),
            plain(10, "int32", [len(offsets), 1, 1, 1], aux_stride),
        ]

        def pack_strided(values: tuple[int, ...]) -> bytes:
            buffer = [-1] * ((len(values) - 1) * aux_stride + 1)
            buffer[::aux_stride] = values
            return struct.pack(f"<{len(buffer)}i", *buffer)

        payloads = {
            0: self.filled_bytes("float", elements),
            1: self.default_bytes("float"),
            10: pack_strided(offsets),
        }
        if lengths is not None:
            inputs[length_field] = 20
            tensors.append(plain(20, "int32", [len(lengths), 1, 1, 1], aux_stride))
            payloads[20] = pack_strided(lengths)
        graph = {
            "nodes": [
                {
                    "type": "SdpaBackwardAttributes",
                    "inputs": inputs,
                    "outputs": {"dq_tensor_uid": 1},
                }
            ],
            "tensors": tensors,
            "io_data_type": "float",
            "compute_data_type": "float",
            "intermediate_data_type": "float",
            "name": "",
        }
        graph_path = bundle_dir / f"{name}.json"
        graph_path.write_text(json.dumps(graph))
        for uid, payload in payloads.items():
            (bundle_dir / f"{name}.tensor{uid}.bin").write_bytes(payload)
        (bundle_dir / f"{name}.meta.json").write_text(
            json.dumps({"generator": "manual", "reference_source": "manual"})
        )
        return graph_path

    def verify_sdpa_ragged(self, **bundle: object) -> subprocess.CompletedProcess[str]:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_sdpa_ragged_bundle(root, **bundle)
            return self.run_verifier(root)

    def test_ragged_file_holds_exactly_the_element_space(self) -> None:
        # (offsets, lengths, elementSpace): shifted origin (L8), non-monotonic
        # origins (L9), a trailing gap that the file does not hold (L14), and a zero
        # length beside a non-zero origin (L15).
        for offsets, lengths, space in (
            ((4, 8, 14), (2, 3), 14),
            ((10, 0, 0), (2, 3), 14),
            ((0, 4, 12), (2, 3), 10),
            ((6, 0, 0), (0, 3), 6),
        ):
            with self.subTest(offsets=offsets, lengths=lengths):
                passed = self.verify_sdpa_ragged(
                    elements=space, offsets=offsets, lengths=lengths
                )
                self.assertEqual(passed.returncode, 0, passed.stderr)
                for wrong in (space - 1, space + 1):
                    failed = self.verify_sdpa_ragged(
                        elements=wrong, offsets=offsets, lengths=lengths
                    )
                    self.assertEqual(failed.returncode, 1)
                    self.assertIn(f"graph expects {space * 4} bytes", failed.stderr)

    def test_ragged_strided_offsets_and_seq_len_are_walked_by_stride(self) -> None:
        # L14 with batch stride 2 on both aux tensors, -1 in the gaps: the files
        # hold the strided element space and the size check sees the packed values.
        passed = self.verify_sdpa_ragged(
            elements=10, offsets=(0, 4, 12), lengths=(2, 3), aux_stride=2
        )
        self.assertEqual(passed.returncode, 0, passed.stderr)

        failed = self.verify_sdpa_ragged(
            elements=11, offsets=(0, 4, 12), lengths=(2, 3), aux_stride=2
        )
        self.assertEqual(failed.returncode, 1)
        self.assertIn("graph expects 40 bytes", failed.stderr)

    def test_ragged_file_sized_to_the_last_offset_fails(self) -> None:
        failed = self.verify_sdpa_ragged(
            elements=12, offsets=(0, 4, 12), lengths=(2, 3)
        )

        self.assertEqual(failed.returncode, 1)
        self.assertIn("file has 48 bytes but graph expects 40 bytes", failed.stderr)
        self.assertIn("ragged elementSpace=10", failed.stderr)

    def test_ragged_tensor_must_have_rank_4(self) -> None:
        failed = self.verify_sdpa_ragged(
            elements=16,
            offsets=(0, 4, 12),
            lengths=(2, 3),
            dims=(2, 4, 8),
            strides=(32, 8, 1),
        )

        self.assertEqual(failed.returncode, 1)
        self.assertIn("ragged tensors must have rank 4", failed.stderr)
        self.assertNotIn("Traceback", failed.stderr)

    def test_ragged_offset_tensor_must_hold_exactly_b_plus_one_values(self) -> None:
        failed = self.verify_sdpa_ragged(
            elements=10, offsets=(0, 4, 12, 12), lengths=(2, 3)
        )

        self.assertEqual(failed.returncode, 1)
        self.assertIn("must have shape [B + 1, 1, 1, 1]", failed.stderr)

    def test_ragged_seq_len_tensor_must_hold_exactly_b_values(self) -> None:
        failed = self.verify_sdpa_ragged(
            elements=10, offsets=(0, 4, 12), lengths=(2, 3, 1)
        )

        self.assertEqual(failed.returncode, 1)
        self.assertIn("must have shape [B, 1, 1, 1]", failed.stderr)

    def test_ragged_multiplier_scales_the_origins(self) -> None:
        # Token offsets 5, 0 with multiplier 2 are the element origins 10, 0 (L9m).
        passed = self.verify_sdpa_ragged(
            elements=14, offsets=(5, 0, 7), lengths=(2, 3), multiplier=2
        )

        self.assertEqual(passed.returncode, 0, passed.stderr)

    def test_ragged_lengths_come_from_the_roles_seq_len(self) -> None:
        # K takes seq_len_kv. With only seq_len_q named, K has no lengths, so its
        # blocks (2 and 4 rows, clipped to S_max = 4) size it: elementSpace 12.
        offsets = (0, 4, 12)
        role = {"role_field": "k_tensor_uid"}
        with_kv = self.verify_sdpa_ragged(
            elements=10,
            offsets=offsets,
            lengths=(2, 3),
            length_field="seq_len_kv_tensor_uid",
            **role,
        )
        with_q_only = self.verify_sdpa_ragged(
            elements=12,
            offsets=offsets,
            lengths=(2, 3),
            length_field="seq_len_q_tensor_uid",
            **role,
        )

        self.assertEqual(with_kv.returncode, 0, with_kv.stderr)
        self.assertEqual(with_q_only.returncode, 0, with_q_only.stderr)

    def test_ragged_length_uid_roles_per_node(self) -> None:
        q_roles = {
            "SdpaAttributes": [
                ("inputs", "q_tensor_uid"),
                ("outputs", "o_tensor_uid"),
                ("outputs", "stats_tensor_uid"),
                ("outputs", "max_tensor_uid"),
                ("outputs", "sum_exp_tensor_uid"),
            ],
            "SdpaBackwardAttributes": [
                ("inputs", "q_tensor_uid"),
                ("inputs", "o_tensor_uid"),
                ("inputs", "do_tensor_uid"),
                ("outputs", "dq_tensor_uid"),
                ("inputs", "stats_tensor_uid"),
            ],
        }
        kv_roles = {
            "SdpaAttributes": [
                ("inputs", "k_tensor_uid"),
                ("inputs", "v_tensor_uid"),
            ],
            "SdpaBackwardAttributes": [
                ("inputs", "k_tensor_uid"),
                ("inputs", "v_tensor_uid"),
                ("outputs", "dk_tensor_uid"),
                ("outputs", "dv_tensor_uid"),
            ],
        }
        for node_type in q_roles:
            with self.subTest(node_type=node_type):
                node = {
                    "type": node_type,
                    "inputs": {
                        "seq_len_q_tensor_uid": 100,
                        "seq_len_kv_tensor_uid": 101,
                    },
                    "outputs": {},
                }
                expected = {}
                for uid, (section, field) in enumerate(
                    q_roles[node_type] + kv_roles[node_type]
                ):
                    node[section][field] = uid
                    is_q = (section, field) in q_roles[node_type]
                    expected[uid] = 100 if is_q else 101
                self.assertEqual(
                    verify_golden_bundles.extract_ragged_length_uids([node]), expected
                )
                del node["inputs"]["seq_len_kv_tensor_uid"]
                self.assertEqual(
                    set(verify_golden_bundles.extract_ragged_length_uids([node])),
                    {uid for uid, length in expected.items() if length == 100},
                )

    def test_ragged_tensor_bound_to_two_seq_len_tensors_is_an_error(self) -> None:
        node = {
            "type": "SdpaAttributes",
            "inputs": {
                "q_tensor_uid": 1,
                "k_tensor_uid": 1,
                "seq_len_q_tensor_uid": 100,
                "seq_len_kv_tensor_uid": 101,
            },
            "outputs": {},
        }
        with self.assertRaisesRegex(ValueError, "two seq_len tensors"):
            verify_golden_bundles.extract_ragged_length_uids([node])
        node["inputs"]["seq_len_kv_tensor_uid"] = 100
        self.assertEqual(
            verify_golden_bundles.extract_ragged_length_uids([node]), {1: 100}
        )

    CONFLICTING_SEQ_LEN_INPUTS = {
        "q_tensor_uid": 0,
        "k_tensor_uid": 0,
        "seq_len_q_tensor_uid": 20,
        "seq_len_kv_tensor_uid": 21,
    }

    def test_graph_bundle_with_conflicting_seq_len_binding_reports_error(self) -> None:
        with TemporaryDirectory() as tmpdir:
            graph_path = self.write_sdpa_ragged_bundle(
                Path(tmpdir), elements=14, offsets=(0, 4, 8), lengths=(2, 3)
            )
            graph = json.loads(graph_path.read_text())
            graph["nodes"][0]["inputs"] = dict(self.CONFLICTING_SEQ_LEN_INPUTS)
            graph_path.write_text(json.dumps(graph))

            result = verify_golden_bundles.VerificationResult()
            verify_golden_bundles.validate_graph_bundle(
                graph_path, result, "Smoke", False
            )

            messages = [d.message for d in result.diagnostics if d.severity == "error"]
            self.assertEqual(len(messages), 1, messages)
            self.assertIn("invalid ragged seq_len binding", messages[0])
            self.assertIn("two seq_len tensors", messages[0])

    def test_sweep_bundle_with_conflicting_seq_len_binding_reports_error(self) -> None:
        with TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "Smoke" / "Sdpa" / "Ragged"
            bundle_dir.mkdir(parents=True)
            template_path = bundle_dir / "graph.template.json"
            template_path.write_text(
                json.dumps(
                    {
                        "nodes": [
                            {
                                "type": "SdpaAttributes",
                                "inputs": dict(self.CONFLICTING_SEQ_LEN_INPUTS),
                                "outputs": {"o_tensor_uid": 1},
                            }
                        ],
                        "tensors": [{"uid": 0}],
                    }
                )
            )
            sweep_path = bundle_dir / "sweep.json"
            sweep_path.write_text(json.dumps({"version": 1, "cases": []}))

            result = verify_golden_bundles.VerificationResult()
            verify_golden_bundles.validate_sweep_bundle(
                template_path, sweep_path, result, "Smoke", False
            )

            messages = [d.message for d in result.diagnostics if d.severity == "error"]
            self.assertEqual(len(messages), 1, messages)
            self.assertIn("invalid ragged seq_len binding", messages[0])
            self.assertIn("two seq_len tensors", messages[0])

    def test_ragged_layout_violations_name_the_violated_rule(self) -> None:
        cases = (
            ("overlap rule", {"offsets": (0, 4, 8), "lengths": (3, 3)}),
            (
                "sorted-stride rule",
                {
                    "dims": (2, 2, 4, 2),
                    "strides": (16, 8, 2, 1),
                    "offsets": (0, 4, 10),
                    "lengths": (2, 3),
                },
            ),
            ("whole-token origin rule", {"offsets": (1, 4, 10), "lengths": (2, 3)}),
            ("length range rule", {"offsets": (0, 4, 10), "lengths": (2, 5)}),
            (
                "negative-stride rule",
                {"strides": (8, 2, -2, 1), "offsets": (0, 4, 10), "lengths": (2, 3)},
            ),
            ("monotonic rule", {"offsets": (0, 8, 4), "lengths": None}),
        )
        for rule, bundle in cases:
            with self.subTest(rule=rule):
                failed = self.verify_sdpa_ragged(elements=10, **bundle)

                self.assertEqual(failed.returncode, 1)
                self.assertIn(rule, failed.stderr)

    def test_ragged_broadcast_head_axis_is_exempt_from_the_stride_rule(self) -> None:
        # L12: H has stride 0, so the 4 logical rows of 2 heads share 10 elements.
        passed = self.verify_sdpa_ragged(
            elements=10,
            dims=(2, 2, 3, 2),
            strides=(0, 0, 2, 1),
            offsets=(0, 4, 10),
            lengths=None,
        )

        self.assertEqual(passed.returncode, 0, passed.stderr)


class TestRaggedElementSpace(unittest.TestCase):
    """ragged_element_space and the offset/seq_len reader, called directly."""

    DIMS = [2, 1, 4, 2]
    STRIDES = [8, 2, 2, 1]

    def space(self, offsets, lengths, dims=None, strides=None, multiplier=1) -> int:
        return verify_golden_bundles.ragged_element_space(
            dims or self.DIMS, strides or self.STRIDES, multiplier, offsets, lengths
        )

    def test_offset_derived_length_is_clamped_to_s_max(self) -> None:
        # Without seq_len, the first block spans 6 rows but S_max is 4: rows 0..3
        # are live, so elementSpace = 1 + 3 * 2 + 1.
        self.assertEqual(self.space([0, 12, 12], None), 8)

    def test_tensor_with_no_rows_has_no_elements(self) -> None:
        self.assertEqual(self.space([0, 0, 0], [0, 0]), 0)
        self.assertEqual(self.space([6, 6, 6], None), 0)

    def test_sorted_stride_rule_accepts_a_stride_equal_to_the_span(self) -> None:
        # Rows 0..4 of stride 2 span 10 elements: the head stride may be 10, not 9.
        dims = [2, 2, 4, 2]
        self.assertEqual(
            self.space([0, 4, 10], [2, 3], dims, [20, 10, 2, 1]), 1 + 4 * 2 + 10 + 1
        )
        with self.assertRaisesRegex(ValueError, "sorted-stride rule"):
            self.space([0, 4, 10], [2, 3], dims, [20, 9, 2, 1])

    def test_negative_length_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "length range rule"):
            self.space([0, 4, 10], [-1, 2])

    def test_negative_origin_is_rejected(self) -> None:
        # -2 is a whole number of 2-element tokens, so only the sign is wrong.
        with self.assertRaisesRegex(ValueError, "whole-token origin rule"):
            self.space([-2, 4, 10], [2, 3])

    def test_zero_extent_non_sequence_axis_has_no_elements(self) -> None:
        # Blocks hold rows, but a zero-extent head or feature axis leaves no element.
        self.assertEqual(
            self.space([0, 4, 10], [2, 3], dims=[2, 0, 4, 2]),
            0,
        )
        self.assertEqual(
            self.space([0, 4, 10], [2, 3], dims=[2, 1, 4, 0]),
            0,
        )

    def test_zero_extent_axis_is_empty_whatever_the_other_strides(self) -> None:
        # Head and sequence strides that would alias are never checked on an empty
        # tensor.
        self.assertEqual(
            self.space([0, 4, 10], [3, 3], dims=[2, 0, 4, 2], strides=[8, 2, 1, 1]), 0
        )

    def test_batch_axis_stride_is_unused_so_negative_is_accepted(self) -> None:
        self.assertEqual(
            self.space([0, 4, 10], [2, 3], strides=[-8, 2, 2, 1]),
            self.space([0, 4, 10], [2, 3]),
        )

    def test_zero_sequence_stride_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "negative-stride rule"):
            self.space([0, 4, 10], [2, 3], strides=[8, 2, 0, 1])

    def read_aux(
        self,
        dims,
        strides,
        data_type="int32",
        values=(0, 4, 10),
        raw=None,
        count=3,
        count_label="B + 1",
    ):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "offsets.bin"
            path.write_bytes(raw if raw is not None else struct.pack("<3i", *values))
            result = verify_golden_bundles.VerificationResult()
            read = verify_golden_bundles.read_ragged_tensor_values(
                10,
                0,
                "ragged_offset_tensor_uid",
                count,
                count_label,
                {10: {"dims": dims, "strides": strides, "data_type": data_type}},
                lambda uid: path,
                False,
                True,
                result,
            )
            return read, [diagnostic.message for diagnostic in result.diagnostics]

    def test_aux_tensor_reads_when_well_formed(self) -> None:
        read, messages = self.read_aux([3, 1, 1, 1], [1, 1, 1, 1])
        self.assertEqual((read, messages), ([0, 4, 10], []))

    def test_aux_tensor_must_be_rank_4(self) -> None:
        read, messages = self.read_aux([3, 1, 1], [1, 1, 1])
        self.assertIsNone(read)
        self.assertEqual(len(messages), 1)
        self.assertIn("must be rank 4", messages[0])

    def test_aux_tensor_must_have_the_batch_axis_hold_the_values(self) -> None:
        # Right element count and rank, but the values lie on the head axis: the
        # batch-stride read would run past the file.
        for dims in ([1, 3, 1, 1], [3, 1, 1, 2], [4, 1, 1, 1]):
            read, messages = self.read_aux(dims, [3, 1, 1, 1])
            self.assertIsNone(read, dims)
            self.assertEqual(len(messages), 1, dims)
            self.assertIn("must have shape [B + 1, 1, 1, 1]", messages[0])

    def test_seq_len_aux_must_have_shape_b(self) -> None:
        read, messages = self.read_aux(
            [3, 1, 1, 1], [1, 1, 1, 1], count=2, count_label="B"
        )
        self.assertIsNone(read)
        self.assertEqual(len(messages), 1)
        self.assertIn("must have shape [B, 1, 1, 1]", messages[0])

    def test_aux_tensor_batch_stride_must_be_positive(self) -> None:
        read, messages = self.read_aux([3, 1, 1, 1], [0, 1, 1, 1])
        self.assertIsNone(read)
        self.assertEqual(len(messages), 1)
        self.assertIn("positive batch stride", messages[0])

    def test_aux_tensor_data_type_must_be_integer(self) -> None:
        read, messages = self.read_aux([3, 1, 1, 1], [1, 1, 1, 1], data_type="float")
        self.assertIsNone(read)
        self.assertEqual(len(messages), 1)
        self.assertIn("expected int32 or int64", messages[0])


class TestDeriveAdvisory(unittest.TestCase):
    def derive(self, relative_path: str):
        result = verify_golden_bundles.VerificationResult()
        advisory = verify_golden_bundles.derive_advisory(
            Path(relative_path), result, "quick"
        )
        return advisory, result

    def messages(self, result, severity: str) -> list[str]:
        return [d.message for d in result.diagnostics if d.severity == severity]

    def test_without_tier_uses_default_tier(self) -> None:
        advisory, result = self.derive("Op/nhwc/fp32/Small/Small.json")

        self.assertIsNotNone(advisory)
        self.assertEqual(advisory.canonical_path, "quick/Op/nhwc/fp32/Small/")
        self.assertEqual(advisory.full_test_name, "quick_Op_nhwc_fp32_Small.Small")
        self.assertEqual(self.messages(result, "error"), [])
        warnings = self.messages(result, "warning")
        self.assertEqual(len(warnings), 1)
        self.assertIn("variant directories cannot be recovered", warnings[0])

    def test_without_tier_too_short_is_layout_error(self) -> None:
        advisory, result = self.derive("nhwc/fp32/Small/Small.json")

        self.assertIsNone(advisory)
        self.assertEqual(
            self.messages(result, "error"),
            [verify_golden_bundles.ADVISORY_LAYOUT_ERROR],
        )

    def test_with_tier_too_short_is_layout_error(self) -> None:
        advisory, result = self.derive("quick/Op/nhwc/Small/Small.json")

        self.assertIsNone(advisory)
        self.assertEqual(
            self.messages(result, "error"),
            [verify_golden_bundles.ADVISORY_LAYOUT_ERROR],
        )

    def test_multiple_variant_directories_are_kept(self) -> None:
        advisory, result = self.derive(
            "standard/Op/bshd/bf16/hd128/causal/Small/Small.json"
        )

        self.assertIsNotNone(advisory)
        self.assertEqual(
            advisory.canonical_path, "standard/Op/bshd/bf16/hd128/causal/Small/"
        )
        self.assertEqual(
            advisory.test_suite, "standard_Op_bshd_bf16_hd128_causal_Small"
        )
        self.assertEqual(advisory.test_case, "Small")
        self.assertFalse(result.has_errors())

    def test_variant_named_after_tier_is_kept_as_variant(self) -> None:
        advisory, result = self.derive("quick/Op/nhwc/fp32/full/Small/Small.json")

        self.assertIsNotNone(advisory)
        self.assertEqual(advisory.canonical_path, "quick/Op/nhwc/fp32/full/Small/")
        self.assertEqual(advisory.test_suite, "quick_Op_nhwc_fp32_full_Small")
        self.assertFalse(result.has_errors())

    def test_bundle_named_after_tier_is_kept_as_name(self) -> None:
        advisory, result = self.derive("quick/Op/nhwc/fp32/full/full.json")

        self.assertIsNotNone(advisory)
        self.assertEqual(advisory.canonical_path, "quick/Op/nhwc/fp32/full/")
        self.assertEqual(advisory.test_case, "full")
        self.assertFalse(result.has_errors())

    def test_tier_named_ancestor_and_variant_anchor_on_bundle_tier(self) -> None:
        advisory, result = self.derive(
            "quick/checkout/quick/Op/nhwc/fp32/full/Small/Small.json"
        )

        self.assertIsNotNone(advisory)
        self.assertEqual(advisory.full_test_name, "quick_Op_nhwc_fp32_full_Small.Small")
        self.assertFalse(result.has_errors())


if __name__ == "__main__":
    unittest.main()
