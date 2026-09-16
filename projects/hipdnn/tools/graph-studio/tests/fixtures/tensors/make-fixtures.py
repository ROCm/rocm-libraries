"""Regenerate the tensor artifact fixtures with dnn-benchmarking's own writer.

Run from the dnn-benchmarking checkout so its package is importable:

    .venv/bin/python <this file> ../graph-studio/tests/fixtures/tensors

The set covers every supported encoding and includes one strided tensor, so the
reader is exercised against real element-space bytes instead of a hand-written
approximation of them.
"""

import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dnn_benchmarking.graph.tensor_info import TensorInfo
from dnn_benchmarking.execution.tensor_artifacts import write_tensor_manifest

dest_root = Path(sys.argv[1])
graph_path = Path("fixture.hipdnn.json")
graph_json = {"name": "fixture", "nodes": []}

infos = [
    TensorInfo(1, "x_f32", [2, 3], [3, 1], "float", False),
    TensorInfo(2, "x_f16", [4], [1], "half", False),
    TensorInfo(3, "x_bf16", [2, 2], [3, 1], "bfloat16", False),
    TensorInfo(4, "x_f64", [2], [1], "double", False),
    TensorInfo(5, "x_i32", [3], [1], "int32", False),
    TensorInfo(6, "x_i8", [2], [1], "int8", False),
    TensorInfo(7, "x_u8", [2], [1], "uint8", False),
    TensorInfo(9, "y", [2, 2], [2, 1], "float", False, is_output=True),
]

inputs = {
    1: np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    2: np.array([1.5, -0.5, 2.0, 0.25], dtype=np.float16),
    3: np.array([[1.5, -2.25], [0.75, 4.0]], dtype=np.float32),
    4: np.array([1e-9, -2.5], dtype=np.float64),
    5: np.array([-7, 0, 2147483647], dtype=np.int32),
    6: np.array([-128, 127], dtype=np.int8),
    7: np.array([0, 255], dtype=np.uint8),
}
outputs = {9: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
reference = {9: np.array([[1.0, 2.0], [3.0, 4.25]], dtype=np.float32)}

cases = [
    ("input", inputs, {}),
    ("output", outputs, {"provider": "hipdnn", "engine_id": 7}),
    ("reference", reference, {"provider": "pytorch", "engine_id": 0}),
]

with TemporaryDirectory() as tmp:
    for phase, tensors, extra in cases:
        manifest = write_tensor_manifest(
            Path(tmp), graph_path, graph_json, infos, tensors, phase=phase, **extra
        )
        target = dest_root / phase
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(manifest.parent, target)
        print(phase, sorted(p.name for p in target.iterdir()))
