# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only tests of the norm AOT cache keys and the torch-free batchnorm tooling."""

import subprocess
import sys

from benchmarks.common.norm_kernel_cache import cache_key
from rocke.instances.common.batchnorm2d import BatchNorm2DSpec

_COMMON = dict(
    family="batchnorm2d",
    dims={"n": 8, "c": 64, "h": 14, "w": 14},
    dtype="fp16",
    arch="gfx950",
    candidate_name="batchnorm2d_batchnorm_dyn_b256_v4",
    dynamic=True,
)


def _spec(training):
    return BatchNorm2DSpec(
        n=1, c=1, h=1, w=1, dynamic=True, training=training, save_mean_invstd=training
    )


def test_train_and_infer_binaries_get_distinct_keys():
    # one candidate name serves both modes; the kernel name carries the mode
    train = cache_key(**_COMMON, kernel_name=_spec(True).kernel_name())
    infer = cache_key(**_COMMON, kernel_name=_spec(False).kernel_name())
    assert train != infer


def test_dynamic_key_is_shape_free():
    a = cache_key(**_COMMON, kernel_name="k")
    b = cache_key(
        **{**_COMMON, "dims": {"n": 1, "c": 3, "h": 5, "w": 7}}, kernel_name="k"
    )
    assert a == b


def test_dtype_and_arch_are_part_of_the_key():
    base = cache_key(**_COMMON, kernel_name="k")
    assert cache_key(**{**_COMMON, "dtype": "bf16"}, kernel_name="k") != base
    assert cache_key(**{**_COMMON, "arch": "gfx942"}, kernel_name="k") != base


def test_batchnorm_tooling_imports_without_torch():
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"  # any import of torch raises
        "import benchmarks.common.benchmark_batchnorm2d_aot\n"
        "import benchmarks.common.norm_ro_launch\n"
        "import benchmarks.common.norm_split_launch\n"
        "import benchmarks.common.norm_kernel_cache\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        env={"PYTHONPATH": ":".join(p for p in sys.path if p)},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
