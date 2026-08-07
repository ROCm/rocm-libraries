# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from tensilelite.GenerateSummations import _read_benchmark_data


pytestmark = pytest.mark.unit


def test_benchmark_csv_preserves_size_order_and_fitting_vectors(tmp_path):
    benchmark = tmp_path / "benchmark.csv"
    benchmark.write_text(
        ' "SizeL" ,"Cij0","kernel one"\n'
        '32,10,4\n'
        '16,nan,2\n'
        '64,30,8\n',
        encoding="utf-8",
    )

    sizes, columns, maximum = _read_benchmark_data(benchmark)
    perf = (1000 * sizes) / columns["kernel one"]
    model = np.polyfit(sizes, perf, deg=1)

    np.testing.assert_array_equal(sizes, [32.0, 16.0, 64.0])
    np.testing.assert_array_equal(columns["kernel one"], [4.0, 2.0, 8.0])
    assert maximum == 30.0
    np.testing.assert_allclose(model, [0.0, 8000.0], atol=1e-10)
