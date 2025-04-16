################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

import rrperf.optimize_weights as ow

import argparse
import dataclasses as dc
import functools
import pathlib
import pytest
import subprocess
import yaml

import time
import random

from types import SimpleNamespace as NS


@functools.cache
def rocm_gfx():
    """Return GPU architecture (gfxXXXX) for local GPU device."""
    output = None
    try:
        output = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError:
        return None

    for line in output.splitlines():
        if line.startswith("  Name:"):
            _, arch, *_ = list(map(lambda x: x.strip(), line.split()))
            if arch.startswith("gfx"):
                return arch

    return None


@pytest.mark.slow
def test_run_optimize(tmp_path_factory):
    # TODO This is a temporary fix to enable GFX12 CI
    gfx = rocm_gfx()
    if not gfx or gfx.startswith("gfx12"):
        print("FIXME: change test_optimize_weights.py to work on GFX12")
        return

    t = int(time.time())
    print(f"Random seed: {t}")
    random.seed(t)
    output_dir = tmp_path_factory.mktemp("test_run_optimize")
    run_1_dir = output_dir / "run_1"
    run_2_dir = output_dir / "run_2"
    args = [
        f"--output={run_1_dir}",
        "--generations=2",
        "--parents=2",
        "--population=3",
        "--new=1",
        "--suite=guidepost_1",
    ]
    parser = argparse.ArgumentParser()
    ow.get_args(parser)

    parsed_args = parser.parse_args(args)
    ow.run(parsed_args)

    result_file = run_1_dir / "results_1.yaml"
    assert result_file.is_file()

    with result_file.open() as f:
        data = yaml.safe_load(f)

    assert len(data) == 5
    for d in data:
        assert "hash" in d
        assert "time" in d
        assert "weights" in d

    args = args[1:] + [f"--output={run_2_dir}", f"--input={result_file}"]

    parsed_args = parser.parse_args(args)
    ow.run(parsed_args)

    result_file = run_2_dir / "results_1.yaml"
    assert result_file.is_file()

    with result_file.open() as f:
        data = yaml.safe_load(f)

    assert len(data) == 5
    for d in data:
        assert "hash" in d
        assert "time" in d
        assert "weights" in d


def mocked_run(cmd, **kwargs):
    test_yaml = """
---
client:          GEMMv00
device:          0
kernelGenerate:  0
kernelAssemble:  0
kernelExecute:   [ 11345953, 11285794, 11359553, 11373794, 11348034, 11430433,
                   11296513, 11409154, 11417954, 11372353 ]
checked:         true
correct:         true
rnorm:           2.5e-05
...
"""
    assert cmd[0] == "bin/client/rocRoller_gemm"
    yaml_file = None
    for arg in cmd:
        if arg.startswith("--yaml"):
            yaml_file = pathlib.Path(arg.split("=")[1])
    yaml_file.write_text(test_yaml)
    return NS(returncode=0, stdout=test_yaml.encode("ascii"))


def test_mocked_integration(tmp_path_factory, mocker):
    mocker.patch("subprocess.run", new=mocked_run)
    output_dir = tmp_path_factory.mktemp("test_mocked_integration")
    run_1_dir = output_dir / "run_1"
    run_2_dir = output_dir / "run_2"
    args = [
        f"--output={run_1_dir}",
        "--generations=2",
        "--parents=2",
        "--population=3",
        "--new=1",
        "--suite=guidepost_1",
    ]
    parser = argparse.ArgumentParser()
    ow.get_args(parser)

    parsed_args = parser.parse_args(args)
    ow.run(parsed_args)

    result_file = run_1_dir / "results_1.yaml"
    assert result_file.is_file()

    with result_file.open() as f:
        data = yaml.safe_load(f)

    assert len(data) == 5
    for d in data:
        assert "hash" in d
        assert "time" in d
        assert "weights" in d

    args = args[1:] + [f"--output={run_2_dir}", f"--input={result_file}"]

    parsed_args = parser.parse_args(args)
    ow.run(parsed_args)

    result_file = run_2_dir / "results_1.yaml"
    assert result_file.is_file()

    with result_file.open() as f:
        data = yaml.safe_load(f)

    assert len(data) == 5
    for d in data:
        assert "hash" in d
        assert "time" in d
        assert "weights" in d


def test_weights():
    test1 = ow.Weights()
    test2 = ow.Weights()

    assert test1.outOfRegisters >= 1e9
    assert test2.outOfRegisters >= 1e9

    # With 0% mutation rate, all of the fields will come from one of the parents.
    test3 = ow.Weights.Combine([test1, test2], mutation=0)
    for field in dc.fields(ow.Weights):
        assert getattr(test3, field.name) == getattr(test1, field.name) or getattr(
            test3, field.name
        ) == getattr(test2, field.name)

    # With 100% mutation rate, all of the float fields _will probably_ be different
    # from the parents after a combine.
    test4 = ow.Weights.Combine([test1, test2], 1.0)
    assert test4.outOfRegisters >= 1e9

    fields = {
        fld.name
        for fld in dc.fields(ow.Weights)
        if fld.type == float and fld.default_factory.is_variable
    }
    for fld in fields:
        assert getattr(test1, fld) != getattr(test4, fld)
        assert getattr(test2, fld) != getattr(test4, fld)

    test5 = ow.Weights.Combine([test1, test2])
    assert test5.outOfRegisters >= 1e9

    weights = [test1, test2, test3, test4, test5]
    hashes = list([t.short_hash for t in weights])

    # There is a small, but non-zero chance that these are not all different, in which
    # case the test would fail.
    assert len(hashes) == len(set(hashes)), hashes


def test_get_args():
    # Testing that something is required.
    parser = argparse.ArgumentParser()
    ow.get_args(parser)
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        args = parser.parse_args([])
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 2

    # Testing only the required args.
    parser = argparse.ArgumentParser()
    ow.get_args(parser)
    args = parser.parse_args(["--output=testingrequired"])
    assert args.output_dir == pathlib.Path("testingrequired")

    input_args = [
        "--output=testing123",
        "--generations=2",
        "--parents=2",
        "--population=3",
        "--new=1",
    ]
    parser = argparse.ArgumentParser()
    ow.get_args(parser)
    args = parser.parse_args(input_args)

    assert args.output_dir == pathlib.Path("testing123")
    assert args.generations == 2
    assert args.num_parents == 2
    assert args.population == 3
    assert args.num_random == 1


def test_gen_rw(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("test_gen_rw")
    population = 5
    num_gens = 10
    for gen in range(num_gens):
        results = [ow.Result(i, ow.Weights()) for i in range(population)]
        ow.write_generation(output_dir, gen, results)

        test_file = output_dir / f"results_{gen}.yaml"
        test = ow.read_gen_results(test_file)
        assert test == results, (test_file, test[0], results[0])


def test_new_inputs():
    population = 10
    num_parents = 2
    num_rand = 2
    test_gen_1 = ow.new_inputs([], population, num_parents, num_rand, 0.1)
    test_gen_1_results = [ow.Result(i, val) for i, val in enumerate(test_gen_1)]
    assert len(test_gen_1) == population

    test_gen_2 = ow.new_inputs(
        test_gen_1_results, population, num_parents, num_rand, 0.1
    )
    test_gen_2_results = [ow.Result(i, val) for i, val in enumerate(test_gen_2)]
    assert len(test_gen_2) == (population + num_parents)
    # Parents are taken from the beginning and passed through.
    for i in range(num_parents):
        assert test_gen_1[i] in test_gen_2

    test_gen_3 = ow.new_inputs(
        test_gen_2_results, population, num_parents, num_rand, 0.1
    )
    assert len(test_gen_3) == (population + num_parents)
    # Parents are taken from the beginning and passed through.
    for i in range(num_parents):
        assert test_gen_2[i] in test_gen_3
