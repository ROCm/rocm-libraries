################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""Tests for BenchmarkProblems._build_and_validate_solution and related helpers.

Tests cover: MI expansion (len=9 and len=0), WavefrontSize=-1 substitution,
silent vs. verbose rejection, None returned on exception, _generate_single_solution
pass-through, _generate_ga_solutions alignment/deduplication.
"""

import types

import pytest

import Tensile.BenchmarkProblems as BP

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------

def _isa_info(has_wave32=True):
    return types.SimpleNamespace(archCaps={"HasWave32": has_wave32})


def _isa_info_map(isa="gfx942", has_wave32=True):
    return {isa: _isa_info(has_wave32)}


def _debug_config(silent_rejection=True):
    return types.SimpleNamespace(
        splitGSU=False,
        printSolutionRejectionReason=not silent_rejection,
        printIndexAssignmentInfo=False,
    )


class _ValidSolution:
    """Minimal Solution-like that is always valid."""
    def __init__(self, *args, **kwargs):
        self._valid = True

    def __getitem__(self, key):
        if key == "Valid":
            return self._valid
        return None

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class _InvalidSolution(_ValidSolution):
    def __getitem__(self, key):
        if key == "Valid":
            return False
        return None


# ---------------------------------------------------------------------------
# _build_and_validate_solution
# ---------------------------------------------------------------------------

class TestBuildAndValidateSolution:
    def _base_solution(self, mi=(), wavefront=-1):
        return {
            "ProblemType": {"DataType": "f32"},
            "ISA": "gfx942",
            "MatrixInstruction": list(mi),
            "WavefrontSize": wavefront,
            "WorkGroup": [16, 16, 1],
        }

    def test_returns_none_when_mi_validation_fails(self, monkeypatch):
        monkeypatch.setattr(BP, "validateMIParameters", lambda sol, isa_map: False)
        monkeypatch.setattr(BP, "matrixInstructionToMIParameters", lambda *a, **kw: {})

        sol = self._base_solution(mi=[0] * 9, wavefront=64)
        result = BP._build_and_validate_solution(sol, object(), _debug_config(), _isa_info_map())
        assert result is None

    def test_returns_none_when_solution_invalid(self, monkeypatch):
        monkeypatch.setattr(BP, "validateMIParameters", lambda sol, isa_map: True)
        monkeypatch.setattr(BP, "matrixInstructionToMIParameters", lambda *a, **kw: {})
        monkeypatch.setattr(BP, "Solution", _InvalidSolution)

        sol = self._base_solution(mi=[0] * 9, wavefront=64)
        result = BP._build_and_validate_solution(sol, object(), _debug_config(), _isa_info_map())
        assert result is None

    def test_returns_solution_when_valid(self, monkeypatch):
        monkeypatch.setattr(BP, "validateMIParameters", lambda sol, isa_map: True)
        monkeypatch.setattr(BP, "matrixInstructionToMIParameters", lambda *a, **kw: {})
        monkeypatch.setattr(BP, "Solution", _ValidSolution)

        sol = self._base_solution(mi=[0] * 9, wavefront=64)
        result = BP._build_and_validate_solution(sol, object(), _debug_config(), _isa_info_map())
        assert result is not None

    def test_wavefront_minus1_resolved_to_32_when_has_wave32(self, monkeypatch):
        captured = {}

        def fake_mi_params(mi, isa, wavefront, ptype, workgroup, isa_map):
            captured["wavefront"] = wavefront
            return {}

        monkeypatch.setattr(BP, "validateMIParameters", lambda sol, isa_map: True)
        monkeypatch.setattr(BP, "matrixInstructionToMIParameters", fake_mi_params)
        monkeypatch.setattr(BP, "Solution", _ValidSolution)

        sol = self._base_solution(mi=[0] * 9, wavefront=-1)
        BP._build_and_validate_solution(sol, object(), _debug_config(), _isa_info_map(has_wave32=True))
        assert captured["wavefront"] == 32

    def test_wavefront_minus1_resolved_to_64_when_no_wave32(self, monkeypatch):
        captured = {}

        def fake_mi_params(mi, isa, wavefront, ptype, workgroup, isa_map):
            captured["wavefront"] = wavefront
            return {}

        monkeypatch.setattr(BP, "validateMIParameters", lambda sol, isa_map: True)
        monkeypatch.setattr(BP, "matrixInstructionToMIParameters", fake_mi_params)
        monkeypatch.setattr(BP, "Solution", _ValidSolution)

        sol = self._base_solution(mi=[0] * 9, wavefront=-1)
        BP._build_and_validate_solution(sol, object(), _debug_config(), _isa_info_map(has_wave32=False))
        assert captured["wavefront"] == 64

    def test_empty_mi_disables_matrix_instruction(self, monkeypatch):
        captured = {}

        class _CapSolution(_ValidSolution):
            def __init__(self, sol, *a, **kw):
                super().__init__(sol, *a, **kw)
                captured["emi"] = sol.get("EnableMatrixInstruction")

        monkeypatch.setattr(BP, "validateMIParameters", lambda sol, isa_map: True)
        monkeypatch.setattr(BP, "Solution", _CapSolution)

        sol = self._base_solution(mi=[], wavefront=64)
        BP._build_and_validate_solution(sol, object(), _debug_config(), _isa_info_map())
        assert captured.get("emi") is False

    def test_returns_none_on_exception(self, monkeypatch):
        def raise_exc(*a, **kw):
            raise RuntimeError("simulated error")

        monkeypatch.setattr(BP, "validateMIParameters", raise_exc)

        sol = self._base_solution(mi=[], wavefront=64)
        result = BP._build_and_validate_solution(sol, object(), _debug_config(), _isa_info_map())
        assert result is None

    def test_silent_mode_suppresses_rejection_print(self, monkeypatch, capsys):
        monkeypatch.setattr(BP, "validateMIParameters", lambda sol, isa_map: False)
        monkeypatch.setattr(BP, "matrixInstructionToMIParameters", lambda *a, **kw: {})

        sol = self._base_solution(mi=[0] * 9, wavefront=64)
        debug = types.SimpleNamespace(
            splitGSU=False,
            printSolutionRejectionReason=True,
            printIndexAssignmentInfo=False,
        )
        BP._build_and_validate_solution(sol, object(), debug, _isa_info_map(), silent=True)
        out = capsys.readouterr().out
        assert "rejecting" not in out


# ---------------------------------------------------------------------------
# _generate_single_solution — thin wrapper over _build_and_validate_solution
# ---------------------------------------------------------------------------

class TestGenerateSingleSolution:
    def test_passes_perm_and_constant_params_to_build(self, monkeypatch):
        captured = {}

        def fake_build(solution, *a, **kw):
            captured["solution"] = solution
            return None

        monkeypatch.setattr(BP, "_build_and_validate_solution", fake_build)

        perm = {"DepthU": 32}
        constant = {"WorkGroup": [16, 16, 1]}
        problem_type = types.SimpleNamespace(state={"DataType": "f32"})

        BP._generate_single_solution(perm, problem_type, constant, object(), _debug_config(), _isa_info_map())

        sol = captured["solution"]
        assert sol["DepthU"] == 32
        assert sol["WorkGroup"] == [16, 16, 1]
        assert "ProblemType" in sol
        assert "ISA" in sol


# ---------------------------------------------------------------------------
# _generate_ga_solutions (in ductile_backend.py)
# ---------------------------------------------------------------------------

class TestGenerateGaSolutions:
    def test_valid_solutions_appended(self, monkeypatch):
        import Tensile.backends.ductile_backend as dbmod

        monkeypatch.setattr(
            dbmod,
            "_generate_single_solution_with_groups",
            lambda perm, *a, **kw: _ValidSolution(),
        )
        monkeypatch.setattr(dbmod, "getKernelFileBase", lambda splitGSU, sol: str(id(sol)))

        individuals = [{"a": 0}, {"a": 1}]
        result = dbmod._generate_ga_solutions(
            types.SimpleNamespace(state={}),
            {},
            individuals,
            object(),
            types.SimpleNamespace(splitGSU=False),
            {"gfx942": {}},
        )
        assert len(result) == 2
        assert all(r is not None for r in result)

    def test_invalid_candidates_become_none(self, monkeypatch):
        import Tensile.backends.ductile_backend as dbmod

        monkeypatch.setattr(
            dbmod,
            "_generate_single_solution_with_groups",
            lambda perm, *a, **kw: None,  # always invalid
        )

        individuals = [{"a": 0}, {"a": 1}]
        result = dbmod._generate_ga_solutions(
            types.SimpleNamespace(state={}),
            {},
            individuals,
            object(),
            types.SimpleNamespace(splitGSU=False),
            {"gfx942": {}},
        )
        assert result == [None, None]

    def test_duplicate_solutions_become_none(self, monkeypatch):
        import Tensile.backends.ductile_backend as dbmod

        shared_sol = _ValidSolution()
        call_count = [0]

        def _gen(*a, **kw):
            call_count[0] += 1
            return shared_sol  # same object → same hash → duplicate

        monkeypatch.setattr(dbmod, "_generate_single_solution_with_groups", _gen)
        monkeypatch.setattr(dbmod, "getKernelFileBase", lambda splitGSU, sol: "same_base")

        individuals = [{"a": 0}, {"a": 1}]
        result = dbmod._generate_ga_solutions(
            types.SimpleNamespace(state={}),
            {},
            individuals,
            object(),
            types.SimpleNamespace(splitGSU=False),
            {"gfx942": {}},
        )
        # First is kept; second has same base → None
        assert result[0] is not None
        assert result[1] is None

    def test_empty_individuals_returns_empty_list(self, monkeypatch):
        import Tensile.backends.ductile_backend as dbmod

        result = dbmod._generate_ga_solutions(
            types.SimpleNamespace(state={}),
            {},
            [],
            object(),
            types.SimpleNamespace(splitGSU=False),
            {"gfx942": {}},
        )
        assert result == []
