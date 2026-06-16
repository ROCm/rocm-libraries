################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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

import logging

import pytest

from Tensile.ductile.core.mating import Mating
from Tensile.ductile.core.population import Individual, Population
from Tensile.ductile.utils import logger as logger_mod

pytestmark = pytest.mark.unit


class _Selection:
    def __call__(self, pop):
        return pop

    def __repr__(self):
        return "SEL"


class _Crossover:
    def __init__(self, pairs):
        self._pairs = pairs

    def __call__(self, _parents, _n_offsprings):
        for pair in self._pairs:
            yield pair

    def __repr__(self):
        return "CROSS"


class _Mutation:
    def __call__(self, ind):
        return ind

    def __repr__(self):
        return "MUT"


class _Space:
    def __init__(self, valid_fn):
        self._valid_fn = valid_fn

    def valid(self, ind):
        return self._valid_fn(ind)


def _individual(x):
    return Individual({"x": x}, F=float(x))


def test_mating_default_offspring_count_and_capacity_break():
    pop = Population([_individual(0), _individual(1)])
    pairs = [(_individual(2), _individual(3))]

    mating = Mating(
        space=_Space(lambda _ind: True),
        selection=_Selection(),
        crossover=_Crossover(pairs),
        mutation=_Mutation(),
        max_iters=4,
    )

    children = mating(pop)

    assert children.size == pop.size
    assert {ind["x"] for ind in children} == {2, 3}


def test_mating_raises_when_max_iters_reached_without_enough_valid_children():
    pop = Population([_individual(0), _individual(1)])
    pairs = [(_individual(2), _individual(3))]

    mating = Mating(
        space=_Space(lambda _ind: False),
        selection=_Selection(),
        crossover=_Crossover(pairs),
        mutation=_Mutation(),
        max_iters=1,
    )

    with pytest.raises(ValueError, match="max iters reached while generating offsprings"):
        mating(pop, n_offsprings=2)


def test_mating_repr_lists_components():
    mating = Mating(
        space=_Space(lambda _ind: True),
        selection=_Selection(),
        crossover=_Crossover([]),
        mutation=_Mutation(),
    )

    assert repr(mating) == "SEL\nCROSS\nMUT"


def test_setup_clears_handlers_and_writes_file(tmp_path):
    name = "ductile.logger.setup"
    log_path = tmp_path / "logs" / "run.log"

    logger_mod.setup(name, "%(message)s", level=logging.INFO)
    logger = logger_mod.setup(name, "%(message)s", level=logging.INFO, log_file=log_path)
    logger.info("hello")

    for handler in logger.handlers:
        handler.flush()

    assert log_path.is_file()
    assert "hello" in log_path.read_text(encoding="utf-8")


def test_logger_init_handles_racy_log_removal(monkeypatch, tmp_path):
    log_path = tmp_path / "race.log"

    monkeypatch.setattr(logger_mod.os.path, "isfile", lambda _p: True)

    def _raise_not_found(_p):
        raise FileNotFoundError("already removed")

    monkeypatch.setattr(logger_mod.os, "remove", _raise_not_found)

    logger_mod.Logger("ductile.logger.race", log_file=log_path)


def test_logger_print_and_log_lines(monkeypatch):
    lg = logger_mod.Logger("ductile.logger.lines", verbose=2)
    seen = []

    monkeypatch.setattr(lg.msg_logger, "info", lambda msg: seen.append(("print", msg)))
    lg.log = lambda level, line: seen.append((level, line))

    lg.print("hello")
    lg.log_lines("a\nb", logging.WARNING)

    assert ("print", "hello") in seen
    assert (logging.WARNING, "a") in seen
    assert (logging.WARNING, "b") in seen


def test_logger_print_stats_emits_formatted_table(monkeypatch):
    lg = logger_mod.Logger("ductile.logger.stats", verbose=1)
    lines = []

    monkeypatch.setattr(lg, "print", lambda msg: lines.append(msg))

    lg.print_stats(epoch=3, loss=1.23456, label="ok")

    assert len(lines) == 5
    assert "epoch" in lines[1]
    assert "loss" in lines[1]
    assert "label" in lines[1]
    assert "1.235" in lines[3]
