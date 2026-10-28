# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run two pass lists on separate copies of a module, and measure both."""

from dataclasses import dataclass, field


@dataclass
class Side:
    """One pass list's run: its own copy of the module, the records run_passes
    returned, and what the analyses measured on each function afterwards.

    pressure holds None for a function the run left without attached SSA.
    """

    passes: list
    module: object
    records: list
    pressure: dict = field(default_factory=dict)
    size: dict = field(default_factory=dict)
    hazards: dict = field(default_factory=dict)


def _value(measured, attribute):
    return "-" if measured is None else str(getattr(measured, attribute))


def _columns(side, name):
    """(label, text) for each number the summary shows, "-" where there is none."""
    pressure, size, gaps = (
        side.pressure.get(name),
        side.size.get(name),
        side.hazards.get(name),
    )
    violations = "-" if gaps is None else str(sum(1 for gap in gaps if not gap.ok))
    return [
        ("vgpr", _value(pressure, "vgpr")),
        ("sgpr", _value(pressure, "sgpr")),
        ("bytes", _value(size, "bytes")),
        ("cycles", _value(size, "cycles")),
        ("hazard violations", violations),
    ]


@dataclass
class Comparison:
    a: Side
    b: Side

    def __str__(self):
        names = list(self.a.size) + [
            name for name in self.b.size if name not in self.a.size
        ]
        lines = []
        for name in names:
            pairs = zip(_columns(self.a, name), _columns(self.b, name))
            text = ", ".join(
                "%s %s -> %s" % (label, a, b) for (label, a), (_, b) in pairs
            )
            lines.append("%s: %s" % (name or '""', text))
        return "\n".join(lines)


def _run(module, label, passes, run_options):
    copy = module.copy()
    try:
        records = copy.run_passes(passes, **run_options)
    except (ValueError, RuntimeError, TypeError) as error:
        raise type(error)("%s: %s" % (label, error)) from error
    side = Side(list(passes), copy, records)
    only = run_options.get("function")
    for function in copy.functions():
        if only is not None and function.name != only:
            continue
        name = function.name
        side.pressure[name] = copy.pressure(name) if function.has_attached_ssa else None
        side.size[name] = copy.size(name)
        side.hazards[name] = copy.hazards(name)
    return side


def compare(module, passes_a, passes_b, **run_options):
    """Run passes_a and passes_b, each on its own module.copy(), with the same
    run_passes keyword options, and measure the functions the run reached.

    The module given is not changed. A side whose run raises re-raises the same
    exception type, its message prefixed with passes_a: or passes_b:.
    """
    return Comparison(
        _run(module, "passes_a", passes_a, run_options),
        _run(module, "passes_b", passes_b, run_options),
    )
