# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests that need a GPU say so with one ``gpu("gfxNNN")`` mark, checked statically.

The ``gpu`` mark decides which lanes run a hardware test: TheRock's ``standard`` tier
selects ``-m gpu`` on a GPU runner, and the CPU-only lanes run ``-m "not gpu"`` on a
machine without one. These checks run on the CPU lanes too, so a test that breaks one of
the rules below fails there instead of quietly not running:

- No test decides from which GPU or HIP is present whether it runs. These values say
  which: ``hip``, ``gpu_is``, ``detect_gpu_target``, ``GFX_TARGET``, a ``HAS_GFX*``,
  ``ON_GFX*`` or ``IS_GFX*`` flag, the ``/dev/kfd`` and ``/dev/dri`` paths, the
  ``HIP_VISIBLE_DEVICES`` and ``ROCR_VISIBLE_DEVICES`` variables, and ``which`` of a
  tool that lists GPUs (``rocminfo``, ``amd-smi``). None of them may decide a
  ``skipif``, ``skipIf`` or ``skipUnless`` condition (string conditions included), an
  ``xfail(..., run=False)``, an ``if`` either of whose branches skips or, in a test,
  returns, continues or breaks, or a ``try`` that imports ``hip`` and skips in its
  ``except``. Nor may a test call ``importorskip("hip")``. Names are followed through
  every assignment in the file (module level, inside ``try``, local to a test) and
  through the file's own functions, so ``on950 = gpu_is("gfx950")`` or
  ``def _on_950(): return gpu_is("gfx950")`` followed by a skip is caught.
- At most one ``gpu`` mark per test across module, class, function and param, with arch
  strings only. ``-m "not gpu"`` deselects before ``conftest.py`` checks the marks at
  setup, so without this a bad mark only shows up on a GPU lane.
- A test without a ``gpu`` mark does not call ``run_on_gpu`` or the HIP runtime, and
  its own body does not branch on the values above. On a pull request the CPU lanes run
  such a test without a GPU and TheRock's ``standard`` tier does not select it, so its
  GPU branch runs only in Math CI's ``preliminary`` and the nightly tiers. A marked test
  may branch on its arch, for example to pick the expected value.

Not followed: names imported from another module (other than the ones above), a skip
inside a function of a module outside ``unit/``, a return or branch inside a function
defined in a test (fakes of the detection code branch on these values), and values
reached through ``getattr`` or ``eval``. A test that skips itself in one of these ways
still runs wherever the whole unit tree runs on a GPU machine: Math CI's
``preliminary`` and TheRock's nightly tiers.
"""

import ast
from pathlib import Path

import pytest

from gpu_detection import arch_matches, check_gpu_marks

pytestmark = pytest.mark.unit

_UNIT_DIR = Path(__file__).resolve().parent
_HELPERS = _UNIT_DIR / "gpu_test_helpers.py"

# Values that say which GPU or HIP this process has. A skip that reads one of them decides
# from the hardware whether the test runs, which only its gpu mark should decide. Prefix
# match for HAS_GFX950 etc.
_PRODUCERS = {"GFX_TARGET", "HIP_AVAILABLE", "hip", "gpu_is", "detect_gpu_target",
              "hip_available", "get_available_archs", "has_arch"}
_PRODUCER_PREFIXES = ("HAS_GFX", "ON_GFX", "IS_GFX")
# Strings a test could check instead of asking HIP: the device nodes and the variables that
# hide devices. Tools that list GPUs count only as the argument of ``which``: tests of the
# detection code compare fake ``amd-smi`` commands.
_PRODUCER_STRINGS = ("/dev/kfd", "/dev/dri", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
_GPU_TOOLS = ("rocminfo", "amd-smi", "rocm-smi", "amdgpu-arch")
# The conftest.py that turns gpu marks into skips, and this file. Every other file under
# unit/ is checked: gpu_test_helpers.py, and the conftest.py files of its subdirectories.
_EXEMPT = {_UNIT_DIR / "conftest.py", Path(__file__).resolve()}


def _is_producer(name):
    return (name in _PRODUCERS or name.startswith(_PRODUCER_PREFIXES)
            or name in _PRODUCER_STRINGS or name in _GPU_TOOLS)


def _aliases(tree):
    """Every value each name is bound to anywhere in ``tree``, for following a condition back
    to what it reads: assignments at any level (module, ``try``, a test's own locals), the
    file's functions (a predicate such as ``_on_950()``), and ``import ... as`` names."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            if node.value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for name in (n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)):
                aliases.setdefault(name, []).append(node.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            aliases.setdefault(node.name, []).append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.asname:
                    original = ast.Name(id=alias.name.split(".")[-1])
                    aliases.setdefault(alias.asname, []).append(original)
    return aliases


def _module_aliases(tree):
    """Module-level ``name = expr`` assignments, for expanding the names in a mark."""
    aliases = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    aliases.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            aliases.setdefault(node.target.id, []).append(node.value)
    return aliases


def _referenced_names(expr, aliases, seen=None):
    """Names and attribute names ``expr`` reads, following ``aliases``, the
    ``_PRODUCER_STRINGS`` its strings contain, and the ``_GPU_TOOLS`` it passes to ``which``."""
    seen = set() if seen is None else seen
    names = set()
    for node in ast.walk(expr):
        if isinstance(node, ast.Call) and getattr(
                node.func, "attr", getattr(node.func, "id", None)) == "which":
            names |= {a.value for a in node.args
                      if isinstance(a, ast.Constant) and a.value in _GPU_TOOLS}
        if isinstance(node, ast.Name):
            names.add(node.id)
            if node.id not in seen:
                seen.add(node.id)
                for value in aliases.get(node.id, ()):
                    names |= _referenced_names(value, aliases, seen)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names |= {text for text in _PRODUCER_STRINGS if text in node.value}
    return names


def _producers_in(expr, aliases):
    return sorted(n for n in _referenced_names(expr, aliases) if _is_producer(n))


def _skip_imports(tree):
    """Local names bound by ``from pytest import skip`` or ``from unittest import skipIf``
    (and ``as`` aliases), mapped to the name they import."""
    bound = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("pytest", "unittest"):
            for alias in node.names:
                bound[alias.asname or alias.name] = alias.name
    return bound


def _called(call, attr, imported):
    """True if ``call`` is ``<anything>.attr(...)`` or a bare name imported as attr."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr == attr
    return isinstance(func, ast.Name) and imported.get(func.id) == attr


def _condition(call):
    """The condition argument of a skip call. A string condition, which pytest evaluates in
    the test's module, is parsed so the names in it are read like any other."""
    condition = call.args[0] if call.args else next(
        (k.value for k in call.keywords if k.arg == "condition"), None)
    if isinstance(condition, ast.Constant) and isinstance(condition.value, str):
        try:
            return ast.parse(condition.value, mode="eval").body
        except SyntaxError:
            return None
    return condition


def _conditional_skip(call, imported):
    """``(kind, condition)`` if ``call`` skips on a condition, else None."""
    for attr in ("skipif", "skipIf", "skipUnless"):
        if _called(call, attr, imported):
            return attr, _condition(call)
    if _called(call, "xfail", imported) and any(
            k.arg == "run" and isinstance(k.value, ast.Constant) and k.value.value is False
            for k in call.keywords):
        return "xfail(run=False)", _condition(call)
    return None


def _skips(statements, imported):
    """True if ``statements`` skip the test: ``pytest.skip``, ``self.skipTest``, or raising
    ``unittest.SkipTest``."""
    for node in (n for statement in statements for n in ast.walk(statement)):
        if isinstance(node, ast.Call) and (_called(node, "skip", imported)
                                           or _called(node, "skipTest", imported)):
            return True
        if isinstance(node, ast.Raise) and node.exc is not None and (
                "SkipTest" in _referenced_names(node.exc, {})):
            return True
    return False


def _leaves_early(statements):
    """``return``, ``continue`` or ``break`` if ``statements`` end the test or its current
    loop iteration early, else None. Loops and functions inside ``statements`` are their own."""
    stack = [(statement, False) for statement in statements]
    while stack:
        node, in_inner_loop = stack.pop()
        if isinstance(node, ast.Return):
            return "return"
        if isinstance(node, (ast.Continue, ast.Break)) and not in_inner_loop:
            return "continue" if isinstance(node, ast.Continue) else "break"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        inner = in_inner_loop or isinstance(node, (ast.For, ast.AsyncFor, ast.While))
        stack.extend((child, inner) for child in ast.iter_child_nodes(node))
    return None


def _own_nodes(function):
    """The nodes of ``function``'s body, without the functions and classes defined in it."""
    stack = list(function.body)
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(node))


def _test_functions(tree):
    """Module-level ``test*`` functions and the ``test*`` methods of module-level classes."""
    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if isinstance(node, functions) and node.name.startswith("test"):
            yield node
        elif isinstance(node, ast.ClassDef):
            yield from (item for item in node.body
                        if isinstance(item, functions) and item.name.startswith("test"))


def _try_producers(node, aliases):
    """Producers the body of a ``try`` imports (``from hip import hip``) or reads."""
    names = set()
    for statement in node.body:
        for child in ast.walk(statement):
            if isinstance(child, ast.Import):
                names |= {alias.name.split(".")[0] for alias in child.names}
            elif isinstance(child, ast.ImportFrom) and child.module:
                names |= {child.module.split(".")[0]} | {alias.name for alias in child.names}
        names |= _referenced_names(statement, aliases)
    return sorted(n for n in names if _is_producer(n))


def arch_skips(source, name="<test>"):
    """Arch or hip skips in ``source``, as ``name:line: kind on <names>`` strings."""
    tree = ast.parse(source, name)
    aliases = _aliases(tree)
    imported = _skip_imports(tree)
    found = []
    reported = set()

    def report(node, kind, hits):
        if hits and id(node) not in reported:
            reported.add(id(node))
            found.append(f"{name}:{node.lineno}: {kind} on {', '.join(hits)}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _called(node, "importorskip", imported):
            module = node.args[0] if node.args else None
            if isinstance(module, ast.Constant) and str(module.value).split(".")[0] == "hip":
                report(node, "importorskip", ["hip"])
        elif isinstance(node, ast.Call) and _conditional_skip(node, imported):
            kind, condition = _conditional_skip(node, imported)
            if condition is not None:
                report(node, kind, _producers_in(condition, aliases))
        elif isinstance(node, ast.If) and (
                _skips(node.body, imported) or _skips(node.orelse, imported)):
            report(node, "if ...: skip", _producers_in(node.test, aliases))
        elif isinstance(node, ast.Try) and any(_skips(h.body, imported) for h in node.handlers):
            report(node, "try ...: except: skip", _try_producers(node, aliases))
    # An early return passes the test instead of skipping it, which is quieter still.
    for function in _test_functions(tree):
        for node in _own_nodes(function):
            leave = isinstance(node, ast.If) and _leaves_early(node.body + node.orelse)
            if leave:
                report(node, f"if ...: {leave}", _producers_in(node.test, aliases))
    return found


def _is_gpu_mark_node(node):
    """``pytest.mark.gpu`` / ``mark.gpu`` (with or without a call)."""
    return (isinstance(node, ast.Attribute) and node.attr == "gpu"
            and isinstance(node.value, (ast.Attribute, ast.Name))
            and getattr(node.value, "attr", getattr(node.value, "id", None)) == "mark")


def _helper_mark_aliases():
    """Names ``gpu_test_helpers`` binds to gpu marks (``requires_gpu``, ``GPU_MARKS``)."""
    tree = ast.parse(_HELPERS.read_text(), str(_HELPERS))
    aliases = _module_aliases(tree)
    marked = set()
    for name, values in aliases.items():
        if any(any(_is_gpu_mark_node(n) for n in ast.walk(value))
               or _referenced_names(value, {}) & marked for value in values):
            marked.add(name)
    return marked


def _gpu_marks(expr, mark_aliases):
    """The gpu-mark expressions in ``expr``: ``(count, problems)``."""
    count, problems = 0, []
    for node in ast.walk(expr):
        if _is_gpu_mark_node(node):
            count += 1
        elif isinstance(node, ast.Name) and node.id in mark_aliases:
            count += 1
        if isinstance(node, ast.Call) and _is_gpu_mark_node(node.func):
            if node.keywords:
                problems.append("keyword arguments")
            if any(isinstance(a, ast.Constant) and not isinstance(a.value, str) for a in node.args):
                problems.append("a non-string arch")
    return count, problems


def _pytestmark(body):
    return [n.value for n in body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in n.targets)]


def _hip_names(tree):
    """Names bound to the HIP runtime module: ``hip``, and ``as`` names for it."""
    names = {"hip"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {alias.asname for alias in node.names
                      if alias.asname and alias.name.split(".")[-1] == "hip"}
    return names


def _calls_gpu(function, hip_names):
    """True if ``function``, or a function nested in it, calls ``run_on_gpu`` or a ``hip*``
    runtime function."""
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if getattr(func, "id", getattr(func, "attr", None)) == "run_on_gpu":
            return True
        if isinstance(func, ast.Attribute) and func.attr.startswith("hip"):
            owner = func.value
            if (isinstance(owner, ast.Name) and owner.id in hip_names) or (
                    isinstance(owner, ast.Attribute) and owner.attr == "hip"):
                return True
    return False


def _branches_on_producers(function, aliases):
    """Producers that an ``if``, ``while`` or conditional expression in ``function``'s own
    body reads."""
    hits = set()
    for node in _own_nodes(function):
        if isinstance(node, (ast.If, ast.IfExp, ast.While)):
            hits |= set(_producers_in(node.test, aliases))
    return sorted(hits)


def gpu_mark_problems(source, name="<test>", mark_aliases=frozenset()):
    """Tests in ``source`` with more than one gpu mark or a malformed one, and unmarked tests
    that call ``run_on_gpu`` or the HIP runtime, or branch on which GPU is present (TheRock's
    ``standard`` tier does not select them, so on a pull request their GPU branch runs only in
    Math CI's ``preliminary``)."""
    tree = ast.parse(source, name)
    aliases = _module_aliases(tree)
    producer_aliases = _aliases(tree)
    hip_names = _hip_names(tree)
    found = []

    def expanded(expr):
        """``expr`` plus the module-level values it names, e.g. a CASES list of pytest.param."""
        return [expr] + [value for n in _referenced_names(expr, {})
                         if n in aliases and n not in mark_aliases for value in aliases[n]]

    def param_marks(exprs):
        """The ``marks=`` of each ``pytest.param`` in ``exprs``."""
        return [k.value for e in exprs for n in ast.walk(e)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "param" for k in n.keywords if k.arg == "marks"]

    def scope_marks(exprs):
        total, problems = 0, []
        for expr in exprs:
            count, issues = _gpu_marks(expr, mark_aliases)
            total += count
            problems += issues
        return total, problems

    module_count, module_problems = scope_marks(_pytestmark(tree.body))

    def check(function, outer_count, outer_problems):
        if not function.name.startswith("test"):
            return
        decorators = [expanded(d) for d in function.decorator_list]
        is_param = [any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "param" for e in exprs for n in ast.walk(e))
                    for exprs in decorators]
        own, problems = scope_marks(
            e for exprs, param in zip(decorators, is_param) if not param for e in exprs)
        param_exprs = [e for exprs, param in zip(decorators, is_param) if param for e in exprs]
        _, param_problems = scope_marks(param_exprs)
        # Each pytest.param carries its own marks, so a generated test gets one param's.
        per_param = max((_gpu_marks(marks, mark_aliases)[0] for marks in param_marks(param_exprs)),
                        default=0)
        total = outer_count + own + per_param
        problems = outer_problems + problems + param_problems
        where = f"{name}:{function.lineno}: {function.name}"
        if total > 1:
            found.append(f"{where}: {total} gpu marks across module, class, function and param")
        for problem in dict.fromkeys(problems):
            found.append(f"{where}: gpu() with {problem}; use arch strings, e.g. gpu(\"gfx950\")")
        if total == 0 and _calls_gpu(function, hip_names):
            found.append(f"{where}: calls run_on_gpu or the HIP runtime but has no gpu mark")
        branches = _branches_on_producers(function, producer_aliases) if total == 0 else []
        if branches:
            found.append(f"{where}: branches on {', '.join(branches)} but has no gpu mark")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            check(node, module_count, module_problems)
        elif isinstance(node, ast.ClassDef):
            class_count, class_problems = scope_marks(
                list(node.decorator_list) + _pytestmark(node.body))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    check(item, module_count + class_count, module_problems + class_problems)
    return found


def _unit_test_files():
    return [p for p in sorted(_UNIT_DIR.rglob("*.py")) if p.resolve() not in _EXEMPT]


def test_no_arch_skip_in_unit_tests():
    found = [hit for path in _unit_test_files()
             for hit in arch_skips(path.read_text(), str(path.relative_to(_UNIT_DIR)))]
    assert not found, (
        "Use @pytest.mark.gpu(\"gfxNNN\"), not a skip:\n  " + "\n  ".join(found))


def test_gpu_marks_are_well_formed():
    aliases = _helper_mark_aliases()
    found = [hit for path in _unit_test_files()
             for hit in gpu_mark_problems(path.read_text(), str(path.relative_to(_UNIT_DIR)),
                                          aliases)]
    assert not found, "\n  ".join(["gpu mark problems:"] + found)


@pytest.mark.parametrize("source", [
    # The four forms that got past the first version of this guard.
    "import pytest\nfrom gpu_detection import gpu_is\n_ON_950 = gpu_is('gfx950')\n"
    "@pytest.mark.skipif(not _ON_950, reason='x')\ndef test_a(): pass\n",
    "from pytest import skip\nfrom gpu_test_helpers import hip\n"
    "def test_a():\n    if hip is None:\n        skip('no hip')\n",
    "import pytest\nhip = pytest.importorskip('hip')\ndef test_a(): pass\n",
    "import pytest\nfrom gpu_test_helpers import hip\n"
    "def test_a():\n    if not hip:\n        pytest.skip('no hip')\n",
    # Forms the guard always caught.
    "import pytest\nfrom gpu_test_helpers import GFX_TARGET\n"
    "@pytest.mark.skipif(GFX_TARGET != 'gfx950', reason='x')\ndef test_a(): pass\n",
    "import pytest\nfrom gpu_detection import detect_gpu_target\n"
    "def test_a():\n    if detect_gpu_target() is None:\n        pytest.skip('x')\n",
    # Locals, else branches, early exits, try, string and unittest conditions, flags, paths.
    "import pytest\nfrom gpu_detection import gpu_is\n"
    "def test_a():\n    on950 = gpu_is('gfx950')\n    if not on950:\n        pytest.skip('x')\n",
    "import pytest\nfrom gpu_detection import gpu_is\n"
    "def test_a():\n    if gpu_is('gfx950'):\n        pass\n    else:\n        pytest.skip('x')\n",
    "from gpu_detection import gpu_is\n"
    "def test_a():\n    if not gpu_is('gfx950'):\n        return\n    assert True\n",
    "from gpu_detection import gpu_is\n"
    "def test_a():\n    for arch in ('gfx942', 'gfx950'):\n        if not gpu_is(arch):\n"
    "            continue\n        assert arch\n",
    "import pytest\nfrom gpu_test_helpers import HIP_AVAILABLE\n"
    "@pytest.mark.skipif('not HIP_AVAILABLE', reason='x')\ndef test_a(): pass\n",
    "import pytest\ndef test_a():\n    try:\n        from hip import hip\n"
    "    except ImportError:\n        pytest.skip('x')\n",
    "import pytest\ntry:\n    from hip import hip\n    HAVE_DEVICE = hip.hipGetDeviceCount()[1] > 0\n"
    "except ImportError:\n    HAVE_DEVICE = False\n"
    "@pytest.mark.skipif(not HAVE_DEVICE, reason='x')\ndef test_a(): pass\n",
    "import pytest\nfrom gpu_detection import gpu_is\ndef _on_950():\n    return gpu_is('gfx950')\n"
    "@pytest.mark.skipif(not _on_950(), reason='x')\ndef test_a(): pass\n",
    "import os, pytest\n@pytest.mark.skipif(not os.path.exists('/dev/kfd'), reason='x')\n"
    "def test_a(): pass\n",
    "import shutil, pytest\n@pytest.mark.skipif(not shutil.which('rocminfo'), reason='x')\n"
    "def test_a(): pass\n",
    "import unittest\nfrom gpu_detection import gpu_is\nclass T(unittest.TestCase):\n"
    "    @unittest.skipIf(not gpu_is('gfx950'), 'x')\n    def test_a(self): pass\n",
    "import unittest\nfrom gpu_detection import gpu_is\nclass T(unittest.TestCase):\n"
    "    def test_a(self):\n        if not gpu_is('gfx950'):\n            self.skipTest('x')\n",
    "import pytest\nfrom gpu_detection import gpu_is\n"
    "@pytest.mark.xfail(not gpu_is('gfx950'), run=False, reason='x')\ndef test_a(): pass\n",
    "import pytest\nfrom other_helpers import ON_GFX950\n"
    "@pytest.mark.skipif(not ON_GFX950, reason='x')\ndef test_a(): pass\n",
], ids=["aliased-gpu_is", "from-pytest-skip", "importorskip-hip", "if-not-hip",
        "GFX_TARGET", "detect_gpu_target", "local-alias", "skip-in-else", "early-return",
        "continue-in-arch-loop", "string-condition", "try-import-hip", "flag-set-in-module-try",
        "helper-predicate", "dev-kfd", "which-rocminfo", "unittest-skipIf", "skipTest",
        "xfail-run-false", "ON_GFX-flag"])
def test_arch_skips_are_found(source):
    assert arch_skips(source)


# Tests of the detection code fake it; a branch or return in the fake is not the test's.
_FAKE_DETECTION = (
    "import os\ndef test_a():\n    def fake():\n        if os.environ.get('HIP_VISIBLE_DEVICES'):\n"
    "            return None\n        return 'gfx950'\n    assert fake() in (None, 'gfx950')\n")


@pytest.mark.parametrize("source", [
    # The reason text names a producer; the condition does not.
    "import sys, pytest\n"
    "@pytest.mark.skipif(sys.platform == 'win32', reason='GFX_TARGET is not set on Windows')\n"
    "def test_a(): pass\n",
    "import pytest\nyaml = pytest.importorskip('yaml')\ndef test_a(): pass\n",
    "import os, pytest\ndef test_a():\n    if not os.environ.get('CI'):\n        pytest.skip('x')\n",
    "import pytest\ndef test_a():\n    try:\n        import yaml\n"
    "    except ImportError:\n        pytest.skip('x')\n",
    "import os\ndef test_a():\n    if not os.environ.get('CI'):\n        return\n    assert True\n",
    # A helper may choose a code path by arch; only a test returning early is a gate.
    "from gpu_detection import gpu_is\ndef _pick():\n    if gpu_is('gfx950'):\n        return 1\n"
    "    return 2\ndef test_a():\n    assert _pick() in (1, 2)\n",
    # A tool name is a producer only as the argument of ``which``.
    "def test_a():\n    def fake(cmd):\n        if cmd[0] == 'amd-smi':\n            return 1\n"
    "    assert fake(['amd-smi']) == 1\n",
    _FAKE_DETECTION,
    # pytest.ini sets xfail_strict, so this xfail still runs the test and fails if it passes.
    "import pytest\nfrom gpu_detection import gpu_is\n"
    "@pytest.mark.xfail(gpu_is('gfx942'), reason='x')\ndef test_a(): pass\n",
], ids=["producer-only-in-reason", "importorskip-other-module", "unrelated-condition",
        "try-import-other-module", "early-return-unrelated", "helper-chooses-by-arch",
        "fake-amd-smi", "fake-detection", "xfail-that-runs"])
def test_other_skips_are_allowed(source):
    assert not arch_skips(source)


@pytest.mark.parametrize("source, message", [
    ("import pytest\npytestmark = pytest.mark.gpu('gfx942')\n"
     "@pytest.mark.gpu('gfx950')\ndef test_a(): pass\n", "2 gpu marks"),
    ("import pytest\n@pytest.mark.gpu\nclass TestA:\n"
     "    @pytest.mark.gpu('gfx950')\n    def test_a(self): pass\n", "2 gpu marks"),
    ("import pytest\n@pytest.mark.gpu\n"
     "@pytest.mark.parametrize('a', [pytest.param(1, marks=pytest.mark.gpu('gfx950'))])\n"
     "def test_a(a): pass\n", "2 gpu marks"),
    ("import pytest\n@pytest.mark.gpu(arch='gfx950')\ndef test_a(): pass\n", "keyword"),
    ("import pytest\n@pytest.mark.gpu(950)\ndef test_a(): pass\n", "non-string"),
    ("from gpu_test_helpers import run_on_gpu\ndef test_a():\n    run_on_gpu('k.co', 4)\n",
     "no gpu mark"),
    ("from gpu_test_helpers import hip\ndef test_a():\n    hip.hipModuleLaunchKernel()\n",
     "no gpu mark"),
    ("import pytest\nfrom gpu_test_helpers import GPU_MARKS\npytestmark = GPU_MARKS\n"
     "@pytest.mark.gpu('gfx942')\ndef test_a(): pass\n", "2 gpu marks"),
    ("import pytest\n@pytest.mark.parametrize('a', [pytest.param(1, marks=[\n"
     "    pytest.mark.gpu('gfx942'), pytest.mark.gpu('gfx950')])])\ndef test_a(a): pass\n",
     "2 gpu marks"),
    ("import gpu_test_helpers\ndef test_a():\n    gpu_test_helpers.run_on_gpu('k.co', 4)\n",
     "no gpu mark"),
    ("from hip import hip as h\ndef test_a():\n    h.hipModuleLaunchKernel()\n", "no gpu mark"),
    ("from gpu_detection import gpu_is\ndef test_a():\n    if gpu_is('gfx950'):\n"
     "        assert True\n", "branches on gpu_is but has no gpu mark"),
], ids=["module-and-function", "class-and-method", "function-and-param", "keyword",
        "non-string", "run_on_gpu-unmarked", "hip-call-unmarked", "helper-alias-and-function",
        "two-marks-on-one-param", "module-run_on_gpu-unmarked", "aliased-hip-unmarked",
        "branches-on-gpu_is-unmarked"])
def test_gpu_mark_problems_are_found(source, message):
    found = gpu_mark_problems(source, mark_aliases=_helper_mark_aliases())
    assert any(message in f for f in found), found


@pytest.mark.parametrize("source", [
    "import pytest\n@pytest.mark.gpu('gfx950', 'gfx942')\ndef test_a(): pass\n",
    "import pytest\n@pytest.mark.parametrize('a', [\n"
    "    pytest.param(1, marks=pytest.mark.gpu('gfx950')),\n"
    "    pytest.param(2, marks=pytest.mark.gpu('gfx942'))])\ndef test_a(a): pass\n",
    "import pytest\npytestmark = pytest.mark.gpu\nfrom gpu_test_helpers import run_on_gpu\n"
    "def test_a():\n    run_on_gpu('k.co', 4)\n",
    "import pytest\nTARGET = 'gfx1250'\n@pytest.mark.gpu(TARGET)\ndef test_a(): pass\n",
    # A marked test may branch on its arch, e.g. for the expected value.
    "import pytest\nfrom gpu_detection import gpu_is\n@pytest.mark.gpu('gfx942', 'gfx950')\n"
    "def test_a():\n    expected = 1 if gpu_is('gfx950') else 2\n    assert expected\n",
    _FAKE_DETECTION,
], ids=["one-mark-two-archs", "one-mark-per-param", "module-mark-covers-launch", "named-arch",
        "marked-test-branches-on-arch", "fake-detection"])
def test_well_formed_gpu_marks_pass(source):
    assert not gpu_mark_problems(source, mark_aliases=_helper_mark_aliases())


@pytest.mark.parametrize("found,wanted,expected", [
    ("gfx950", "gfx950", True),
    ("gfx942", "gfx950", False),
    ("gfx1250", "gfx1250", True),
    ("gfx1250-strict", "gfx1250", True),
    ("gfx1250", "gfx1250-strict", False),
    ("gfx1250-strict", "gfx1250-strict", True),
    ("gfx12500", "gfx1250", False),
])
def test_arch_matches(found, wanted, expected):
    assert arch_matches(found, wanted) is expected


class _Item:
    """The part of a pytest item check_gpu_marks reads: its gpu marks, nearest first."""

    def __init__(self, *marks):
        self.nodeid = "t"
        self._marks = [m.mark for m in marks]

    def iter_markers(self, name):
        return (m for m in self._marks if m.name == name)


@pytest.mark.parametrize("item", [
    _Item(pytest.mark.gpu),
    _Item(pytest.mark.gpu("gfx950", "gfx942")),
    _Item(pytest.mark.unit, pytest.mark.gpu("gfx1250")),
], ids=["bare", "archs", "other-marks"])
def test_check_gpu_marks_accepts(item):
    check_gpu_marks(item)


@pytest.mark.parametrize("item, message", [
    (_Item(pytest.mark.gpu("gfx950"), pytest.mark.gpu("gfx942")), "2 gpu marks"),
    (_Item(pytest.mark.gpu("gfx942"), pytest.mark.gpu), "2 gpu marks"),
    (_Item(pytest.mark.gpu(arch="gfx950")), "arch strings only"),
    (_Item(pytest.mark.gpu(950)), "arch strings only"),
], ids=["stacked-or-param", "function-and-class", "keyword", "non-string"])
def test_check_gpu_marks_rejects(item, message):
    with pytest.raises(pytest.UsageError, match=message):
        check_gpu_marks(item)
