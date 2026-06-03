# Minimal `typing` shim for MicroPython.
#
# ck_dsl uses `from __future__ import annotations`, so annotations are strings
# and never evaluated at runtime — these names only need to be *importable* and
# (defensively) subscriptable/callable in case any are touched at runtime.


class _Anything:
    def __init__(self, name="typing._Anything"):
        self._name = name

    def __getitem__(self, item):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __repr__(self):
        return self._name


Any = _Anything("Any")
List = _Anything("List")
Dict = _Anything("Dict")
Tuple = _Anything("Tuple")
Optional = _Anything("Optional")
Union = _Anything("Union")
Sequence = _Anything("Sequence")
Mapping = _Anything("Mapping")
MutableMapping = _Anything("MutableMapping")
Iterable = _Anything("Iterable")
Iterator = _Anything("Iterator")
Callable = _Anything("Callable")
Set = _Anything("Set")
FrozenSet = _Anything("FrozenSet")
Type = _Anything("Type")
Literal = _Anything("Literal")
ClassVar = _Anything("ClassVar")
Final = _Anything("Final")
Annotated = _Anything("Annotated")
Generator = _Anything("Generator")

TYPE_CHECKING = False


def TypeVar(name, *args, **kwargs):
    return _Anything(name)


def NewType(name, tp):
    def _ident(x):
        return x

    return _ident


def cast(typ, val):
    return val


def overload(fn):
    return fn


def runtime_checkable(cls):
    return cls


def no_type_check(fn):
    return fn


class Generic:
    def __class_getitem__(cls, item):
        return cls


class Protocol:
    def __class_getitem__(cls, item):
        return cls
