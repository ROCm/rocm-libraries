import sys

sys.path.insert(
    0, "/home/dahawkin/repo/worktrees/ck-dsl-provider-micropython/spike/mp1/shims"
)
from dataclasses import dataclass, field, replace, fields, astuple, FrozenInstanceError


@dataclass(frozen=True)
class ES:
    op: str = field()
    dtype: str = field(default="f16")
    block_size: int = field(default=256)
    vec: int = field(default=8)
    tags: list = field(default_factory=list)


e = ES("copy", "f32")
print(
    "pos:",
    e.op == "copy" and e.dtype == "f32",
    "| order:",
    [f.name for f in fields(ES)],
)
print("astuple:", astuple(ES("copy")))
k = ES(op="relu", vec=4)
print("kw:", k.op, k.dtype, k.vec)
try:
    ES()
    print("required NOT enforced")
except TypeError:
    print("required ok")
a = ES("a")
b = ES("b")
a.tags.append(1)
print("factory indep:", b.tags == [])
try:
    e.op = "x"
    print("NOT frozen")
except FrozenInstanceError:
    print("frozen ok")
print("eq:", ES("copy", "f32") == ES("copy", "f32"), "neq:", ES("a") != ES("b"))


@dataclass(frozen=True)
class Scalar:
    a: int = field()
    b: int = field(default=2)


print(
    "hash ok:",
    isinstance(hash(Scalar(1)), int),
    "| hash eq:",
    hash(Scalar(1, 2)) == hash(Scalar(1, 2)),
)


@dataclass(frozen=True)
class Sub(ES):
    extra: int = field(default=0)


s = Sub("conv", extra=7)
print("inherit:", s.op, s.extra, "| order:", [f.name for f in fields(Sub)])
