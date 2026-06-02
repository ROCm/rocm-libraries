import sys

sys.path.insert(
    0, "/home/dahawkin/repo/worktrees/ck-dsl-provider-micropython/spike/mp1/shims"
)
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class P:
    a: int
    b: int = 0
    c: list = field(default_factory=list)


p = P(1)
print("init:", p.a, p.b, p.c)
q = replace(p, b=5)
print("replace:", q.a, q.b)
try:
    p.a = 9
    print("MUTATED (BAD: not frozen)")
except Exception as e:
    print("frozen ok:", type(e).__name__)
print("hash ok:", isinstance(hash(p), int))
print("eq ok:", P(1) == P(1), P(1) != P(2))
