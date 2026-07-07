# Golden values

Frozen reference outputs for the `origami::comm` cost model. The byte-identity
test suite asserts that the C++ model reproduces these values exactly.

## `algorithms_grid.csv`

`(algorithm, num_gpus, pid, timestep, my_rank) → ScheduleEntry` for every
sampled tuple across 9 algorithms × 3 world-sizes (2/4/8). 5,936 rows.

Columns:

| column      | meaning                                                |
|-------------|--------------------------------------------------------|
| `algorithm` | algorithm name (e.g. `RingAllGather`)                  |
| `num_gpus`  | world size                                             |
| `pid`      | logical workgroup id (0..7 sample)                     |
| `timestep` | algorithm tick (0..num_timesteps-1)                    |
| `my_rank`  | this GPU's rank                                        |
| `link_id`  | xGMI link the WG uses (-1 = `SELF_LINK`)               |
| `peer_rank`| the peer GPU rank                                      |
| `direction`| `pull` or `push`                                       |
| `is_self`  | 0/1 — whether this timestep is local                   |
| `wg_sig`   | work-graph opcode chain, e.g. `L\|X3\|R`               |

`wg_sig` opcodes (matches `op_sig()` in test_algorithms.cpp):

```
L    Load
S    Store
Sw   Store(write_through=true)
R    Reduce
P<peer>   Pull(peer)
X<peer>   Push(peer)
G<peer>   Signal(peer)
W<peer>   Wait(peer)
```

## Regenerating

These CSVs are a frozen oracle and should change only when the model is
intentionally recalibrated. When that happens, regenerate them from the
reference implementation and update the rows in lockstep with the model change
so the byte-identity gate stays meaningful.
