# RegisterPool Overflow Handling Investigation

## Verdict

Open. The current target source still defines `ResourceOverflowException` as a function, so constructing it returns `None`; overflow listeners receive `None`, and overflow paths without a listener execute `raise None`, producing `TypeError` instead of a domain exception.

## Current Source References

- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Common/RegisterPool.py:32-33` defines `ResourceOverflowException` with `def ResourceOverflowException(Exception): pass`, not `class ResourceOverflowException(Exception): pass`.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Common/RegisterPool.py:57-62` in `allocTmpGpr` constructs `exception = ResourceOverflowException("gpr overflow")`, passes it to `overflowListener`, or raises it.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Common/RegisterPool.py:108-113` in `allocTmpGprList` has the same construction/listener/raise path.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/KernelWriterAssembly.py:555-561` routes `allocTmpSgpr` overflow through a listener that sets `self.states.overflowedResources = 2` and, when `AssertOnSgprOverflow` is enabled, does `raise e`.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/KernelWriterAssembly.py:564-570` does the same for `allocTmpSgprList`.

## Characterization Comparison

The referenced characterization file is available and matches the current target copy byte-for-byte:

- Referenced: `/home/alvasile/repos/rocm-libraries-investigation/projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/RegisterPool/test_register_pool_char.py`
- Current target: `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/RegisterPool/test_register_pool_char.py`

Relevant current target characterization references:

- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/RegisterPool/test_register_pool_char.py:46-55` expects `allocTmpGpr` listener overflow to receive `[None]`.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/RegisterPool/test_register_pool_char.py:58-63` expects `allocTmpGpr` overflow without a listener to raise `TypeError`.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/RegisterPool/test_register_pool_char.py:100-104` expects `allocTmpGprList` overflow without a listener to raise `TypeError`.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/RegisterPool/test_register_pool_char.py:107-112` expects `allocTmpGprList` listener overflow to receive `[None]`.

## Reproduction / Evidence

Static Python semantics are sufficient: a function with no return statement returns `None`, so `ResourceOverflowException("gpr overflow")` returns `None`.

I also loaded the current `Tensile/Common/RegisterPool.py` directly with minimal in-memory `rocisa` stubs and forced the overflow branches. Output:

```text
ResourceOverflowException return None
allocTmpGpr yielded 0 4
allocTmpGpr listener seen [None]
allocTmpGpr no-listener exception TypeError exceptions must derive from BaseException
allocTmpGprList yielded [(0, 4)]
allocTmpGprList listener seen [None]
allocTmpGprList no-listener exception TypeError exceptions must derive from BaseException
```

Focused pytest could not be completed in this environment:

- `tox -e unit -- Tensile/Tests/unit/characterization/RegisterPool/test_register_pool_char.py -q` attempted package install and failed resolving PyPI build dependencies under restricted network.
- `.tox/unit/bin/python -m pytest Tensile/Tests/unit/characterization/RegisterPool/test_register_pool_char.py -q` failed before test collection because the unit conftest imports `rocisa.rocIsa`, but this checkout lacks the compiled `rocisa._rocisa` extension.

## Impact

`allocTmpGpr` and `allocTmpGprList` do not deliver a usable exception object to callers on overflow. Code that installs an overflow listener only receives `None`, so it cannot inspect or re-raise a meaningful exception. Code without a listener raises `TypeError`, which obscures the actual register overflow. In `KernelWriterAssembly.allocTmpSgpr` and `allocTmpSgprList`, `AssertOnSgprOverflow=True` will also raise `TypeError` from `raise e` because `e` is `None`, while the non-asserting listener path still marks `overflowedResources = 2` and allows the oversized allocation to yield.

## Recommended Fix / Test

Fix `ResourceOverflowException` by changing it from a function to an exception class:

```python
class ResourceOverflowException(Exception):
  pass
```

Then update or add focused tests for both allocators:

- Listener path: force overflow and assert the listener receives an instance of `ResourceOverflowException` with the expected message.
- No-listener path: force overflow and assert `pytest.raises(ResourceOverflowException)`, not `TypeError`.
- KernelWriterAssembly listener path: with `AssertOnSgprOverflow=True`, verify the listener re-raises `ResourceOverflowException`; with it disabled, verify `overflowedResources` is set and no exception is raised.
