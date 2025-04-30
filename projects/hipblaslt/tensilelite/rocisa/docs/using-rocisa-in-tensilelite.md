# Using rocIsa in TensileLite

## countType

countType is a legacy member function, but quite handy so we movedt into rocIsa as a function.

```c++
module.countType(Instruction) -> countType(module, Instruction)
```

It is suitable for prototyping, however it runs slower than directly using templates. If possible, add template function in ``count.cpp`` and export it to Python.

```
module.countType(Instruction) -> countInstruction(module)
```

## Exporting vectors to Python

Vector memory management between Python and C++ is different, so exporting vectors to Python is copy instead of reference.

```python
module = Module("Test")
vec = module.items() # the vec here is a copy
```

Since the elements are ``shared_ptr``, you can modify the contents of the elements.

BUT You cannot assign a new element or replace an existing element.

```python
# The following codes still work, but does not modify the vector in module.
vec[0] = item
vec.append(item)
```

Use the setter function to solve this problem.

```python
module.add(item)
module.add(item, pos=0)
```

Same as reading ``srcs`` from ``CommonInstruction``

```
Item.srcs[0] = vgpr(1)  # Wrong
item.setSrcs(0, vgpr(1))  # Correct
```

Also avoid using ``module.items()`` to check size, use ``module.itemSize()`` instead

```python
if module.items():  # Bad
    # your code here

if module.itemSize():  # Good
    # your code here
```
