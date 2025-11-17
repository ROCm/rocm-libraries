# How to Add a New StinkyTofu Assembly IR

To add a new StinkyTofu assembly IR, you'll need to access the following files:

```bash
CommonInstsDSL.cpp
Flags.def
Gfx1250.cpp
RocisaHwInstMappings.hpp
```

Here we will add the instruction `s_wait_tensorcnt` step by step.

## Step 1. Add a flag (Optional)

Add a flag in `Flags.def` for a new instruction type if needed.

```c++
MACRO(IF_WaitTensorCnt)
```

# Step 2. Create a new instruction structure (Optional)

Add a new instruction type `CommonInstsDSL.cpp` if needed.

```c++
    struct WaitTensorCntInst : GfxInstDef
    {
        WaitTensorCntInst()
        {
            hwInstDesc.flags.set(IF_WaitTensorCnt);
        }
    };
```

# Step 3. Add definition to the corresponding architecture

`s_wait_tensorcnt` is a new feature in GFX1250. We'll add the definition to `Gfx1250.cpp`.

```c++
        DEF_T(WaitTensorCntInst, "s_wait_tensorcnt");
```

# Step 4. At last, add the mapping to `rocisa`

In `RocisaHwInstMapping.cpp`, add the name of the `struct` in `rocisa` (`SWaitTensorcnt`) and the assembly instruction `s_wait_tensorcnt`.

```c++
{"SWaitTensorcnt", "s_wait_tensorcnt"},
```
