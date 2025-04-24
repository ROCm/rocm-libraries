#!/usr/bin/env python3
import os
import sys
import subprocess

# Order of redops, tys, protos, algos must match src/include/device.h
all_colls =  ["AllGather","AllReduce","AllToAllPivot","Broadcast","Reduce","ReduceScatter","SendRecv"]
all_redops = ["Sum","Prod","MinMax","PreMulSum","SumPostDiv"]
all_tys =    ["i8","u8","i32","u32","i64","u64","f16","f32","f64","bf16","f8e4m3","f8e5m2"]
all_protos = ["LL","LL128","SIMPLE"]
all_algos =  ["TREE","RING"]
all_unroll = ["1", "2", "4"]

all_params = [all_colls, all_algos, all_protos, all_redops, all_tys, all_unroll]

################################################################################
# The first command line argument is the path to the directory to generate and
# populate.

gensrc = sys.argv[1]

if os.path.exists(gensrc):
  for name in os.listdir(gensrc):
    os.remove(os.path.join(gensrc, name))
    #os.truncate(os.path.join(gensrc, name), 0)
else:
  os.makedirs(gensrc)

################################################################################
# The command line argument is used as a regex to filter the functions
# which make it into librccl. This is helpful for reducing the binary when
# developing device code. The regex supports non-space containing globs '*',
# and union 'a|b'. The string representing the function has the form:
#
# <coll> <algo> <proto> <redop> <type>
#
# The possible values for redop, type, algo, proto can be found in the all_<foo>
# lists at the top of this file.
#
# Example use-cases:
#
# # Only send/recv:
# make ONLY_FUNCS="SendRecv"
#
# # Only AllReduce and Reduce
# make ONLY_FUNCS="AllReduce|Reduce"
#
# # Only non-reductions:
# make ONLY_FUNCS="AllGather * *|Broadcast * *|SendRecv"
#
# # Only AllReduce Sum int32_t (but all algos, protos)
# make ONLY_FUNCS="AllReduce * * Sum int32_t"
#
# # Only AllReduce RING Max float (but all protos)
# make ONLY_FUNCS="AllReduce RING * Max float"
#
# # AllReduce TREE LL128 Prod rccl_bfloat16
# make ONLY_FUNCS="AllReduce TREE LL128 Prod rccl_bfloat16"
#
# # AllReduce RING SIMPLE and ReduceScatter RING LL float (but all redops, types for AllReduce and all redops for ReduceScatter)
# make ONLY_FUNCS="AllReduce RING SIMPLE * *|ReduceScatter RING LL * float"
#                         --- or ---
# make ONLY_FUNCS="AllReduce RING SIMPLE|ReduceScatter RING LL * float"
# make ONLY_FUNCS="AllReduce RING/TREE LL/SIMPLE Sum/MinMax int8_t/uint8_t/half/float/double/hip_bfloat16/rccl_float8/rccl_bfloat8|AllGather RING LL/SIMPLE Sum int8_t|AllToAllPivot RING SIMPLE Sum int8_t|Broadcast RING LL/SIMPLE Sum int8_t|Reduce RING LL/SIMPLE Sum/MinMax int8_t/uint8_t/half/float/double/hip_bfloat16/rccl_float8/rccl_bfloat8|ReduceScatter RING LL/SIMPLE Sum/MinMax int8_t/uint8_t/half/float/double/hip_bfloat16/rccl_float8/rccl_bfloat8|SendRecv RING SIMPLE Sum int8_t"

# Paste all non-None arguments together with `sep`.
def paste(sep, *args):
  return sep.join(x for x in args if x is not None)

is_ifc             = 1 if sys.argv[2] == "ON" else 0
is_colltrace       = 1 if sys.argv[3] == "ON" else 0
is_msccl_kernels   = 1 if sys.argv[4] == "ON" else 0
is_local_arch_only = 1 if sys.argv[5] == "ON" else 0

func_pattern = sys.argv[6:7]
if func_pattern and func_pattern[0]:
  func_pattern = func_pattern[0]
else:
  func_pattern = "AllGather|AllReduce|AllToAllPivot|Broadcast|Reduce|ReduceScatter|SendRecv"

################################################################################

algos_of_coll = {
  "AllGather":     ["RING"],
  "AllReduce":     all_algos,
  "AllToAllPivot": ["RING"],
  "Broadcast":     ["RING"],
  "Reduce":        ["RING"],
  "ReduceScatter": ["RING"],
  "SendRecv":      ["RING"]
}

protos_of_coll = {
  "AllGather":     all_protos,
  "AllReduce":     all_protos,
  "AllToAllPivot": ["SIMPLE"],
  "Broadcast":     all_protos,
  "Reduce":        all_protos,
  "ReduceScatter": all_protos,
  "SendRecv":      ["SIMPLE"]
}

redops_of_coll = {
  "AllGather":     ["Sum"],
  "AllReduce":     all_redops,
  "AllToAllPivot": ["Sum"],
  "Broadcast":     ["Sum"],
  "Reduce":        all_redops,
  "ReduceScatter": all_redops,
  "SendRecv":      ["Sum"]
}

tys_of_coll = {
  "AllGather":     ["i8"],
  "AllReduce":     all_tys,
  "AllToAllPivot": ["i8"],
  "Broadcast":     ["i8"],
  "Reduce":        all_tys,
  "ReduceScatter": all_tys,
  "SendRecv":      ["i8"]
}

coll_camel_to_lower = {
  "AllGather":     "all_gather",
  "AllReduce":     "all_reduce",
  "AllToAllPivot": "alltoall_pivot",
  "Broadcast":     "broadcast",
  "Reduce":        "reduce",
  "ReduceScatter": "reduce_scatter",
  "SendRecv":      "sendrecv"
}
coll_lower_to_camel = {coll_camel_to_lower[x]: x for x in coll_camel_to_lower}

################################################################################

def calc_unroll_for_local_arch():
  if not is_local_arch_only: 
    return

  rocminfo_path = os.environ.get('ROCM_PATH') + "/bin/rocminfo"

  res = subprocess.run([rocminfo_path], stdout=subprocess.PIPE, universal_newlines=True)
  rocminfo_output = res.stdout
  
  # Parse rocminfo binary output
  gfx_targets = {}
  curr_name = None
  for line in rocminfo_output.splitlines():
    line = line.strip()

    if line.startswith("Name:"):
      name = line.split(':')[-1].strip()
      if "gfx" in name:
        curr_name = name
    if line.startswith("Compute Unit:") and curr_name:
      cu_count = int(line.split(':')[-1].strip())
      gfx_targets[(curr_name, cu_count)] = None
      curr_name = None
  
  # We want to remove duplicates but cannot use a dictionary since same gfx name can have different cu counts
  # Use (gfx_name, cu_count) as key for dictionary and convert it to list here
  gfx_targets = list(gfx_targets.keys())

  # Homogeneous system is required to build for only 1 varient of unroll factor
  if len(gfx_targets) == 1:
    gfx_name, cu_count = gfx_targets[0]
    if "gfx950" == gfx_name:
        return 1
    elif "gfx908" == gfx_name or ("gfx942" == gfx_name and cu_count > 80):
      return 2
    else:
      return 4

# Helper function to check if the conditions for the collective is being met
def func_validate(coll, algo, proto, redop, ty):
  if redop == "SumPostDiv" and ty[0] not in ("i","u"):
    return False
  if algo not in algos_of_coll[coll] or proto not in protos_of_coll[coll] or redop not in redops_of_coll[coll] or ty not in tys_of_coll[coll]:
    return False
  return True

# A recursive helper to generate collective functions based on the input given
def func_filter(function_params, current_idx, item_list=None):
  if item_list is None:
    item_list = []

  # Check if current_idx exceeds the max depth
  if current_idx < len(all_params):
    # Current element is the config parameter
    current_element = function_params[current_idx]

    # If the paramter is equal to '*', include all possible cases for it
    if current_element == "*":
      if current_idx == 0:
        raise ValueError("Error: Paramter 'COLL' can not be type all '*'.")
      
      # all_params list must be in the same order as function_params --> <coll> <algo> <proto> <redop> <type>
      # Get the current list from all_params
      current_list = all_params[current_idx]

      # Iterate over the items int the current_list
      for item in current_list:
        # Add item to item_list which will be used in the inner most loop
        item_list.append(item)
        yield from func_filter(function_params, current_idx+1, item_list)

        # For each loop layer remove the last element in item_list
        item_list.pop()
    else:
      # Check if the current element is recognized
      elements = current_element.split("/")
      current_param = all_params[current_idx]
      
      # Iterate over the elements in the elements list
      for item in elements:
        if item not in current_param:
          raise ValueError(f"Error: {item} is unrecognized or does not belong to this category {current_param}.")
        
      for item in elements:
        item_list.append(item)
        yield from func_filter(function_params, current_idx+1, item_list)

        # For each loop layer remove the last element in item_list
        item_list.pop()
  else:
    coll, algo, proto, redop, ty, unroll = item_list

    if func_validate(coll, algo, proto, redop, ty):
      yield(coll, algo, proto, redop, ty, unroll)

# Parse ONLY_FUNCS input and feed it to func_filter
def parse_input(func_pattern):
  input_list = sorted(func_pattern.split("|"))

  for input in input_list:
    function_params = input.split()
    params_length = len(function_params)

    # If a parameter is missing, append '*'
    while params_length < len(all_params):
      function_params.append("*")
      params_length += 1

    # Filter functions/kernels based on input
    yield from func_filter(function_params, 0)

# Maps functions to the chosen representative for the equivalence class it
# belongs to. For instance (sum, signed int) maps to (sum, unsigned int).
def equivalent_primary(coll, algo, proto, redop, ty, unroll):
  # if local arch only, we only need to build for 1 varient of coll_unroll.
  # map the other varient of coll_unroll to this one.
  if coll_unroll:
    unroll = str(coll_unroll)
  if coll in ("AllReduce", "Reduce", "ReduceScatter"):
    # map signed integer sum/prod to unsigned
    if redop in ("Sum","Prod","PreMulSum","SumPostDiv") and ty[0]=="i":
      ty = "u"+ty[1:]
    # map signed integer min/max to unsigned for non-NVLS
    elif redop=="MinMax" and ty[0]=="i" and ("NVLS" not in algo):
      ty = "u"+ty[1:]
  return (coll, algo, proto, redop, ty, unroll)

# Order rows are enumerated must match formula of `ncclDevFuncId()`:
def enumerate_func_rows():
  for unroll in all_unroll:
    for coll in all_colls:
      for algo in all_algos:
        for proto in all_protos:
          for redop in all_redops:
            for ty in all_tys:
              if func_validate(coll, algo, proto, redop, ty):
                yield (coll, algo, proto, redop, ty, unroll)

# Sort the hashmap based on custom key <coll> <algo> <proto> <redop> <ty>
def custom_sort_key(fn):
    coll, algo, proto, redop, ty, unroll = fn
    
    return (
        all_unroll.index(unroll),
        all_colls.index(coll),
        all_algos.index(algo),
        all_protos.index(proto),
        all_redops.index(redop),
        all_tys.index(ty)
    )

################################################################################

coll_unroll = calc_unroll_for_local_arch()

# Corresponds to ncclDevFuncRowToId[]
func_rows = [fn for fn in enumerate_func_rows()]

# Corresponds to ncclDevFuncTable[]
primary_funcs = sorted(set(equivalent_primary(*fn) for fn in parse_input(func_pattern)), key=custom_sort_key)

# primary_to_index[primary_funcs[i]] == i
primary_to_index = {fn: primary_funcs.index(fn) if fn in primary_funcs else -1 for fn in func_rows}

################################################################################

# Generate <gensrc>/device_table.h
with open(os.path.join(gensrc, "device_table.h"), "w") as f:
  print("-- Generating %s" % os.path.join(gensrc, "device_table.h"))
  out = f.write

  if is_ifc: func_declaration = "__device__ void"
  else: func_declaration = "__device__ __attribute__((noinline)) void"

  for fn in primary_funcs:
    sym = paste("_", "ncclDevFunc", *fn)
    if fn[2] == "LL128":
      out("#if (defined(__gfx90a__) || defined(__gfx942__)) && defined(ENABLE_LL128)\n")
      out("%s %s();\n#else\n" % (func_declaration, sym))
      fn_ll = fn[:2] + ("LL",) + fn[3:]
      sym_ll = paste("_", "ncclDevFunc", *fn_ll)
      out("%s %s();\n#endif\n" % (func_declaration, sym_ll))
    else:
      out("%s %s();\n" % (func_declaration, sym))
  out("\n")

  out("typedef void(*ncclDevFuncPtr_t)();\n\n")
  out("__device__ ncclDevFuncPtr_t const ncclDevFuncTable_1[] = {\n")
  index1 = 0
  for fn in primary_funcs:
    coll, algo, proto, redop, ty, unroll = fn
    if unroll != "1": continue
    sym = paste("_", "ncclDevFunc", *fn)
    if fn[2] == "LL128":
      out("#if (defined(__gfx90a__) || defined(__gfx942__)) && defined(ENABLE_LL128)\n")
      out("/*%4d*/ %s,\n#else\n" % (index1, sym))
      fn_ll = fn[:2] + ("LL",) + fn[3:]
      sym_ll = paste("_", "ncclDevFunc", *fn_ll)
      out("/*%4d*/ %s,\n#endif\n" % (index1, sym_ll))
    else:
      out("/*%4d*/ %s,\n" % (index1, sym))
    index1 += 1
  out("nullptr};\n")
  out("\n")
  out("__device__ ncclDevFuncPtr_t const ncclDevFuncTable_2[] = {\n")
  index2 = 0
  for fn in primary_funcs:
    coll, algo, proto, redop, ty, unroll = fn
    if unroll != "2": continue
    sym = paste("_", "ncclDevFunc", *fn)
    if fn[2] == "LL128":
      out("#if (defined(__gfx90a__) || defined(__gfx942__)) && defined(ENABLE_LL128)\n")
      out("/*%4d*/ %s,\n#else\n" % (index2, sym))
      fn_ll = fn[:2] + ("LL",) + fn[3:]
      sym_ll = paste("_", "ncclDevFunc", *fn_ll)
      out("/*%4d*/ %s,\n#endif\n" % (index2, sym_ll))
    else:
      out("/*%4d*/ %s,\n" % (index2, sym))
    index2 += 1
  out("nullptr};\n")
  out("\n")
  out("__device__ ncclDevFuncPtr_t const ncclDevFuncTable_4[] = {\n")
  index4 = 0
  for fn in primary_funcs:
    coll, algo, proto, redop, ty, unroll = fn
    if unroll != "4": continue
    sym = paste("_", "ncclDevFunc", *fn)
    if fn[2] == "LL128":
      out("#if (defined(__gfx90a__) || defined(__gfx942__)) && defined(ENABLE_LL128)\n")
      out("/*%4d*/ %s,\n#else\n" % (index4, sym))
      fn_ll = fn[:2] + ("LL",) + fn[3:]
      sym_ll = paste("_", "ncclDevFunc", *fn_ll)
      out("/*%4d*/ %s,\n#endif\n" % (index4, sym_ll))
    else:
      out("/*%4d*/ %s,\n" % (index4, sym))
    index4 += 1
  out("nullptr};\n")
  out("\n")
  
  if not is_ifc:
    out("template<unsigned short f, unsigned short l>\n"
      "struct Caller1 {\n"
      "  static __forceinline__ __device__ __host__\n"
      "  void call1(unsigned short funcIndex) noexcept\n"
      "  {\n"
      "    constexpr unsigned short m = f + (l - f) / 2;\n"
      "    return (funcIndex < m) ? Caller1<f, m>::call1(funcIndex) : Caller1<m, l>::call1(funcIndex);\n"
      "  }\n"
      "};\n"
      "\n"
      "template<unsigned short f>\n"
      "struct Caller1<f, f + 1>{\n"
      "  static __forceinline__ __device__ __host__\n"
      "  void call1(unsigned short funcIndex) noexcept { ncclDevFuncTable_1[f](); }\n"
      "};\n")
    out("__forceinline__ __device__ void NCCL_CALL_FUNCTIONS_1(unsigned short funcIndex) noexcept {\n")
    out(f"  Caller1<0, {index1}>::call1(funcIndex);\n")
    out("}\n\n")
    out("template<unsigned short f, unsigned short l>\n"
      "struct Caller2 {\n"
      "  static __forceinline__ __device__ __host__\n"
      "  void call2(unsigned short funcIndex) noexcept\n"
      "  {\n"
      "    constexpr unsigned short m = f + (l - f) / 2;\n"
      "    return (funcIndex < m) ? Caller2<f, m>::call2(funcIndex) : Caller2<m, l>::call2(funcIndex);\n"
      "  }\n"
      "};\n"
      "\n"
      "template<unsigned short f>\n"
      "struct Caller2<f, f + 1>{\n"
      "  static __forceinline__ __device__ __host__\n"
      "  void call2(unsigned short funcIndex) noexcept { ncclDevFuncTable_2[f](); }\n"
      "};\n")
    out("__forceinline__ __device__ void NCCL_CALL_FUNCTIONS_2(unsigned short funcIndex) noexcept {\n")
    out(f"  Caller2<0, {index2}>::call2(funcIndex);\n")
    out("}\n\n")
    out("template<unsigned short f, unsigned short l>\n"
      "struct Caller4 {\n"
      "  static __forceinline__ __device__ __host__\n"
      "  void call4(unsigned short funcIndex) noexcept\n"
      "  {\n"
      "    constexpr unsigned short m = f + (l - f) / 2;\n"
      "    return (funcIndex < m) ? Caller4<f, m>::call4(funcIndex) : Caller4<m, l>::call4(funcIndex);\n"
      "  }\n"
      "};\n"
      "\n"
      "template<unsigned short f>\n"
      "struct Caller4<f, f + 1>{\n"
      "  static __forceinline__ __device__ __host__\n"
      "  void call4(unsigned short funcIndex) noexcept { ncclDevFuncTable_4[f](); }\n"
      "};\n")
    out("__forceinline__ __device__ void NCCL_CALL_FUNCTIONS_4(unsigned short funcIndex) noexcept {\n")
    out(f"  Caller4<0, {index4}>::call4(funcIndex);\n")
    out("}\n\n")

# Generate <gensrc>/device_table.cpp
if is_colltrace:
  with open(os.path.join(gensrc, "device_table.cpp"), "w") as f:
    print("-- Generating %s" % os.path.join(gensrc, "device_table.cpp"))

    out = f.write
    out('#include "nccl_common.h"\n#include "device.h"\n')
    out("\n")
    
    seen_fns = set()
    out("const char* funcNames[FUNC_INDEX_TOTAL] = {\n")
    for fn in primary_funcs:
      fn_no_unroll = fn[:-1]
      if fn_no_unroll not in seen_fns:
        out('   "%s",\n' % paste("_", "ncclDevFunc", *fn_no_unroll))
        seen_fns.add(fn_no_unroll)
    for ty in all_tys:
      out(f'   "ncclDevFunc_OneRankReduce_PreMulSum_{ty}",\n')
    out("};\n")

# Generate <gensrc>/host_table.cpp
with open(os.path.join(gensrc, "host_table.cpp"), "w") as f:
  print("-- Generating %s" % os.path.join(gensrc, "host_table.cpp"))

  out = f.write
  out('#include "device.h"\n')
  out("\n")

  # The mapping from function rows to valid primary function ids.
  out("extern int const ncclDevFuncRowToId[] = {\n")
  index = 0
  for fn in func_rows[:len(func_rows)//3]:
    fn_id, comment = -1, ""
    if fn is not None:
      fn_id = primary_to_index[equivalent_primary(*fn)]
      comment = " // " + paste(" ", *fn[:-1])
    out("/*%4d*/ %d,%s\n" % (index, fn_id, comment))
    index += 1
  out(f"{index}")
  out("};\n")

# Maps to .cu filename which implements this func. The only constraint is that
# "coll" is reflected in the name: formally that no two funcs having different
# coll's map to the same filename.
def impl_filename(coll, algo, proto, redop, ty, unroll):
  return "%s.cpp" % paste("_", coll_camel_to_lower[coll], redop and redop.lower(), ty)

# Partition the functions and kernels to the .cu filenames. The partition is
# a dictionary mapping filename to (coll, func-tuple list)
def partition_by_name(fns):
  ans = {}
  for fn in fns:
    name = impl_filename(*fn)
    coll = fn[0]
    if name not in ans:
      ans[name] = (coll, [])
    ans[name][1].append(fn)
  return ans

name_to_funcs = partition_by_name(fn for fn in primary_funcs if fn[0]!="Nop")

redop_to_cxx = {
  None: "FuncCopy",
  "Sum": "FuncSum",
  "Prod": "FuncProd",
  "MinMax": "FuncMinMax",
  "PreMulSum": "FuncPreMulSum",
  "SumPostDiv": "FuncSumPostDiv"
}

ty_to_cxx = {
  None: "int8_t",
  "i8": "int8_t",
  "u8": "uint8_t",
  "i32": "int32_t",
  "u32": "uint32_t",
  "i64": "int64_t",
  "u64": "uint64_t",
  "f16": "half",
  "f32": "float",
  "f64": "double",
  "bf16": "hip_bfloat16",
  "f8e4m3":  "rccl_float8",
  "f8e5m2": "rccl_bfloat8"
}

# Generate each <gensrc>/<impl>.cpp:
for name in name_to_funcs.keys():
  (coll, fns) = name_to_funcs[name]
  with open(os.path.join(gensrc, name), "w") as f:
    print("-- Generating %s" % os.path.join(gensrc, name))

    out = f.write
    out(
      '#include "common.h"\n'
      '#include "{lower_coll}.h"\n'
      .format(lower_coll=coll_camel_to_lower[coll])
    )

    for fn in fns:
      (coll, algo, proto, redop, ty, unroll) = fn
      sym = paste("_", coll, algo, proto, redop, ty, unroll)
      if proto == "LL128":
        out("#if (defined(__gfx90a__) || defined(__gfx942__)) && defined(ENABLE_LL128)\n")
      out(
        "DEFINE_ncclDevFunc({sym}, ncclFunc{coll}, {redop_cxx}, {ty_cxx}, NCCL_ALGO_{algo}, NCCL_PROTO_{proto}, {unroll})\n"
        .format(sym=sym, coll=coll, redop_cxx=redop_to_cxx[redop], ty_cxx=ty_to_cxx[ty],
                algo=(algo or "RING"), proto=(proto or "SIMPLE"), unroll=unroll)
      )
      if proto == "LL128":
        out("#endif\n")

# Generate each <gensrc>/<msccl_impl>.cpp
if is_msccl_kernels:
  for redop in all_redops:
    if redop in ("Sum", "Prod", "MinMax"):
      for ty in all_tys:
        with open(os.path.join(gensrc, f"msccl_kernel_{redop}_{ty}.cpp"), "w") as f:
          print("-- Generating %s" % os.path.join(gensrc, f"msccl_kernel_{redop}_{ty}.cpp"))

          out = f.write
          out('#include "msccl_kernel_impl.h"\n#include "nccl_common.h"\n')
          out(
            "MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE({redop}, {ty_cxx}, false);\n"
            .format(redop=redop, ty_cxx=ty_to_cxx[ty])
          )
