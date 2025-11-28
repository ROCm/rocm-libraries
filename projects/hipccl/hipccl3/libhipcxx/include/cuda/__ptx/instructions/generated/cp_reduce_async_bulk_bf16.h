// MIT License
//
// Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_REDUCE_ASYNC_BULK_BF16_H_
#define _CUDA_PTX_GENERATED_CP_REDUCE_ASYNC_BULK_BF16_H_

/*
// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .bf16 }
// .op        = { .min }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_min_t,
  __hip_bfloat16* dstMem,
  const __hip_bfloat16* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  space_global_t,
  space_shared_t,
  op_min_t,
  __hip_bfloat16* __dstMem,
  const __hip_bfloat16* __srcMem,
  _CUDA_VSTD::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_bf16 (due to parameter type constraint)
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.bf16  [%0], [%1], %2; // 4."
      :
      : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .bf16 }
// .op        = { .max }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_max_t,
  __hip_bfloat16* dstMem,
  const __hip_bfloat16* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  space_global_t,
  space_shared_t,
  op_max_t,
  __hip_bfloat16* __dstMem,
  const __hip_bfloat16* __srcMem,
  _CUDA_VSTD::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_bf16 (due to parameter type constraint)
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.bf16  [%0], [%1], %2; // 4."
      :
      : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.bulk_group.op.noftz.type  [dstMem], [srcMem], size; // 5. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .bf16 }
// .op        = { .add }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  __hip_bfloat16* dstMem,
  const __hip_bfloat16* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  space_global_t,
  space_shared_t,
  op_add_t,
  __hip_bfloat16* __dstMem,
  const __hip_bfloat16* __srcMem,
  _CUDA_VSTD::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_bf16 (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16  [%0], [%1], %2; // 5."
      :
      : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_CP_REDUCE_ASYNC_BULK_BF16_H_
