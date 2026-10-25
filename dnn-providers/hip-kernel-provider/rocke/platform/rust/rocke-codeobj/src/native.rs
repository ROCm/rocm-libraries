// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Flat C ABI for direct rocKE IR compilation.

use std::mem;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;
use std::str;

use crate::direct_gemm::compile_gfx1151_wmma_gemm_ir;
use crate::error::CompileError;

pub const ROCKE_CODEOBJ_FFI_ABI_VERSION: u32 = 1;

#[repr(C)]
pub struct RockeDirectGemmCompileRequestV1 {
    pub struct_size: u32,
    pub serialized_ir: *const u8,
    pub serialized_ir_size: usize,
}

#[repr(C)]
pub struct RockeCodeobjCompileResultV1 {
    pub struct_size: u32,
    pub status: i32,
    pub hsaco: *mut u8,
    pub hsaco_size: usize,
    pub diagnostics: *mut u8,
    pub diagnostics_size: usize,
    pub bc_seconds: f64,
    pub link_seconds: f64,
    pub relocatable_seconds: f64,
    pub executable_seconds: f64,
}

fn leak_bytes(bytes: Vec<u8>) -> (*mut u8, usize) {
    if bytes.is_empty() {
        return (ptr::null_mut(), 0);
    }
    let mut bytes = bytes.into_boxed_slice();
    let pointer = bytes.as_mut_ptr();
    let size = bytes.len();
    mem::forget(bytes);
    (pointer, size)
}

unsafe fn reclaim_bytes(pointer: *mut u8, size: usize) {
    if !pointer.is_null() {
        let raw = ptr::slice_from_raw_parts_mut(pointer, size);
        drop(unsafe { Box::from_raw(raw) });
    }
}

fn error_text(error: &CompileError) -> Vec<u8> {
    format!("{}: {}", error.kind, error.message).into_bytes()
}

fn panic_text() -> Vec<u8> {
    b"internal: Rust code-object backend panicked".to_vec()
}

unsafe fn input_bytes<'a>(pointer: *const u8, size: usize, name: &str) -> Result<&'a [u8], String> {
    if pointer.is_null() || size == 0 {
        return Err(format!("{name} is empty"));
    }
    Ok(unsafe { slice::from_raw_parts(pointer, size) })
}

unsafe fn input_string(pointer: *const u8, size: usize, name: &str) -> Result<String, String> {
    let bytes = unsafe { input_bytes(pointer, size, name)? };
    str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|error| format!("{name} is not UTF-8: {error}"))
}

fn empty_compile_result() -> RockeCodeobjCompileResultV1 {
    RockeCodeobjCompileResultV1 {
        struct_size: mem::size_of::<RockeCodeobjCompileResultV1>() as u32,
        status: 0,
        hsaco: ptr::null_mut(),
        hsaco_size: 0,
        diagnostics: ptr::null_mut(),
        diagnostics_size: 0,
        bc_seconds: 0.0,
        link_seconds: 0.0,
        relocatable_seconds: 0.0,
        executable_seconds: 0.0,
    }
}

unsafe fn reset_compile_result(result: *mut RockeCodeobjCompileResultV1) -> Result<(), ()> {
    if result.is_null()
        || unsafe { (*result).struct_size } != mem::size_of::<RockeCodeobjCompileResultV1>() as u32
    {
        return Err(());
    }
    unsafe { ptr::write(result, empty_compile_result()) };
    Ok(())
}

fn set_compile_error(result: &mut RockeCodeobjCompileResultV1, diagnostics: Vec<u8>) -> i32 {
    let (pointer, size) = leak_bytes(diagnostics);
    result.status = 1;
    result.diagnostics = pointer;
    result.diagnostics_size = size;
    result.status
}

#[no_mangle]
pub extern "C" fn rocke_codeobj_ffi_abi_version() -> u32 {
    ROCKE_CODEOBJ_FFI_ABI_VERSION
}

/// Compile one serialized rocKE gfx1151 WMMA GEMM graph directly to HSACO.
///
/// This entry point performs no toolchain probing, process launch, compiler
/// library loading, or cache lookup. The returned HSACO allocation is the
/// writer's original allocation and remains owned by the result until dispose.
///
/// # Safety
///
/// `request` and `result` must point to valid v1 structs. `serialized_ir` must
/// remain readable for `serialized_ir_size` bytes for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn rocke_codeobj_compile_gfx1151_wmma_gemm_v1(
    request: *const RockeDirectGemmCompileRequestV1,
    result: *mut RockeCodeobjCompileResultV1,
) -> i32 {
    if unsafe { reset_compile_result(result) }.is_err() {
        return 1;
    }
    let result = unsafe { &mut *result };
    let operation = catch_unwind(AssertUnwindSafe(|| {
        if request.is_null()
            || unsafe { (*request).struct_size }
                != mem::size_of::<RockeDirectGemmCompileRequestV1>() as u32
        {
            return Err(CompileError::new(
                "request",
                "direct GEMM request has an incompatible struct_size",
            ));
        }
        let request = unsafe { &*request };
        let ir = unsafe {
            input_string(
                request.serialized_ir,
                request.serialized_ir_size,
                "serialized_ir",
            )
        }
        .map_err(|message| CompileError::new("request", message))?;
        compile_gfx1151_wmma_gemm_ir(&ir)
    }));
    match operation {
        Ok(Ok(compiled)) => {
            let (hsaco, hsaco_size) = leak_bytes(compiled.hsaco);
            let (diagnostics, diagnostics_size) = leak_bytes(compiled.diagnostics.into_bytes());
            result.hsaco = hsaco;
            result.hsaco_size = hsaco_size;
            result.diagnostics = diagnostics;
            result.diagnostics_size = diagnostics_size;
            result.executable_seconds = compiled.seconds;
            0
        }
        Ok(Err(error)) => set_compile_error(result, error_text(&error)),
        Err(_) => set_compile_error(result, panic_text()),
    }
}

/// Compile one serialized rocKE kernel (`ck.dsl.ir/v1`) directly to a gfx1151
/// HSACO. Unsupported IR is rejected with a diagnostic; nothing falls back to
/// another compiler.
///
/// # Safety
///
/// Same contract as [`rocke_codeobj_compile_gfx1151_wmma_gemm_v1`].
#[no_mangle]
pub unsafe extern "C" fn rocke_codeobj_compile_rocke_ir_v1(
    request: *const RockeDirectGemmCompileRequestV1,
    result: *mut RockeCodeobjCompileResultV1,
) -> i32 {
    if unsafe { reset_compile_result(result) }.is_err() {
        return 1;
    }
    let result = unsafe { &mut *result };
    let operation = catch_unwind(AssertUnwindSafe(|| {
        if request.is_null()
            || unsafe { (*request).struct_size }
                != mem::size_of::<RockeDirectGemmCompileRequestV1>() as u32
        {
            return Err(CompileError::new(
                "request",
                "rocKE IR request has an incompatible struct_size",
            ));
        }
        let request = unsafe { &*request };
        let ir = unsafe {
            input_string(
                request.serialized_ir,
                request.serialized_ir_size,
                "serialized_ir",
            )
        }
        .map_err(|message| CompileError::new("request", message))?;
        let start = std::time::Instant::now();
        let compiled = crate::rir::compile(&ir)?;
        Ok((compiled, start.elapsed().as_secs_f64()))
    }));
    match operation {
        Ok(Ok((compiled, seconds))) => {
            let s = &compiled.stats;
            let phases: Vec<String> = s
                .phases
                .iter()
                .map(|(name, time)| format!("{name}:{:.3}", time.as_secs_f64() * 1000.0))
                .collect();
            let diagnostics = format!(
                "instructions={} code_bytes={} vgprs={} sgprs={} lds_bytes={} phases_ms={}",
                s.instructions,
                s.code_bytes,
                s.vgprs,
                s.sgprs,
                s.lds_bytes,
                phases.join(",")
            );
            let (hsaco, hsaco_size) = leak_bytes(compiled.hsaco);
            let (diagnostics, diagnostics_size) = leak_bytes(diagnostics.into_bytes());
            result.hsaco = hsaco;
            result.hsaco_size = hsaco_size;
            result.diagnostics = diagnostics;
            result.diagnostics_size = diagnostics_size;
            result.executable_seconds = seconds;
            0
        }
        Ok(Err(error)) => set_compile_error(result, error_text(&error)),
        Err(_) => set_compile_error(result, panic_text()),
    }
}

/// Releases allocations owned by a compile result.
///
/// # Safety
///
/// `result` must be null or a v1 compile result last initialized by this
/// library, and it must not have been disposed already.
#[no_mangle]
pub unsafe extern "C" fn rocke_codeobj_compile_result_dispose_v1(
    result: *mut RockeCodeobjCompileResultV1,
) {
    if result.is_null() {
        return;
    }
    let result = unsafe { &mut *result };
    unsafe {
        reclaim_bytes(result.hsaco, result.hsaco_size);
        reclaim_bytes(result.diagnostics, result.diagnostics_size);
    }
    let struct_size = result.struct_size;
    unsafe {
        ptr::write(
            result,
            RockeCodeobjCompileResultV1 {
                struct_size,
                ..empty_compile_result()
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_abi_and_invalid_request_are_stable() {
        assert_eq!(rocke_codeobj_ffi_abi_version(), 1);
        let mut result = empty_compile_result();
        let status = unsafe {
            rocke_codeobj_compile_gfx1151_wmma_gemm_v1(ptr::null(), &mut result as *mut _)
        };
        assert_eq!(status, 1);
        let diagnostics =
            unsafe { slice::from_raw_parts(result.diagnostics, result.diagnostics_size) };
        assert!(String::from_utf8_lossy(diagnostics).contains("struct_size"));
        unsafe { rocke_codeobj_compile_result_dispose_v1(&mut result as *mut _) };
        assert!(result.hsaco.is_null());
        assert!(result.diagnostics.is_null());
    }

    #[test]
    fn direct_ffi_returns_the_writer_allocation() {
        let ir = include_str!("gfx1151_wmma_gemm.ir");
        let request = RockeDirectGemmCompileRequestV1 {
            struct_size: mem::size_of::<RockeDirectGemmCompileRequestV1>() as u32,
            serialized_ir: ir.as_ptr(),
            serialized_ir_size: ir.len(),
        };
        let mut result = empty_compile_result();
        let status =
            unsafe { rocke_codeobj_compile_gfx1151_wmma_gemm_v1(&request, &mut result as *mut _) };
        assert_eq!(status, 0);
        assert_eq!(result.status, 0);
        assert!(!result.hsaco.is_null());
        assert!(result.hsaco_size > 0);
        let image = unsafe { slice::from_raw_parts(result.hsaco, result.hsaco_size) };
        assert!(image.starts_with(b"\x7fELF"));
        unsafe { rocke_codeobj_compile_result_dispose_v1(&mut result as *mut _) };
        assert!(result.hsaco.is_null());
    }
}
