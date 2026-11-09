// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Flat C ABI for direct rocKE IR compilation.

use std::ffi::{c_char, CStr};
use std::mem;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;
use std::str;
use std::sync::Arc;

use crate::direct_gemm::compile_gfx1151_wmma_gemm_ir;
use crate::error::CompileError;
use crate::rir::session::{Session, SessionOptions};
use crate::rir::CompiledKernel;
use crate::target::Target;

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

/// The request's text, read in place: the caller keeps it readable for the
/// duration of the call.
unsafe fn input_string<'a>(pointer: *const u8, size: usize, name: &str) -> Result<&'a str, String> {
    let bytes = unsafe { input_bytes(pointer, size, name)? };
    str::from_utf8(bytes).map_err(|error| format!("{name} is not UTF-8: {error}"))
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
    set_compile_outcome(result, operation)
}

/// Request for `rocke_codeobj_compile_rocke_ir_v2`: the v1 request and the
/// target to compile for.
#[repr(C)]
pub struct RockeIrCompileRequestV2 {
    pub struct_size: u32,
    pub serialized_ir: *const u8,
    pub serialized_ir_size: usize,
    /// NUL-terminated target name (`"gfx1151"`, `"gfx950"`); an unknown or unsupported
    /// one is a `request` error.
    pub target: *const c_char,
}

/// The supported target `name` points at, or the `request` error.
unsafe fn input_target(name: *const c_char) -> Result<Target, CompileError> {
    if name.is_null() {
        return Err(CompileError::new("request", "target is null"));
    }
    let name = unsafe { CStr::from_ptr(name) }
        .to_str()
        .map_err(|error| CompileError::new("request", format!("target is not UTF-8: {error}")))?;
    Target::requested(name)
}

/// `rocke_codeobj_compile_rocke_ir_v1` for the request's target.
///
/// # Safety
///
/// Same contract as [`rocke_codeobj_compile_rocke_ir_v1`], and `target`
/// must be null or point to a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn rocke_codeobj_compile_rocke_ir_v2(
    request: *const RockeIrCompileRequestV2,
    result: *mut RockeCodeobjCompileResultV1,
) -> i32 {
    if unsafe { reset_compile_result(result) }.is_err() {
        return 1;
    }
    let result = unsafe { &mut *result };
    let operation = catch_unwind(AssertUnwindSafe(|| {
        if request.is_null() || unsafe { (*request).struct_size } != mem::size_of::<RockeIrCompileRequestV2>() as u32 {
            return Err(CompileError::new("request", "rocKE IR request has an incompatible struct_size"));
        }
        let request = unsafe { &*request };
        let target = unsafe { input_target(request.target) }?;
        let ir = unsafe { input_string(request.serialized_ir, request.serialized_ir_size, "serialized_ir") }
            .map_err(|message| CompileError::new("request", message))?;
        let start = std::time::Instant::now();
        let compiled = crate::rir::compile_for(&ir, target)?;
        Ok((compiled, start.elapsed().as_secs_f64()))
    }));
    set_compile_outcome(result, operation)
}

/// The targets this library compiles for: a static NUL-terminated,
/// comma-separated list of names (`"gfx1151,gfx950"`).
#[no_mangle]
pub extern "C" fn rocke_codeobj_supported_targets_v1() -> *const c_char {
    crate::target::SUPPORTED_NAMES.as_ptr()
}

/// Fills `result` from a compile, as every rocKE IR entry point reports it.
fn set_compile_outcome(
    result: &mut RockeCodeobjCompileResultV1,
    operation: std::thread::Result<Result<(CompiledKernel, f64), CompileError>>,
) -> i32 {
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

#[repr(C)]
pub struct RockeCodeobjSessionOptionsV1 {
    pub struct_size: u32,
    /// Worker threads the session starts; 0 picks a default.
    pub worker_threads: u32,
}

/// Creates a compile session (see `rir::session`) and stores it in
/// `*session`, or stores null and returns 1 when `session` is null, the
/// options' `struct_size` does not match, or the session cannot be created.
/// Null `options` takes the defaults.
///
/// # Safety
///
/// `options` must be null or point to a valid v1 options struct; `session`
/// must be null or point to writable storage for a pointer.
#[no_mangle]
pub unsafe extern "C" fn rocke_codeobj_session_create_v1(
    options: *const RockeCodeobjSessionOptionsV1,
    session: *mut *mut Session,
) -> i32 {
    if session.is_null() {
        return 1;
    }
    unsafe { *session = ptr::null_mut() };
    let worker_threads = if options.is_null() {
        0
    } else if unsafe { (*options).struct_size } != mem::size_of::<RockeCodeobjSessionOptionsV1>() as u32 {
        return 1;
    } else {
        unsafe { (*options).worker_threads }
    };
    match catch_unwind(|| Box::new(Session::new(SessionOptions { worker_threads, target: Target::Gfx1151 }))) {
        Ok(created) => {
            unsafe { *session = Box::into_raw(created) };
            0
        }
        Err(_) => 1,
    }
}

#[repr(C)]
pub struct RockeCodeobjSessionOptionsV2 {
    pub struct_size: u32,
    /// Worker threads the session starts; 0 picks a default.
    pub worker_threads: u32,
    /// NUL-terminated name of the target every compile of the session is
    /// for (`"gfx1151"`, `"gfx950"`).
    pub target: *const c_char,
}

/// `rocke_codeobj_session_create_v1` with a target: the session's compile,
/// feed and finish entry points compile for it. Null `options` takes the
/// defaults (gfx1151); a null, unknown or unsupported target fails as a
/// mismatched `struct_size` does.
///
/// # Safety
///
/// `options` must be null or point to a valid v2 options struct whose
/// `target` is null or NUL-terminated; `session` as for
/// `rocke_codeobj_session_create_v1`.
#[no_mangle]
pub unsafe extern "C" fn rocke_codeobj_session_create_v2(
    options: *const RockeCodeobjSessionOptionsV2,
    session: *mut *mut Session,
) -> i32 {
    if session.is_null() {
        return 1;
    }
    unsafe { *session = ptr::null_mut() };
    let (worker_threads, target) = if options.is_null() {
        (0, Target::Gfx1151)
    } else if unsafe { (*options).struct_size } != mem::size_of::<RockeCodeobjSessionOptionsV2>() as u32 {
        return 1;
    } else {
        match unsafe { input_target((*options).target) } {
            Ok(target) => (unsafe { (*options).worker_threads }, target),
            Err(_) => return 1,
        }
    };
    match catch_unwind(|| Box::new(Session::new(SessionOptions { worker_threads, target }))) {
        Ok(created) => {
            unsafe { *session = Box::into_raw(created) };
            0
        }
        Err(_) => 1,
    }
}

/// Stops and joins the session's threads and frees it. Null is ignored.
///
/// # Safety
///
/// `session` must be null or a session from `rocke_codeobj_session_create_v1`
/// that is not freed yet and that no other call is using.
#[no_mangle]
pub unsafe extern "C" fn rocke_codeobj_session_free_v1(session: *mut Session) {
    if !session.is_null() {
        drop(unsafe { Box::from_raw(session) });
    }
}

/// `rocke_codeobj_compile_rocke_ir_v1` with the parse on the session's
/// threads; the result is the same.
///
/// # Safety
///
/// As `rocke_codeobj_compile_rocke_ir_v1`, and `session` must be a live
/// session.
#[no_mangle]
pub unsafe extern "C" fn rocke_codeobj_session_compile_v1(
    session: *mut Session,
    request: *const RockeDirectGemmCompileRequestV1,
    result: *mut RockeCodeobjCompileResultV1,
) -> i32 {
    if unsafe { reset_compile_result(result) }.is_err() {
        return 1;
    }
    let result = unsafe { &mut *result };
    let operation = catch_unwind(AssertUnwindSafe(|| {
        let session = unsafe { session.as_ref() }.ok_or_else(|| CompileError::new("request", "session is null"))?;
        if request.is_null()
            || unsafe { (*request).struct_size } != mem::size_of::<RockeDirectGemmCompileRequestV1>() as u32
        {
            return Err(CompileError::new("request", "rocKE IR request has an incompatible struct_size"));
        }
        let request = unsafe { &*request };
        let bytes = unsafe { input_bytes(request.serialized_ir, request.serialized_ir_size, "serialized_ir") }
            .map_err(|message| CompileError::new("request", message))?;
        let ir: Arc<str> = str::from_utf8(bytes)
            .map(Arc::from)
            .map_err(|error| CompileError::new("request", format!("serialized_ir is not UTF-8: {error}")))?;
        let start = std::time::Instant::now();
        let compiled = session.compile_shared(&ir)?;
        Ok((compiled, start.elapsed().as_secs_f64()))
    }));
    set_compile_outcome(result, operation)
}

/// Appends `size` bytes at `data` to the input being streamed into the
/// session (starting a stream if none is), and returns 0; complete lines are
/// parsed on the session's threads while the caller produces the rest.
/// Returns 1, changing nothing, for a null session, null `data` with a
/// nonzero `size`, or a call that overlaps another on the session.
///
/// # Safety
///
/// `session` must be null or a live session; `data` must be readable for
/// `size` bytes during the call (it is copied).
#[no_mangle]
pub unsafe extern "C" fn rocke_codeobj_session_feed_v1(session: *mut Session, data: *const u8, size: usize) -> i32 {
    let Some(session) = (unsafe { session.as_ref() }) else {
        return 1;
    };
    if data.is_null() && size > 0 {
        return 1;
    }
    let bytes = if size == 0 { &[][..] } else { unsafe { slice::from_raw_parts(data, size) } };
    match catch_unwind(AssertUnwindSafe(|| session.feed(bytes))) {
        Ok(Ok(())) => 0,
        _ => 1,
    }
}

/// Ends the stream and compiles everything fed since the last finish, with
/// the result `rocke_codeobj_compile_rocke_ir_v1` gives for the whole input
/// (an empty or non-UTF-8 input fails the same way). `executable_seconds`
/// is the time from this call on.
///
/// # Safety
///
/// `session` must be null or a live session; `result` as for
/// `rocke_codeobj_compile_rocke_ir_v1`.
#[no_mangle]
pub unsafe extern "C" fn rocke_codeobj_session_finish_v1(
    session: *mut Session,
    result: *mut RockeCodeobjCompileResultV1,
) -> i32 {
    if unsafe { reset_compile_result(result) }.is_err() {
        return 1;
    }
    let result = unsafe { &mut *result };
    let operation = catch_unwind(AssertUnwindSafe(|| {
        let session = unsafe { session.as_ref() }.ok_or_else(|| CompileError::new("request", "session is null"))?;
        let start = std::time::Instant::now();
        let compiled = session.finish()?;
        Ok((compiled, start.elapsed().as_secs_f64()))
    }));
    set_compile_outcome(result, operation)
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

    const PROBE: &str = "ckdsl.ir v1\nkernel @probe {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    \
                         %Y : ptr<f32,global>\n  }\n  region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
                         %e : f32 = arith.bitcast ( %tid )\n    memref.global_store_typed ( %Y, %tid, %e ) { align = i:4 }\n    \
                         cf.return (  )\n  }\n}\n";

    /// HSACO and diagnostics without the phase timings, or the error text.
    fn outcome(status: i32, result: &mut RockeCodeobjCompileResultV1) -> (i32, Vec<u8>, String) {
        let diagnostics = if result.diagnostics.is_null() {
            String::new()
        } else {
            String::from_utf8_lossy(unsafe { slice::from_raw_parts(result.diagnostics, result.diagnostics_size) }).into_owned()
        };
        let summary = diagnostics.split(" phases_ms=").next().unwrap_or("").to_string();
        let hsaco = if result.hsaco.is_null() {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(result.hsaco, result.hsaco_size) }.to_vec()
        };
        unsafe { rocke_codeobj_compile_result_dispose_v1(result as *mut _) };
        (status, hsaco, summary)
    }

    #[test]
    fn sessions_compile_as_the_plain_entry_point() {
        let mut session = ptr::null_mut();
        let options =
            RockeCodeobjSessionOptionsV1 { struct_size: mem::size_of::<RockeCodeobjSessionOptionsV1>() as u32, worker_threads: 2 };
        assert_eq!(unsafe { rocke_codeobj_session_create_v1(&options, &mut session) }, 0);
        let bad = RockeCodeobjSessionOptionsV1 { struct_size: 3, worker_threads: 2 };
        let mut none = ptr::null_mut();
        assert_eq!(unsafe { rocke_codeobj_session_create_v1(&bad, &mut none) }, 1);
        assert!(none.is_null());
        let broken = PROBE.replace("%tid )\n", "%nope )\n");
        for text in [PROBE.as_bytes(), broken.as_bytes(), b"", b"ckdsl.ir v1\n\xff\n"] {
            let request = RockeDirectGemmCompileRequestV1 {
                struct_size: mem::size_of::<RockeDirectGemmCompileRequestV1>() as u32,
                serialized_ir: text.as_ptr(),
                serialized_ir_size: text.len(),
            };
            let mut plain = empty_compile_result();
            let status = unsafe { rocke_codeobj_compile_rocke_ir_v1(&request, &mut plain) };
            let reference = outcome(status, &mut plain);
            let mut via = empty_compile_result();
            let status = unsafe { rocke_codeobj_session_compile_v1(session, &request, &mut via) };
            assert_eq!(outcome(status, &mut via), reference, "{text:?}");
            for piece in [1usize, 5, 64] {
                for part in text.chunks(piece) {
                    assert_eq!(unsafe { rocke_codeobj_session_feed_v1(session, part.as_ptr(), part.len()) }, 0);
                }
                let mut streamed = empty_compile_result();
                let status = unsafe { rocke_codeobj_session_finish_v1(session, &mut streamed) };
                assert_eq!(outcome(status, &mut streamed), reference, "{text:?} in pieces of {piece}");
            }
        }
        assert_eq!(unsafe { rocke_codeobj_session_feed_v1(session, ptr::null(), 4) }, 1);
        unsafe { rocke_codeobj_session_free_v1(session) };
        unsafe { rocke_codeobj_session_free_v1(ptr::null_mut()) };
    }

    fn v2_request(text: &[u8], target: *const c_char) -> RockeIrCompileRequestV2 {
        RockeIrCompileRequestV2 {
            struct_size: mem::size_of::<RockeIrCompileRequestV2>() as u32,
            serialized_ir: text.as_ptr(),
            serialized_ir_size: text.len(),
            target,
        }
    }

    #[test]
    fn v2_compiles_for_its_target_and_refuses_the_rest() {
        let broken = PROBE.replace("%tid )\n", "%nope )\n");
        for text in [PROBE.as_bytes(), broken.as_bytes(), b"", b"ckdsl.ir v1\n\xff\n"] {
            let request = RockeDirectGemmCompileRequestV1 {
                struct_size: mem::size_of::<RockeDirectGemmCompileRequestV1>() as u32,
                serialized_ir: text.as_ptr(),
                serialized_ir_size: text.len(),
            };
            let mut plain = empty_compile_result();
            let status = unsafe { rocke_codeobj_compile_rocke_ir_v1(&request, &mut plain) };
            let reference = outcome(status, &mut plain);
            let mut via = empty_compile_result();
            let status = unsafe { rocke_codeobj_compile_rocke_ir_v2(&v2_request(text, c"gfx1151".as_ptr()), &mut via) };
            assert_eq!(outcome(status, &mut via), reference, "{text:?}");
        }
        for target in [c"gfx942".as_ptr(), c"".as_ptr(), ptr::null()] {
            let mut result = empty_compile_result();
            let status = unsafe { rocke_codeobj_compile_rocke_ir_v2(&v2_request(PROBE.as_bytes(), target), &mut result) };
            let (status, hsaco, diagnostics) = outcome(status, &mut result);
            assert_eq!((status, hsaco.len()), (1, 0));
            assert!(diagnostics.starts_with("request: "), "{diagnostics}");
        }
        // gfx950 compiles the kernel to a gfx950 code object, and refuses an
        // operation it does not lower as unsupported.
        let mut result = empty_compile_result();
        let status = unsafe { rocke_codeobj_compile_rocke_ir_v2(&v2_request(PROBE.as_bytes(), c"gfx950".as_ptr()), &mut result) };
        let (status, hsaco, _) = outcome(status, &mut result);
        assert_eq!(status, 0);
        assert_eq!(u32::from_le_bytes(hsaco[48..52].try_into().unwrap()), 0x54f);
        let permlane = PROBE.replace("%e : f32 = arith.bitcast ( %tid )", "%p : i32 = tile.permlane64 ( %tid )\n    %e : f32 = arith.bitcast ( %p )");
        let mut result = empty_compile_result();
        let status = unsafe { rocke_codeobj_compile_rocke_ir_v2(&v2_request(permlane.as_bytes(), c"gfx950".as_ptr()), &mut result) };
        let (status, hsaco, diagnostics) = outcome(status, &mut result);
        assert_eq!((status, hsaco.len()), (1, 0));
        assert!(diagnostics.starts_with("ROCKE_DIRECT_UNSUPPORTED: "), "{diagnostics}");
        let mut request = v2_request(PROBE.as_bytes(), c"gfx1151".as_ptr());
        request.struct_size = mem::size_of::<RockeDirectGemmCompileRequestV1>() as u32;
        let mut result = empty_compile_result();
        let status = unsafe { rocke_codeobj_compile_rocke_ir_v2(&request, &mut result) };
        assert!(outcome(status, &mut result).2.contains("struct_size"));
        let supported = unsafe { CStr::from_ptr(rocke_codeobj_supported_targets_v1()) };
        assert_eq!(supported.to_str().ok(), Some("gfx1151,gfx950"));
    }

    #[test]
    fn v2_sessions_compile_for_their_target() {
        let options = |target: *const c_char| RockeCodeobjSessionOptionsV2 {
            struct_size: mem::size_of::<RockeCodeobjSessionOptionsV2>() as u32,
            worker_threads: 1,
            target,
        };
        let request = RockeDirectGemmCompileRequestV1 {
            struct_size: mem::size_of::<RockeDirectGemmCompileRequestV1>() as u32,
            serialized_ir: PROBE.as_ptr(),
            serialized_ir_size: PROBE.len(),
        };
        let mut plain = empty_compile_result();
        let status = unsafe { rocke_codeobj_compile_rocke_ir_v1(&request, &mut plain) };
        let reference = outcome(status, &mut plain);
        assert_eq!(reference.0, 0);
        for opts in [Some(options(c"gfx1151".as_ptr())), None] {
            let mut session = ptr::null_mut();
            let opts_ptr = opts.as_ref().map_or(ptr::null(), |o| o as *const _);
            assert_eq!(unsafe { rocke_codeobj_session_create_v2(opts_ptr, &mut session) }, 0);
            assert_eq!(unsafe { (*session).target() }, Target::Gfx1151);
            let mut via = empty_compile_result();
            let status = unsafe { rocke_codeobj_session_compile_v1(session, &request, &mut via) };
            assert_eq!(outcome(status, &mut via), reference);
            unsafe { rocke_codeobj_session_free_v1(session) };
        }
        let mut session = ptr::null_mut();
        assert_eq!(unsafe { rocke_codeobj_session_create_v2(&options(c"gfx950".as_ptr()), &mut session) }, 0);
        assert_eq!(unsafe { (*session).target() }, Target::Gfx950);
        let mut via = empty_compile_result();
        let status = unsafe { rocke_codeobj_session_compile_v1(session, &request, &mut via) };
        let (status, hsaco, _) = outcome(status, &mut via);
        assert_eq!((status, u32::from_le_bytes(hsaco[48..52].try_into().unwrap())), (0, 0x54f));
        unsafe { rocke_codeobj_session_free_v1(session) };
        for target in [c"gfx942".as_ptr(), c"gfx9".as_ptr(), ptr::null()] {
            let mut session = ptr::null_mut();
            assert_eq!(unsafe { rocke_codeobj_session_create_v2(&options(target), &mut session) }, 1);
            assert!(session.is_null());
        }
        let mut short = options(c"gfx1151".as_ptr());
        short.struct_size = mem::size_of::<RockeCodeobjSessionOptionsV1>() as u32;
        let mut session = ptr::null_mut();
        assert_eq!(unsafe { rocke_codeobj_session_create_v2(&short, &mut session) }, 1);
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
