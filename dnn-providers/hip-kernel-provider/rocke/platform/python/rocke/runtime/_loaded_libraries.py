# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Cold inventory of actual compiler/runtime API providers, not guessed filenames.

References returned to the owner pin already-loaded modules. No probe opens an
unmapped compiler, and dependent/proxy modules are attributed to symbol owners.
"""
from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass

from ._compiler_query import ComgrError, file_signature


@dataclass(frozen=True)
class LoadedLibrary:
    library: object
    path: str
    signature: tuple[int, int, int, int]


def _windows(
    symbols=(b"amd_comgr_get_version", b"amd_comgr_do_action")
) -> list[LoadedLibrary]:
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcAddress.argtypes = [wintypes.HMODULE, ctypes.c_char_p]
    kernel.GetProcAddress.restype = ctypes.c_void_p
    kernel.GetModuleHandleExW.argtypes = [
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.HMODULE),
    ]
    kernel.GetModuleHandleExW.restype = wintypes.BOOL
    kernel.GetModuleFileNameW.argtypes = [
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    kernel.GetModuleFileNameW.restype = wintypes.DWORD
    kernel.FreeLibrary.argtypes = [wintypes.HMODULE]
    psapi.EnumProcessModules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    process = kernel.GetCurrentProcess()
    count = 256
    while True:
        modules = (wintypes.HMODULE * count)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModules(
            process, modules, ctypes.sizeof(modules), ctypes.byref(needed)
        ):
            raise ComgrError("cannot inspect loaded module inventory")
        if needed.value <= ctypes.sizeof(modules):
            break
        count = needed.value // ctypes.sizeof(wintypes.HMODULE)
    result, retained = [], set()
    try:
        for module in modules[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
            version = kernel.GetProcAddress(module, symbols[0])
            action = kernel.GetProcAddress(module, symbols[1])
            if not version or not action:
                continue
            owner, action_owner = wintypes.HMODULE(), wintypes.HMODULE()
            if not kernel.GetModuleHandleExW(
                0x6, version, ctypes.byref(owner)
            ) or not kernel.GetModuleHandleExW(0x6, action, ctypes.byref(action_owner)):
                raise ComgrError("cannot identify loaded COMGR symbol owner")
            if owner.value != action_owner.value:
                raise ComgrError("COMGR API symbols have different module owners")
            if owner.value in retained:
                continue
            pinned = wintypes.HMODULE()
            if not kernel.GetModuleHandleExW(0x4, version, ctypes.byref(pinned)):
                raise ComgrError("cannot retain loaded COMGR module")
            retained.add(pinned.value)
            buf = ctypes.create_unicode_buffer(32768)
            size = kernel.GetModuleFileNameW(pinned, buf, len(buf))
            if not 0 < size < len(buf):
                raise ComgrError("cannot identify loaded COMGR file")
            path = buf.value
            library = ctypes.CDLL(path, handle=pinned.value)
            result.append(LoadedLibrary(library, path, file_signature(path)))
        return result
    except BaseException:
        for handle in retained:
            kernel.FreeLibrary(handle)
        raise


def _linux(
    symbols=(b"amd_comgr_get_version", b"amd_comgr_do_action")
) -> list[LoadedLibrary]:
    import _ctypes

    class Info(ctypes.Structure):
        _fields_ = [("address", ctypes.c_void_p), ("name", ctypes.c_char_p)]

    class DlInfo(ctypes.Structure):
        _fields_ = [
            ("filename", ctypes.c_char_p),
            ("base", ctypes.c_void_p),
            ("symbol", ctypes.c_char_p),
            ("address", ctypes.c_void_p),
        ]

    process = ctypes.CDLL(None)
    callback_type = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.POINTER(Info), ctypes.c_size_t, ctypes.c_void_p
    )
    modules = []

    @callback_type
    def collect(info, size, data):
        if size >= ctypes.sizeof(Info):
            modules.append(
                (info.contents.address, os.fsdecode(info.contents.name or b""))
            )
        return 0

    iterate = process.dl_iterate_phdr
    iterate.argtypes, iterate.restype = [callback_type, ctypes.c_void_p], ctypes.c_int
    iterate(collect, None)
    lookup = process.dladdr
    lookup.argtypes, lookup.restype = [
        ctypes.c_void_p,
        ctypes.POINTER(DlInfo),
    ], ctypes.c_int
    mappings = []
    try:
        with open("/proc/self/maps") as file:
            for line in file:
                parts = line.rstrip("\n").split(maxsplit=5)
                if len(parts) == 6:
                    start, end = parts[0].split("-", 1)
                    major, minor = parts[3].split(":", 1)
                    mappings.append(
                        (
                            int(start, 16),
                            int(end, 16),
                            os.makedev(int(major, 16), int(minor, 16)),
                            int(parts[4]),
                            parts[5],
                        )
                    )
    except OSError as error:
        raise ComgrError(
            f"cannot inspect mapped COMGR file identities: {error}"
        ) from error
    attached, keep, result = {}, set(), []
    try:
        for base, name in modules:
            if name == "linux-vdso.so.1":
                continue  # Kernel-supplied pseudo-module, not a user compiler.
            try:
                attached[base] = ctypes.CDLL(
                    name or None, mode=os.RTLD_NOLOAD | os.RTLD_NOW
                )
            except OSError as error:
                raise ComgrError(
                    f"cannot inspect already-loaded module {name!r}: {error}"
                ) from error
        for library in attached.values():
            try:
                version, action = (
                    getattr(library, symbols[0].decode()),
                    getattr(library, symbols[1].decode()),
                )
            except AttributeError:
                continue
            owners = []
            for fn in (version, action):
                address = ctypes.cast(fn, ctypes.c_void_p)
                info = DlInfo()
                if not lookup(address, ctypes.byref(info)):
                    raise ComgrError("cannot identify loaded COMGR symbol owner")
                owners.append((info.base, address.value))
            if owners[0][0] != owners[1][0]:
                raise ComgrError("COMGR API symbols have different module owners")
            base, address = owners[0]
            if base in keep:
                continue
            owner = attached.get(base)
            if owner is None:
                raise ComgrError("loaded COMGR owner disappeared during inventory")
            mapping = next((m for m in mappings if m[0] <= address < m[1]), None)
            if mapping is None:
                raise ComgrError("loaded COMGR mapping identity unavailable")
            path = mapping[4]
            try:
                signature = file_signature(path)
            except OSError as error:
                raise ComgrError(
                    f"mapped COMGR {path!r} file identity unavailable"
                ) from error
            if signature[:2] != mapping[2:4]:
                raise ComgrError(
                    f"mapped COMGR {path!r} no longer names its loaded file"
                )
            keep.add(base)
            result.append(LoadedLibrary(owner, path, signature))
        return result
    except BaseException:
        keep.clear()
        raise
    finally:
        for base, library in attached.items():
            if base not in keep:
                _ctypes.dlclose(library._handle)


def loaded_compilers() -> list[LoadedLibrary]:
    if sys.platform == "win32":
        return _windows()
    if sys.platform == "linux":
        return _linux()
    raise ComgrError("safe compiler admission supports Linux and Windows")


def loaded_runtimes() -> list[LoadedLibrary]:
    symbols = (b"hipGetDeviceCount", b"hipInit")
    if sys.platform == "win32":
        return _windows(symbols)
    if sys.platform == "linux":
        return _linux(symbols)
    raise ComgrError("safe runtime admission supports Linux and Windows")


def release(compiler: LoadedLibrary) -> None:
    if sys.platform == "win32":
        kernel = ctypes.WinDLL("kernel32")
        kernel.FreeLibrary.argtypes = [ctypes.c_void_p]
        kernel.FreeLibrary(compiler.library._handle)
    else:
        import _ctypes

        _ctypes.dlclose(compiler.library._handle)
