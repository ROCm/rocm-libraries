/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/py_path_split.h -- os.path.split, as C.
 *
 * The lowerer names source files in its DIFile nodes the way Python's
 * os.path.split does, because the Python engine builds those nodes with that
 * call and the two engines' text has to match byte for byte. os.path is ntpath
 * on Windows and posixpath everywhere else, and the two disagree: ntpath splits
 * on '\' as well as '/' and understands drive and UNC roots. So a native
 * Windows location such as "C:\proj\k.py" is a directory plus a basename to one
 * module and one long basename to the other, and splitting it the POSIX way on
 * Windows put the whole path in the filename with an empty directory.
 *
 * `windows` selects the module rather than the build selecting it, so a POSIX
 * host can exercise the Windows behaviour; ROCKE_PY_PATH_WINDOWS is the answer
 * for the host being compiled for.
 */
#ifndef ROCKE_PY_PATH_SPLIT_H
#define ROCKE_PY_PATH_SPLIT_H

#include <stdbool.h>
#include <stddef.h>

#ifdef _WIN32
#define ROCKE_PY_PATH_WINDOWS true
#else
#define ROCKE_PY_PATH_WINDOWS false
#endif

/* Where os.path.split cuts a path: the first `head_len` bytes are the
 * directory, and the basename starts at `tail_off`. The two halves are not
 * always adjacent -- repeated separators between them belong to neither -- which
 * is why this reports both ends instead of one split point. */
typedef struct
{
    size_t head_len;
    size_t tail_off;
} rocke_py_path_split_t;

static inline bool rocke_py_path_is_sep(char c, bool windows)
{
    return c == '/' || (windows && c == '\\');
}

/* Bytes of drive and root that ntpath.splitroot holds back, which the split
 * below neither scans nor strips: "C:" and its following separator, a leading
 * separator on its own, or the "\\server\share\" and "\\?\C:\" roots a path
 * opening with two separators carries. posixpath has no such notion, so this is
 * Windows-only and always treats both separators as separators.
 *
 * A path starting with a separator is never a drive, which is the order
 * splitroot tests in: "/:" is the root "/" plus the name ":", not the drive
 * ":". */
static inline size_t rocke_py_path_nt_stem(const char* path, size_t n)
{
    if(n >= 1 && rocke_py_path_is_sep(path[0], true))
    {
        if(n < 2 || !rocke_py_path_is_sep(path[1], true))
        {
            return 1; /* a rooted relative path, e.g. "\Windows" */
        }
        /* "\\?\UNC\server\share" names a host and a share after a verbatim
         * prefix, so the scan starts past that prefix and finds them where a
         * plain UNC path keeps them. Matched case-insensitively, as splitroot
         * upper-cases first. */
        size_t start = 2;
        if(n >= 8 && path[2] == '?' && rocke_py_path_is_sep(path[3], true)
           && (path[4] == 'U' || path[4] == 'u') && (path[5] == 'N' || path[5] == 'n')
           && (path[6] == 'C' || path[6] == 'c') && rocke_py_path_is_sep(path[7], true))
        {
            start = 8;
        }
        size_t host = start;
        while(host < n && !rocke_py_path_is_sep(path[host], true))
        {
            host++;
        }
        if(host >= n) /* nothing after the host: all of it is the drive */
        {
            return n;
        }
        size_t share = host + 1;
        while(share < n && !rocke_py_path_is_sep(path[share], true))
        {
            share++;
        }
        /* The separator closing the share is the root; past the end there is
         * no root and the whole path is the drive. */
        return share >= n ? n : share + 1;
    }
    if(n >= 2 && path[1] == ':')
    {
        return (n >= 3 && rocke_py_path_is_sep(path[2], true)) ? 3 : 2;
    }
    return 0;
}

/* os.path.split over `path[0:n]`, which need not be NUL-terminated. */
static inline rocke_py_path_split_t rocke_py_path_split(const char* path, size_t n, bool windows)
{
    rocke_py_path_split_t out;
    const size_t stem = windows ? rocke_py_path_nt_stem(path, n) : 0;

    size_t cut = n;
    while(cut > stem && !rocke_py_path_is_sep(path[cut - 1], windows))
    {
        cut--;
    }
    out.tail_off = cut;

    size_t head = cut;
    while(head > stem && rocke_py_path_is_sep(path[head - 1], windows))
    {
        head--;
    }
    /* Trailing separators drop off the directory. posixpath, having carved out
     * no root, instead keeps a directory that is nothing but separators, which
     * is how "/x.py" gets "/" and "//x.py" gets "//". */
    out.head_len = (windows || head > 0) ? head : cut;
    return out;
}

#endif /* ROCKE_PY_PATH_SPLIT_H */
