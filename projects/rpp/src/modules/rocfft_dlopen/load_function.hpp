/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef RPP_ROCFFT_DLOPEN_LOAD_FUNCTION_HPP
#define RPP_ROCFFT_DLOPEN_LOAD_FUNCTION_HPP

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif /* _WIN32 */

namespace rpp {
namespace rocfft_dlopen {

/*! \brief Resolve one symbol out of an already-opened shared library.
 *  \param handle Library handle from dlopen()/LoadLibrary().
 *  \param symbol Symbol name to resolve.
 *  \param fn     Function pointer to populate.
 *  \return true if the symbol was resolved, false otherwise.
 */
template <typename Fn>
bool load_function(void* handle, const char* symbol, Fn& fn) {
#ifdef _WIN32
    fn = reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(handle), symbol));
#else
    fn = reinterpret_cast<Fn>(dlsym(handle, symbol));
    (void)dlerror();  // clear any sticky error state for the next lookup
#endif /* _WIN32 */
    return fn != nullptr;
}

}  // namespace rocfft_dlopen
}  // namespace rpp

#endif  // RPP_ROCFFT_DLOPEN_LOAD_FUNCTION_HPP
