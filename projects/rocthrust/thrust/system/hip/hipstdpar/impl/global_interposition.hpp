/*
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file thrust/system/hip/hipstdpar/impl/global_interposition.hpp
 *  \brief Namespace-scope variable host-to-accelerator binding implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#if __has_include(<link.h>) && __has_include(<gelf.h>) && __has_include(<sys/mman.h>) && defined(__CLANG_RDC__)
    #include <gelf.h>
    #include <link.h>
    #include <sys/mman.h>
    #define __HIPSTDPAR_HAS_TRUE_GLOBALS__
#endif

#if defined(__HIPSTDPAR_HAS_TRUE_GLOBALS__)
#include <cstddef>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace hipstd
{
class __Global_binder;
} // Namespace hipstd.

extern "C" {
    inline __constant__ class {
        friend class hipstd::__Global_binder;

        std::size_t n_;
        const struct {
            const char* name_;
            void** address_;
        }* symbols_;
        std::remove_const_t<std::remove_reference_t<decltype(*symbols_)>> _;
    } __hipstdpar_symbol_indirection_table{};
} // extern "C"

namespace hipstd
{
class __Global_binder final {
    // DATA
    hipDevice_t d_{};
    decltype(__hipstdpar_symbol_indirection_table)* ps_{};
    ::std::deque<::std::size_t> ud_{};

    // IMPLEMENTATION - MANIPULATORS
    void define_symbols_(const dl_phdr_info* info)
    {
        if (::std::empty(ud_)) return;

        ::std::string_view so{info->dlpi_name};

        if (::std::empty(so)) so = "/proc/self/exe";
        else if (so.find("linux-vdso") != ::std::string_view::npos) return;

        const auto fsz{::std::filesystem::file_size(so)};
        ::std::unique_ptr<::std::FILE, decltype(::std::fclose)*> f{
            ::std::fopen(::std::data(so), "r"), ::std::fclose};
        ::std::unique_ptr<char[], ::std::function<void(char*)>> t{
            static_cast<char*>(mmap(nullptr, fsz, PROT_READ, MAP_SHARED,
                                    fileno(f.get()), 0)),
            [=](char* p) { munmap(p, fsz); }
        };

        const auto ehdr{reinterpret_cast<const GElf_Ehdr*>(&t[0])};
        const auto shdr{reinterpret_cast<const GElf_Shdr*>(&t[ehdr->e_shoff])};

        ::std::unordered_map<::std::string_view, const GElf_Sym*> symtab{};
        ::std::for_each_n(shdr, ehdr->e_shnum, [&, shdr](auto&& x) {
            if (::std::empty(ud_)) return;
            if (x.sh_type != SHT_SYMTAB && x.sh_type != SHT_DYNSYM) return;

            const auto& strtab_hdr{shdr[x.sh_link]};
            const auto syms{reinterpret_cast<const GElf_Sym*>(&t[x.sh_offset])};
            ::std::for_each_n(syms, x.sh_size / x.sh_entsize, [&](auto&& y) {
                symtab[&t[strtab_hdr.sh_offset + y.st_name]] = &y;
            });
        });

        for (auto i = 0u; i != ::std::size(ud_); ++i) {
            auto sidx{ud_.back()};
            ud_.pop_back();

            const auto it{symtab.find(ps_->symbols_[sidx].name_)};
            if (it != ::std::cend(symtab)) {
                *ps_->symbols_[sidx].address_ = reinterpret_cast<void*>(
                    info->dlpi_addr + it->second->st_value);
            }
            else ud_.push_front(sidx);
        }
    }

    // IMPLEMENTATION - ACCESSORS
    void make_segments_accessible_(const dl_phdr_info* info) const
    {
        const auto base = reinterpret_cast<::std::uint8_t*>(info->dlpi_addr);
        ::std::for_each_n(info->dlpi_phdr, info->dlpi_phnum, [=](auto&& h) {
            if (h.p_type != PT_LOAD && h.p_type != PT_GNU_RELRO) return;

            const auto p = base + h.p_vaddr;
            if (hipMemAdvise(p, h.p_memsz, hipMemAdviseSetAccessedBy, d_) !=
                hipSuccess) {
                throw ::std::runtime_error("Failed to make segment accessible");
            }
        });
    }
public:
    // STATICS
    static void bind()
    {
        __attribute__((used)) static const __Global_binder r{};
    }
    // CREATORS
    __Global_binder()
    {
        if (hipGetDevice(&d_) != hipSuccess) {
            throw ::std::runtime_error(
                "Failed to retrieve accelerator for HIPSTDPAR");
        }
        if (hipGetSymbolAddress(reinterpret_cast<void**>(&ps_),
                                __hipstdpar_symbol_indirection_table) !=
            hipSuccess) {
            return; // Compiler did not set this up for us.
        }

        for (auto i = 0u; i != ps_->n_; ++i) ud_.push_back(i);

        dl_iterate_phdr([](dl_phdr_info* i, size_t, void* p) {
            const auto self{static_cast<__Global_binder*>(p)};

            if (__HIPSTDPAR_INTERPOSE_ALLOC__) {
                self->make_segments_accessible_(i);
            }

            self->define_symbols_(i);

            return 0;
        }, this);
    }
    ~__Global_binder()
    {
        if (!__HIPSTDPAR_INTERPOSE_ALLOC__ || !ps_) return;

        dl_iterate_phdr([](dl_phdr_info* i, size_t, void* p) {
            const auto self{static_cast<__Global_binder*>(p)};
            const auto base = reinterpret_cast<::std::uint8_t*>(i->dlpi_addr);

            ::std::for_each_n(i->dlpi_phdr, i->dlpi_phnum, [=](auto&& h) {
                if (h.p_type != PT_LOAD && h.p_type != PT_GNU_RELRO) return;

                const auto p = base + h.p_vaddr;
                if (hipMemAdvise(p, h.p_memsz, hipMemAdviseUnsetAccessedBy,
                                 self->d_) !=
                    hipSuccess) {
                    throw ::std::runtime_error(
                        "Failed to unset loadable segment accessiblity.");
                }
            });

            return 0;
        }, this);
    }
};

inline void __maybe_bind_globals()
{   // This has to ve invoked AFTER static init has completed.
    return __Global_binder::bind();
}
} // Namespace hipstd.
#else // __HIPSTDPAR_HAS_TRUE_GLOBALS__
namespace hipstd
{
inline void __maybe_bind_globals() {}
} // Namespace hipstd.
#endif // __HIPSTDPAR_HAS_TRUE_GLOBALS__

#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
