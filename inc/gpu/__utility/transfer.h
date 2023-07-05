#ifndef __GPU___UTILITY_TRANSFER_H__
#define __GPU___UTILITY_TRANSFER_H__

#include <type_traits>

namespace gpu {

struct transferToDevice_t { explicit transferToDevice_t() = default; };
// extern const transferToDevice_t transferToDevice;

struct transferToHost_t { explicit transferToHost_t() = default; };
// extern const transferToHost_t transferToHost;

// Transfer from host to device. Invokes the copy-transfer or move-transfer constructor and returns an rvalue.
template<typename _Tp, bool = std::is_trivially_copyable_v<std::remove_reference_t<_Tp> /* false */>
[[nodiscard]] std::remove_reference_t<_Tp>
forward_or_transfer(_Tp&& __t, transferToDevice_t) noexcept
{ return _Tp(std::forward<_Tp>(__t), transferToDevice_t()); }

// Transfer from device to host. Invokes the copy-transfer or move-transfer constructor and returns an rvalue.
template<typename _Tp, bool = std::is_trivially_copyable_v<std::remove_reference_t<_Tp> /* false */>
[[nodiscard]] std::remove_reference_t<_Tp>
forward_or_transfer(_Tp&& __t, transferToHost_t) noexcept
{ return _Tp(std::forward<_Tp>(__t), transferToHost_t()); }

// Forward only, no transfer
template<typename _Tp>
[[nodiscard]] _Tp&&
forward_or_transfer<_Tp, true>(_Tp&& __t, transferToDevice_t) noexcept
{ return std::forward<_Tp>(__t); }

// Forward only, no transfer
template<typename _Tp>
[[nodiscard]] _Tp&&
forward_or_transfer<_Tp, true>(_Tp&& __t, transferToHost_t) noexcept
{ return std::forward<_Tp>(__t); }

}

#endif // __GPU___UTILITY_TRANSFER_H__
