#ifndef __GPU___ATOMIC_MEMORY_ORDER_H__
#define __GPU___ATOMIC_MEMORY_ORDER_H__

#include <atomic>
#include <type_traits>

namespace gpu {

using std::memory_order;
using std::memory_order_acq_rel;
using std::memory_order_acquire;
using std::memory_order_consume;
using std::memory_order_relaxed;
using std::memory_order_release;
using std::memory_order_seq_cst;

namespace internal {

typedef std::underlying_type_t<std::memory_order> __memory_order_underlying_t;

}

} // namespace gpu

#endif // __GPU___ATOMIC_MEMORY_ORDER_H__
