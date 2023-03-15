#ifndef __GPU___TYPE_TRAITS_DEPENDENT_TYPE_H__
#define __GPU___TYPE_TRAITS_DEPENDENT_TYPE_H__

#include "gpu/__config"

namespace gpu {

template <class _Tp, bool>
struct _LIBGPU_TEMPLATE_VIS __dependent_type : public _Tp {};

} // namespace gpu


#endif // __GPU___TYPE_TRAITS_DEPENDENT_TYPE_H__
