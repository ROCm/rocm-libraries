*************************************************************************** /

#include "testing.hpp"

    template <typename I, typename X, typename Y, typename T>
    void testing_axpby_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle      local_handle;
    rocsparse_handle            handle = local_handle;
    rocsparse_const_spvec_descr x      = (rocsparse_const_spvec_descr)0x4;
    rocsparse_dnvec_descr       y      = (rocsparse_dnvec_descr)0x4;
    const void*                 alpha  = (const void*)0x4;
    const void*                 beta   = (const void*)0x4;
    bad_arg_analysis(rocsparse_axpby, handle, alpha, x, beta, y);
}

template <typename I, typename X, typename Y, typename T>
void testing_axpby(const Arguments& arg)
{
    I size = arg.M;
    I nnz  = arg.nnz;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    rocsparse_index_base base = arg.baseA;

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  xtype = get_datatype<X>();
    rocsparse_datatype  ytype = get_datatype<Y>();

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Allocate host memory for matrix
    host_vector<I> hx_ind(nnz);
    host_vector<X> hx_val(nnz);
    host_vector<Y> hy_1(size);
    host_vector<Y> hy_2(size);
    host_vector<Y> hy_gold(size);

    // Initialize data on CPU
    rocsparse_seedrand();
    rocsparse_init_index(hx_ind, nnz, base, size + base);