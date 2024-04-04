#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "./lib/cpu/sparse_linear.cpp"
#include "./lib/cpu/sparse_conv2d.cpp"
#include "./lib/cpu/sparse_conv2d_over_on.cpp"
#include "./lib/cpu/spmm.cpp"
#include "./lib/cpu/sddmm.cpp"
#include "./lib/cpu/csr_add.cpp"
#include "./lib/cpu/utils.cpp"

PYBIND11_MODULE(spops_backend_cpu, m)
{
    // linear
    m.def("sparse_linear_vectorized_forward", &sparse_linear_vectorized_forward_wrapper);
    m.def("sparse_linear_vectorized_backward", &sparse_linear_vectorized_backward_wrapper);

    // conv2d
    m.def("sparse_conv2d_vectorized_forward_stride_1", &sparse_conv2d_vectorized_forward_stride_1_wrapper);
    m.def("sparse_conv2d_vectorized_backward_stride_1", &sparse_conv2d_vectorized_backward_stride_1_wrapper);
    m.def("sparse_conv2d_vectorized_forward_stride_2", &sparse_conv2d_vectorized_forward_stride_2_wrapper);
    m.def("sparse_conv2d_vectorized_backward_stride_2", &sparse_conv2d_vectorized_backward_stride_2_wrapper);

    // conv2d over on
    m.def("sparse_conv2d_vectorized_forward_over_on_stride_1", &sparse_conv2d_vectorized_forward_over_on_stride_1_wrapper);
    m.def("sparse_conv2d_vectorized_backward_over_on_stride_1", &sparse_conv2d_vectorized_backward_over_on_stride_1_wrapper);
    m.def("sparse_conv2d_vectorized_backward_over_on_stride_2", &sparse_conv2d_vectorized_backward_over_on_stride_2_wrapper);

    // utils
    m.def("transpose", &transpose_wrapper);
    m.def("sparsify_conv2d", &sparsify_conv2d_wrapper);
    m.def("densify_conv2d", &densify_conv2d_wrapper);
    m.def("further_sparsify_conv2d", &further_sparsify_conv2d_wrapper);


    // general
    m.def("spmm", &spmm_wrapper);
    m.def("sddmm", &sddmm_wrapper);
    m.def("sddmm_v2", &sddmm_v2_wrapper);
    m.def("sddmm_v3", &sddmm_v3_wrapper);
    m.def("csr_add", &csr_add_wrapper);
    m.def("sddmm_coo", &sddmm_coo_wrapper);
    m.def("sddmm_coo_v2", &sddmm_coo_v2_wrapper);
    m.def("same_nnz_bspmm", &same_nnz_bspmm_wrapper);
    m.def("same_nnz_bsddmm", &same_nnz_bsddmm_wrapper);
    m.def("same_nnz_bsddmm_coo", &same_nnz_bsddmm_coo_wrapper);
}