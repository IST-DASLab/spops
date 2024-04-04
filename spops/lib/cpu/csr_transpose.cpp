#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <immintrin.h>

namespace py = pybind11;

inline void csr_transpose(float* __restrict__ A_val, float* __restrict__ A_row_offsets,
                          float* __restrict__ A_col_idx, const int p, const int q, float* __restrict__ AT_val,
                          float* __restrict__ AT_row_offsets, float* __restrict__ AT_col_idx) {
    #pragma omp parallel for
    for(int i=0; i<N; i+=block_size) {
        for(int j=0; j<M; j+=block_size) {
            int max_i2 = i+block_size < N ? i + block_size : N;
            int max_j2 = j+block_size < M ? j + block_size : M;
            for(int i2=i; i2<max_i2; i2+=8) {
                for(int j2=j; j2<max_j2; j2+=8) {
                    tran(&X[i2*M +j2], &XT[j2*N + i2], M, N);
                }
            }
        }
    }
}


// ====================================== Wrappers ===========================================

void csr_transpose_wrapper(py::array_t<float> A_val, py::array_t<int> A_row_offsets, 
                           py::array_t<int> A_col_idx, int p, int q, py::array_t<float> AT_val,
                           py::array_t<int> AT_row_offsets, py::array_t<int> AT_col_idx) {

    auto buf_A_val = A_val.request();
    auto buf_A_row_offsets = A_row_offsets.request();
    auto buf_A_col_idx = A_col_idx.request();
    auto buf_AT_val = AT_val.request();
    auto buf_AT_row_offsets = AT_row_offsets.request();
    auto buf_AT_col_idx = AT_col_idx.request();

    float* ptr_A_val = (float*) buf_A_val.ptr;
    int* ptr_A_row_offsets = (int*) buf_A_row_offsets.ptr;
    int* ptr_A_col_idx = (int*) buf_A_col_idx.ptr;
    float* ptr_AT_val = (float*) buf_AT_val.ptr;
    int* ptr_AT_row_offsets = (int*) buf_AT_row_offsets.ptr;
    int* ptr_AT_col_idx = (int*) buf_AT_col_idx.ptr;
    

	csr_transpose(ptr_A_val, ptr_A_row_offsets, ptr_A_col_idx, p, q, ptr_AT_val, ptr_AT_row_offsets, ptr_AT_col_idx);
}