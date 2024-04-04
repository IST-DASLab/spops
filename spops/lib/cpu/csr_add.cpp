#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <immintrin.h>

namespace py = pybind11;

// B = A + B
// A: (p, q), csr
// B: (p, q), dense
void csr_add(int p, int q, float* A_val, int* A_row_offsets, int* A_col_idx, float* B) {
    #pragma omp parallel for
    for(int i = 0; i < p; i++){
        int k = A_row_offsets[i];
        int k_end = A_row_offsets[i + 1];
        for(; k < k_end-7; k+=8){
            union { __m256i j; int ja[8]; };
            j = _mm256_loadu_si256((__m256i*)(A_col_idx + k));
            union { __m256 av; float ava[8]; };
            av = _mm256_loadu_ps(A_val + k);

            union { __m256 bv; float bva[8]; };
            bv = _mm256_i32gather_ps(B + i * q, j, 4);

            // for (int s=0; s<8; s++){
            //     printf("av=%f ", ava[s]);
            //     printf("bv=%f ", bva[s]);
            //     printf("j=%d\n", ja[s]);
            // }

            bv = _mm256_add_ps(av, bv);
    
            // unfortunately only supported in avx512
            // _mm256_i32scatter_ps(B + i * q, j, bv, 4);

            for (int s=0; s<8; s++) {
                B[i * q + ja[s]] = bva[s];
            }
        }

        for(; k < k_end; k++){
            int j = A_col_idx[k];
            B[i * q + j] += A_val[k];
        }
    }
}

// ====================================== Wrappers ===========================================

void csr_add_wrapper(py::array_t<float> A_val, py::array_t<int> A_row_offsets, 
                  py::array_t<int> A_col_idx, py::array_t<float> B) {

    int p = B.shape()[0];
    int q = B.shape()[1];

    auto buf_A_val = A_val.request();
    auto buf_A_row_offsets = A_row_offsets.request();
    auto buf_A_col_idx = A_col_idx.request();
    auto buf_B = B.request();

    float* ptr_A_val = (float*) buf_A_val.ptr;
    int* ptr_A_row_offsets = (int*) buf_A_row_offsets.ptr;
    int* ptr_A_col_idx = (int*) buf_A_col_idx.ptr;
    float* ptr_B = (float*) buf_B.ptr;

    csr_add(p, q, ptr_A_val, ptr_A_row_offsets, ptr_A_col_idx, ptr_B);
}