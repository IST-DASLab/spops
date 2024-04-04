#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <immintrin.h>

namespace py = pybind11;

// C = AB
// A: (p, r), csr
// B: (r, q), dense
// C: (p, q), dense
void spmm(int p, int q, int r, int A_nnz, float* A_val, int* A_row_offsets, int* A_col_idx, float* B, float* C) {
    #pragma omp parallel
	{
        #pragma omp for
		for(int i = 0; i < p; i++){
			int k = A_row_offsets[i];
            int k_end = A_row_offsets[i + 1];
			for(; k < k_end; k++){
				int idx = A_col_idx[k];
				__m256 a = _mm256_set1_ps(A_val[k]);
				int j = 0;

				for(; j < q-7; j+=8){
					__m256 b = _mm256_loadu_ps(B + (idx * q + j));
					__m256 c = _mm256_loadu_ps(C + (i * q + j));

					c = _mm256_fmadd_ps(b, a, c);

					_mm256_storeu_ps(C + (i * q + j), c);
				}

				for(; j < q; j++){
					C[i * q + j] += A_val[k] * B[idx * q + j];
				}
			}
		}
	}
}

// C = batched_mm(A, B)
// A: b x p x r (same_size_bcsr)
// A_row_offsets: b x (p + 1)
// A_val, A_col_idx: b x A_nnz
// B: b x r x q (dense)
// C: b x p x q (dense)
void same_nnz_bspmm(int b, int p, int q, int r, int A_nnz, float* A_val, int* A_row_offsets,
                    int* A_col_idx, float* B, float* C) {
    #pragma omp parallel
	{
        #pragma omp for collapse(2)
        for (int z = 0; z < b; z++){
            for(int i = 0; i < p; i++){
                int k = A_row_offsets[z * (p + 1) + i];
                int k_end = A_row_offsets[z * (p + 1) + i + 1];
                for(; k < k_end; k++){
                    int idx = A_col_idx[z * A_nnz + k];
                    __m256 a = _mm256_set1_ps(A_val[z * A_nnz + k]);
                    int j = 0;

                    for(; j < q-7; j+=8){
                        __m256 b = _mm256_loadu_ps(B + (z * r * q + idx * q + j));
                        __m256 c = _mm256_loadu_ps(C + (z * p * q + i * q + j));

                        c = _mm256_fmadd_ps(b, a, c);

                        _mm256_storeu_ps(C + (z * p * q + i * q + j), c);
                    }

                    for(; j < q; j++){
                        C[z * p * q + i * q + j] += A_val[z * A_nnz + k] * B[z * r * q + idx * q + j];
                    }
                }
            }
        }
	}
}

// ====================================== Wrappers ===========================================

void spmm_wrapper(py::array_t<float> A_val, py::array_t<int> A_row_offsets, 
                  py::array_t<int> A_col_idx, py::array_t<float> B, py::array_t<float> C) {

    int p = C.shape()[0];
    int r = B.shape()[0];
    int q = B.shape()[1];
    int A_nnz = A_val.shape()[0];

    auto buf_A_val = A_val.request();
    auto buf_A_row_offsets = A_row_offsets.request();
    auto buf_A_col_idx = A_col_idx.request();
    auto buf_B = B.request();
    auto buf_C = C.request();

    float* ptr_A_val = (float*) buf_A_val.ptr;
    int* ptr_A_row_offsets = (int*) buf_A_row_offsets.ptr;
    int* ptr_A_col_idx = (int*) buf_A_col_idx.ptr;
    float* ptr_B = (float*) buf_B.ptr;
    float* ptr_C = (float*) buf_C.ptr;

    spmm(p, q, r, A_nnz, ptr_A_val, ptr_A_row_offsets, ptr_A_col_idx, ptr_B, ptr_C);
}

void same_nnz_bspmm_wrapper(py::array_t<float> A_val, py::array_t<int> A_row_offsets, 
                            py::array_t<int> A_col_idx, py::array_t<float> B, py::array_t<float> C) {
    // A: b x p x r (same_size_bcsr)
    // B: b x r x q (dense)
    // C: b x p x q (dense)

    int b = C.shape()[0];
    int p = C.shape()[1];
    int r = B.shape()[1];
    int q = B.shape()[2];
    int A_nnz = A_val.shape()[1];

    auto buf_A_val = A_val.request();
    auto buf_A_row_offsets = A_row_offsets.request();
    auto buf_A_col_idx = A_col_idx.request();
    auto buf_B = B.request();
    auto buf_C = C.request();

    float* ptr_A_val = (float*) buf_A_val.ptr;
    int* ptr_A_row_offsets = (int*) buf_A_row_offsets.ptr;
    int* ptr_A_col_idx = (int*) buf_A_col_idx.ptr;
    float* ptr_B = (float*) buf_B.ptr;
    float* ptr_C = (float*) buf_C.ptr;

    same_nnz_bspmm(b, p, q, r, A_nnz, ptr_A_val, ptr_A_row_offsets, ptr_A_col_idx, ptr_B, ptr_C);
}