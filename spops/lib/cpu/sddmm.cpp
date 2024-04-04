#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <immintrin.h>

namespace py = pybind11;

void sddmm(int p, int q, int r, float* A, float* BT, int C_nnz, int* C_row_offsets, int* C_col_idx, float* C_val) {
    #pragma omp parallel
	{
		#pragma omp for
		for(int i = 0; i < p; i++){
			for(int j = C_row_offsets[i]; j < C_row_offsets[i+1]; j++){
				int col = C_col_idx[j];
				float sacc = 0;
				__m256 acc = _mm256_setzero_ps();

				int k = 0;
				for(; k < r-7; k+=8){
					__m256 a0 = _mm256_loadu_ps(A + (i * r + k));
					__m256 b0 = _mm256_loadu_ps(BT + (col * r + k));
					acc = _mm256_fmadd_ps(a0, b0, acc); 
                }
                
				//cleanup
				for(; k < r; k++){
					sacc += A[i * r + k] * BT[col * r + k];
				}

				//reduce sum
				const __m128 hiQuad0 = _mm256_extractf128_ps(acc, 1);
				const __m128 loQuad0 = _mm256_castps256_ps128(acc);
				const __m128 sumQuad0 = _mm_add_ps(loQuad0, hiQuad0);
				const __m128 hiDual0 = _mm_movehl_ps(sumQuad0, sumQuad0);
				const __m128 sumDual0 = _mm_add_ps(sumQuad0, hiDual0);
				const __m128 hi0 = _mm_shuffle_ps(sumDual0, sumDual0, 0x1);
				const __m128 sum0 = _mm_add_ss(sumDual0, hi0);

				C_val[j] = sacc + _mm_cvtss_f32(sum0);
			}
		}
	}
}

void sddmm_v2(int p, int q, int r, float* A, float* BT, int C_nnz, int* C_row_offsets, int* C_col_idx, float* C_val) {
    #pragma omp parallel
	{
		#pragma omp for
		for(int i = 0; i < p; i++){
			for(int j = C_row_offsets[i]; j < C_row_offsets[i+1]; j++){
				int col = C_col_idx[j];
				float sacc = 0;
				__m256 acc = _mm256_setzero_ps();

				int k = 0;
				for(; k < r-15; k+=16){
					__m256 a0 = _mm256_loadu_ps(A + (i * r + k));
					__m256 a1 = _mm256_loadu_ps(A + (i * r + k + 8));

					__m256 b0 = _mm256_loadu_ps(BT + (col * r + k));
					__m256 b1 = _mm256_loadu_ps(BT + (col * r + k + 8));

					acc = _mm256_fmadd_ps(a0, b0, acc);
					acc = _mm256_fmadd_ps(a1, b1, acc);
                }
                
				//cleanup
				for(; k < r; k++){
					sacc += A[i * r + k] * BT[col * r + k];
				}

				//reduce sum
				const __m128 hiQuad0 = _mm256_extractf128_ps(acc, 1);
				const __m128 loQuad0 = _mm256_castps256_ps128(acc);
				const __m128 sumQuad0 = _mm_add_ps(loQuad0, hiQuad0);
				const __m128 hiDual0 = _mm_movehl_ps(sumQuad0, sumQuad0);
				const __m128 sumDual0 = _mm_add_ps(sumQuad0, hiDual0);
				const __m128 hi0 = _mm_shuffle_ps(sumDual0, sumDual0, 0x1);
				const __m128 sum0 = _mm_add_ss(sumDual0, hi0);

				C_val[j] = sacc + _mm_cvtss_f32(sum0);
			}
		}
	}
}

void sddmm_v3(int p, int q, int r, float* A, float* BT, int C_nnz, int* C_row_offsets, int* C_col_idx, float* C_val) {
    #pragma omp parallel
	{
		#pragma omp for
		for(int i = 0; i < p; i++){
			int k = 0;
			for(; k < r-31; k+=32){
				__m256 a0 = _mm256_loadu_ps(A + (i * r + k));
				__m256 a1 = _mm256_loadu_ps(A + (i * r + k + 8));
				__m256 a2 = _mm256_loadu_ps(A + (i * r + k + 16));
				__m256 a3 = _mm256_loadu_ps(A + (i * r + k + 24));

				for(int j = C_row_offsets[i]; j < C_row_offsets[i+1]; j++){
					int col = C_col_idx[j];

					const __m256 b0 = _mm256_loadu_ps(BT + (col * r + k));
					const __m256 b1 = _mm256_loadu_ps(BT + (col * r + k + 8));
					const __m256 b2 = _mm256_loadu_ps(BT + (col * r + k + 16));
					const __m256 b3 = _mm256_loadu_ps(BT + (col * r + k + 24));

					__m256 acc = _mm256_mul_ps(a0, b0);
					acc = _mm256_fmadd_ps(a1, b1, acc);
					acc = _mm256_fmadd_ps(a2, b2, acc);
					acc = _mm256_fmadd_ps(a3, b3, acc);

					const __m128 hiQuad0 = _mm256_extractf128_ps(acc, 1);
					const __m128 loQuad0 = _mm256_castps256_ps128(acc);
					const __m128 sumQuad0 = _mm_add_ps(loQuad0, hiQuad0);
					const __m128 hiDual0 = _mm_movehl_ps(sumQuad0, sumQuad0);
					const __m128 sumDual0 = _mm_add_ps(sumQuad0, hiDual0);
					const __m128 hi0 = _mm_shuffle_ps(sumDual0, sumDual0, 0x1);
					const __m128 sum0 = _mm_add_ss(sumDual0, hi0);

					C_val[j] += _mm_cvtss_f32(sum0);
				}
			}

			for(; k < r; k++){
				float af = A[i * r + k];

				for(int j = C_row_offsets[i]; j < C_row_offsets[i+1]; j++){
					int col = C_col_idx[j];

					C_val[j] += af * BT[col * r + k];
				}
			}
		}
	}
}



void sddmm_coo(int p, int q, int r, float* A, float* BT, int C_nnz, int* C_row_idx, int* C_col_idx, float* C_val) {
    #pragma omp parallel
	{
		#pragma omp for
		for(int j = 0; j < C_nnz; j++){
			int i = C_row_idx[j];
			int col = C_col_idx[j];
			float sacc = 0;
			__m256 acc = _mm256_setzero_ps();

			int k = 0;
			for(; k < r-7; k+=8){
				__m256 a0 = _mm256_loadu_ps(A + (i * r + k));
				__m256 b0 = _mm256_loadu_ps(BT + (col * r + k));
				acc = _mm256_fmadd_ps(a0, b0, acc); 
			}
			
			//cleanup
			for(; k < r; k++){
				sacc += A[i * r + k] * BT[col * r + k];
			}

			//reduce sum
			const __m128 hiQuad0 = _mm256_extractf128_ps(acc, 1);
			const __m128 loQuad0 = _mm256_castps256_ps128(acc);
			const __m128 sumQuad0 = _mm_add_ps(loQuad0, hiQuad0);
			const __m128 hiDual0 = _mm_movehl_ps(sumQuad0, sumQuad0);
			const __m128 sumDual0 = _mm_add_ps(sumQuad0, hiDual0);
			const __m128 hi0 = _mm_shuffle_ps(sumDual0, sumDual0, 0x1);
			const __m128 sum0 = _mm_add_ss(sumDual0, hi0);

			C_val[j] = sacc + _mm_cvtss_f32(sum0);
		}
	}
}

void sddmm_coo_v2(int p, int q, int r, float* A, float* BT, int C_nnz, int* C_row_idx, int* C_col_idx, float* C_val) {
    #pragma omp parallel
	{
		#pragma omp for
		for(int j = 0; j < C_nnz; j++){
			int row = C_row_idx[j];
			int col = C_col_idx[j];
			float sacc = 0;
			__m256 acc = _mm256_setzero_ps();

			int k = 0;
			for(; k < r-15; k+=16){
				__m256 a0 = _mm256_loadu_ps(A + (row * r + k));
				__m256 a1 = _mm256_loadu_ps(A + (row * r + k + 8));

				__m256 b0 = _mm256_loadu_ps(BT + (col * r + k));
				__m256 b1 = _mm256_loadu_ps(BT + (col * r + k + 8));

				acc = _mm256_fmadd_ps(a0, b0, acc); 
				acc = _mm256_fmadd_ps(a1, b1, acc); 
			}
			
			//cleanup
			for(; k < r; k++){
				sacc += A[row * r + k] * BT[col * r + k];
			}

			//reduce sum
			const __m128 hiQuad0 = _mm256_extractf128_ps(acc, 1);
			const __m128 loQuad0 = _mm256_castps256_ps128(acc);
			const __m128 sumQuad0 = _mm_add_ps(loQuad0, hiQuad0);
			const __m128 hiDual0 = _mm_movehl_ps(sumQuad0, sumQuad0);
			const __m128 sumDual0 = _mm_add_ps(sumQuad0, hiDual0);
			const __m128 hi0 = _mm_shuffle_ps(sumDual0, sumDual0, 0x1);
			const __m128 sum0 = _mm_add_ss(sumDual0, hi0);

			C_val[j] = sacc + _mm_cvtss_f32(sum0);
		}
	}
}

// A: b x p x r (dense)
// BT: b x q x r (dense)
// C: b x p x q (same_size_bcsr)
// C_row_offsets: b x (p + 1)
// C_val, C_col_idx: b x C_nnz
void same_nnz_bsddmm(int b, int p, int q, int r, float* A, float* BT, int C_nnz, int* C_row_offsets,
					 int* C_col_idx, float* C_val) {
    #pragma omp parallel
	{
		#pragma omp for collapse(2)
		for (int z = 0; z < b; z++){
			for(int i = 0; i < p; i++){
				for(int j = C_row_offsets[z * (p + 1) + i]; j < C_row_offsets[z * (p + 1) + i + 1]; j++){
					int col = C_col_idx[z * C_nnz + j];
					float sacc = 0;
					__m256 acc = _mm256_setzero_ps();

					int k = 0;
					for(; k < r-7; k+=8){
						__m256 a0 = _mm256_loadu_ps(A + (z * p * r + i * r + k));
						__m256 b0 = _mm256_loadu_ps(BT + (z * q * r + col * r + k));
						acc = _mm256_fmadd_ps(a0, b0, acc); 
					}
					
					//cleanup
					for(; k < r; k++){
						sacc += A[z * p * r + i * r + k] * BT[z * q * r + col * r + k];
					}

					//reduce sum
					const __m128 hiQuad0 = _mm256_extractf128_ps(acc, 1);
					const __m128 loQuad0 = _mm256_castps256_ps128(acc);
					const __m128 sumQuad0 = _mm_add_ps(loQuad0, hiQuad0);
					const __m128 hiDual0 = _mm_movehl_ps(sumQuad0, sumQuad0);
					const __m128 sumDual0 = _mm_add_ps(sumQuad0, hiDual0);
					const __m128 hi0 = _mm_shuffle_ps(sumDual0, sumDual0, 0x1);
					const __m128 sum0 = _mm_add_ss(sumDual0, hi0);

					C_val[z * C_nnz + j] = sacc + _mm_cvtss_f32(sum0);
				}
			}
		}
	}
}

// A: b x p x r (dense)
// BT: b x q x r (dense)
// C: b x p x q (same_size_bcsr)
// C_row_offsets: b x (p + 1)
// C_val, C_col_idx: b x C_nnz
void same_nnz_bsddmm_coo(int b, int p, int q, int r, float* A, float* BT, int C_nnz, int* C_row_idx,
					 int* C_col_idx, float* C_val) {
    #pragma omp parallel
	{
		#pragma omp for collapse(2)
		for (int z = 0; z < b; z++){
			for (int j = 0; j < C_nnz; j++){
				int i = C_row_idx[z * C_nnz + j];
				int col = C_col_idx[z * C_nnz + j];
				float sacc = 0;
				__m256 acc = _mm256_setzero_ps();

				int k = 0;
				for(; k < r-7; k+=8){
					__m256 a0 = _mm256_loadu_ps(A + (z * p * r + i * r + k));
					__m256 b0 = _mm256_loadu_ps(BT + (z * q * r + col * r + k));
					acc = _mm256_fmadd_ps(a0, b0, acc); 
				}
				
				//cleanup
				for(; k < r; k++){
					sacc += A[z * p * r + i * r + k] * BT[z * q * r + col * r + k];
				}

				//reduce sum
				const __m128 hiQuad0 = _mm256_extractf128_ps(acc, 1);
				const __m128 loQuad0 = _mm256_castps256_ps128(acc);
				const __m128 sumQuad0 = _mm_add_ps(loQuad0, hiQuad0);
				const __m128 hiDual0 = _mm_movehl_ps(sumQuad0, sumQuad0);
				const __m128 sumDual0 = _mm_add_ps(sumQuad0, hiDual0);
				const __m128 hi0 = _mm_shuffle_ps(sumDual0, sumDual0, 0x1);
				const __m128 sum0 = _mm_add_ss(sumDual0, hi0);

				C_val[z * C_nnz + j] = sacc + _mm_cvtss_f32(sum0);
			}
		}
	}
}

// ====================================== Wrappers ===========================================

void sddmm_wrapper(py::array_t<float> A, py::array_t<float> BT, py::array_t<int> C_row_offsets, py::array_t<int> C_col_idx,
                                               py::array_t<float> C_val) {
    int p = A.shape()[0];
	int q = BT.shape()[0];
	int r = A.shape()[1];
    int C_nnz = C_val.shape()[0];

    auto buf_A = A.request();
    auto buf_BT = BT.request();
	auto buf_C_row_offsets = C_row_offsets.request();
	auto buf_C_col_idx = C_col_idx.request();
	auto buf_C_val = C_val.request();

    float* ptr_A = (float*) buf_A.ptr;
    float* ptr_BT = (float*) buf_BT.ptr;
	int* ptr_C_row_offsets = (int*) buf_C_row_offsets.ptr;
	int* ptr_C_col_idx = (int*) buf_C_col_idx.ptr;
	float* ptr_C_val = (float*) buf_C_val.ptr;
    
    sddmm(p, q, r, ptr_A, ptr_BT, C_nnz, ptr_C_row_offsets, ptr_C_col_idx, ptr_C_val);
}

void sddmm_v2_wrapper(py::array_t<float> A, py::array_t<float> BT, py::array_t<int> C_row_offsets, py::array_t<int> C_col_idx,
                                               py::array_t<float> C_val) {
    int p = A.shape()[0];
	int q = BT.shape()[0];
	int r = A.shape()[1];
    int C_nnz = C_val.shape()[0];

    auto buf_A = A.request();
    auto buf_BT = BT.request();
	auto buf_C_row_offsets = C_row_offsets.request();
	auto buf_C_col_idx = C_col_idx.request();
	auto buf_C_val = C_val.request();

    float* ptr_A = (float*) buf_A.ptr;
    float* ptr_BT = (float*) buf_BT.ptr;
	int* ptr_C_row_offsets = (int*) buf_C_row_offsets.ptr;
	int* ptr_C_col_idx = (int*) buf_C_col_idx.ptr;
	float* ptr_C_val = (float*) buf_C_val.ptr;
    
    sddmm_v2(p, q, r, ptr_A, ptr_BT, C_nnz, ptr_C_row_offsets, ptr_C_col_idx, ptr_C_val);
}

void sddmm_v3_wrapper(py::array_t<float> A, py::array_t<float> BT, py::array_t<int> C_row_offsets, py::array_t<int> C_col_idx,
                                               py::array_t<float> C_val) {
    int p = A.shape()[0];
	int q = BT.shape()[0];
	int r = A.shape()[1];
    int C_nnz = C_val.shape()[0];

    auto buf_A = A.request();
    auto buf_BT = BT.request();
	auto buf_C_row_offsets = C_row_offsets.request();
	auto buf_C_col_idx = C_col_idx.request();
	auto buf_C_val = C_val.request();

    float* ptr_A = (float*) buf_A.ptr;
    float* ptr_BT = (float*) buf_BT.ptr;
	int* ptr_C_row_offsets = (int*) buf_C_row_offsets.ptr;
	int* ptr_C_col_idx = (int*) buf_C_col_idx.ptr;
	float* ptr_C_val = (float*) buf_C_val.ptr;
    
    sddmm_v3(p, q, r, ptr_A, ptr_BT, C_nnz, ptr_C_row_offsets, ptr_C_col_idx, ptr_C_val);
}

void sddmm_coo_wrapper(py::array_t<float> A, py::array_t<float> BT, py::array_t<int> C_row_idx,
					   py::array_t<int> C_col_idx, py::array_t<float> C_val) {
    int p = A.shape()[0];
	int q = BT.shape()[0];
	int r = A.shape()[1];
    int C_nnz = C_val.shape()[0];

    auto buf_A = A.request();
    auto buf_BT = BT.request();
	auto buf_C_row_idx = C_row_idx.request();
	auto buf_C_col_idx = C_col_idx.request();
	auto buf_C_val = C_val.request();

    float* ptr_A = (float*) buf_A.ptr;
    float* ptr_BT = (float*) buf_BT.ptr;
	int* ptr_C_row_idx = (int*) buf_C_row_idx.ptr;
	int* ptr_C_col_idx = (int*) buf_C_col_idx.ptr;
	float* ptr_C_val = (float*) buf_C_val.ptr;
    
    sddmm_coo(p, q, r, ptr_A, ptr_BT, C_nnz, ptr_C_row_idx, ptr_C_col_idx, ptr_C_val);
}

void sddmm_coo_v2_wrapper(py::array_t<float> A, py::array_t<float> BT, py::array_t<int> C_row_idx,
					   py::array_t<int> C_col_idx, py::array_t<float> C_val) {
    int p = A.shape()[0];
	int q = BT.shape()[0];
	int r = A.shape()[1];
    int C_nnz = C_val.shape()[0];

    auto buf_A = A.request();
    auto buf_BT = BT.request();
	auto buf_C_row_idx = C_row_idx.request();
	auto buf_C_col_idx = C_col_idx.request();
	auto buf_C_val = C_val.request();

    float* ptr_A = (float*) buf_A.ptr;
    float* ptr_BT = (float*) buf_BT.ptr;
	int* ptr_C_row_idx = (int*) buf_C_row_idx.ptr;
	int* ptr_C_col_idx = (int*) buf_C_col_idx.ptr;
	float* ptr_C_val = (float*) buf_C_val.ptr;
    
    sddmm_coo_v2(p, q, r, ptr_A, ptr_BT, C_nnz, ptr_C_row_idx, ptr_C_col_idx, ptr_C_val);
}

void same_nnz_bsddmm_wrapper(py::array_t<float> A, py::array_t<float> BT, py::array_t<int> C_row_offsets,
							 py::array_t<int> C_col_idx, py::array_t<float> C_val) {
	// A: b x p x r (dense)
    // BT: b x q x r (dense)
    // C: b x p x q (same_size_bcsr)

	int b = A.shape()[0];		
    int p = A.shape()[1];
	int r = A.shape()[2];
	int q = BT.shape()[1];
    int C_nnz = C_val.shape()[1];

    auto buf_A = A.request();
    auto buf_BT = BT.request();
	auto buf_C_row_offsets = C_row_offsets.request();
	auto buf_C_col_idx = C_col_idx.request();
	auto buf_C_val = C_val.request();

    float* ptr_A = (float*) buf_A.ptr;
    float* ptr_BT = (float*) buf_BT.ptr;
	int* ptr_C_row_offsets = (int*) buf_C_row_offsets.ptr;
	int* ptr_C_col_idx = (int*) buf_C_col_idx.ptr;
	float* ptr_C_val = (float*) buf_C_val.ptr;
    
    same_nnz_bsddmm(b, p, q, r, ptr_A, ptr_BT, C_nnz, ptr_C_row_offsets, ptr_C_col_idx, ptr_C_val);
}

void same_nnz_bsddmm_coo_wrapper(py::array_t<float> A, py::array_t<float> BT, py::array_t<int> C_row_idx,
							 py::array_t<int> C_col_idx, py::array_t<float> C_val) {
	// A: b x p x r (dense)
    // BT: b x q x r (dense)
    // C: b x p x q (same_size_bcoo)

	int b = A.shape()[0];		
    int p = A.shape()[1];
	int r = A.shape()[2];
	int q = BT.shape()[1];
    int C_nnz = C_val.shape()[1];

    auto buf_A = A.request();
    auto buf_BT = BT.request();
	auto buf_C_row_idx = C_row_idx.request();
	auto buf_C_col_idx = C_col_idx.request();
	auto buf_C_val = C_val.request();

    float* ptr_A = (float*) buf_A.ptr;
    float* ptr_BT = (float*) buf_BT.ptr;
	int* ptr_C_row_idx = (int*) buf_C_row_idx.ptr;
	int* ptr_C_col_idx = (int*) buf_C_col_idx.ptr;
	float* ptr_C_val = (float*) buf_C_val.ptr;
    
    same_nnz_bsddmm_coo(b, p, q, r, ptr_A, ptr_BT, C_nnz, ptr_C_row_idx, ptr_C_col_idx, ptr_C_val);
}

