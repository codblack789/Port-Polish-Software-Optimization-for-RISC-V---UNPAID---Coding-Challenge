#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <stdexcept>
#include <cmath>
#include <algorithm> // For std::min

// Check for RISC-V Vector Extension
#if defined(__riscv_v_ext)
#include <riscv_vector.h>
#endif

// Use a single contiguous vector for matrix storage for better cache performance
using Matrix = std::vector<float>;

// --- Helper Functions ---

// Initializes a matrix with random values
void init_matrix(Matrix& mat, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < N * N; ++i) {
        mat[i] = static_cast<float>(dis(gen));
    }
}

// Verifies if two matrices are approximately equal
void verify_results(const Matrix& C1, const Matrix& C2, int N) {
    const float epsilon = 1e-4f;
    for (int i = 0; i < N * N; ++i) {
        if (std::abs(C1[i] - C2[i]) > epsilon) {
            std::cerr << "Verification FAILED at index " << i << "! (" << C1[i] << " vs " << C2[i] << ")" << std::endl;
            return;
        }
    }
    std::cout << "Verification PASSED." << std::endl;
}

// --- Matrix Multiplication Implementations ---

// 1. Naive implementation (ikj loop order)
void matmul_naive(const Matrix& A, const Matrix& B, Matrix& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

// 2. Cache-aware (Blocked/Tiled) implementation
void matmul_blocked(const Matrix& A, const Matrix& B, Matrix& C, int N, int BLOCK_SIZE) {
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                for (int i = ii; i < std::min(ii + BLOCK_SIZE, N); ++i) {
                    for (int k = kk; k < std::min(kk + BLOCK_SIZE, N); ++k) {
                        for (int j = jj; j < std::min(jj + BLOCK_SIZE, N); ++j) {
                            C[i * N + j] += A[i * N + k] * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// 3. Blocked implementation with RISC-V Vector Intrinsics
#if defined(__riscv_v_ext)
void matmul_vectorized(const Matrix& A, const Matrix& B, Matrix& C, int N, int BLOCK_SIZE) {
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                for (int i = ii; i < std::min(ii + BLOCK_SIZE, N); ++i) {
                    for (int k = kk; k < std::min(kk + BLOCK_SIZE, N); ++k) {
                        // Vectorize the innermost loop (j-loop)
                        size_t j = jj;
                        size_t vl; // Vector Length
                        float a_ik = A[i * N + k];
                        for (; j < std::min(jj + BLOCK_SIZE, N); j += vl) {
                            vl = vsetvl_e32m4(std::min(jj + BLOCK_SIZE, N) - j);

                            // Load a vector from C
                            vfloat32m4_t c_vec = vle32_v_f32m4(&C[i * N + j], vl);
                            // Load a vector from B
                            vfloat32m4_t b_vec = vle32_v_f32m4(&B[k * N + j], vl);
                            
                            // Perform fused multiply-add: C[i,j] += A[i,k] * B[k,j]
                            c_vec = vfmacc_vf_f32m4(c_vec, a_ik, b_vec, vl);
                            
                            // Store the result vector back to C
                            vse32_v_f32m4(&C[i * N + j], c_vec, vl);
                        }
                    }
                }
            }
        }
    }
}
#endif

// --- Main Program ---

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }
    int N = std::atoi(argv[1]);
    if (N <= 0) {
        std::cerr << "Matrix size must be positive." << std::endl;
        return 1;
    }

    const int BLOCK_SIZE = 32;
    std::cout << "Matrix Multiplication for N = " << N << ", BLOCK_SIZE = " << BLOCK_SIZE << std::endl;

    // Allocate matrices
    Matrix A(N * N);
    Matrix B(N * N);
    Matrix C_naive(N * N, 0.0f);
    Matrix C_blocked(N * N, 0.0f);
    
    // Initialize with random data
    init_matrix(A, N);
    init_matrix(B, N);
    
    // --- Performance Measurement ---
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed;

    // 1. Naive
    start = std::chrono::high_resolution_clock::now();
    matmul_naive(A, B, C_naive, N);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "1. Naive time:          " << elapsed.count() << " seconds" << std::endl;

    // 2. Blocked
    start = std::chrono::high_resolution_clock::now();
    matmul_blocked(A, B, C_blocked, N, BLOCK_SIZE);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "2. Blocked time:        " << elapsed.count() << " seconds" << std::endl;
    verify_results(C_naive, C_blocked, N);
    
#if defined(__riscv_v_ext)
    // 3. Vectorized
    Matrix C_vectorized(N * N, 0.0f);
    start = std::chrono::high_resolution_clock::now();
    matmul_vectorized(A, B, C_vectorized, N, BLOCK_SIZE);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "3. Vectorized time:     " << elapsed.count() << " seconds" << std::endl;
    verify_results(C_naive, C_vectorized, N);
    std::cout << "\nCompiled with RISC-V Vector Instructions." << std::endl;
#else
    std::cout << "\nCompiled without RISC-V Vector Instructions." << std::endl;
#endif

    return 0;
}
