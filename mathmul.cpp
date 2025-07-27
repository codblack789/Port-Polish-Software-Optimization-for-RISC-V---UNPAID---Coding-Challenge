#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <stdexcept>
#include <cmath>
#include <algorithm>

// Type alias for our matrix
using Matrix = std::vector<std::vector<double>>;

// Fills a matrix with random values between -1.0 and 1.0
void init_matrix(Matrix& mat, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            mat[i][j] = dis(gen);
        }
    }
}

// Naive matrix multiplication: C = A * B
void matmul_naive(const Matrix& A, const Matrix& B, Matrix& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// Tiled (cache-aware) matrix multiplication: C = A * B
void matmul_tiled(const Matrix& A, const Matrix& B, Matrix& C, int N, int TILE_SIZE) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0;
        }
    }
    
    for (int ii = 0; ii < N; ii += TILE_SIZE) {
        for (int jj = 0; jj < N; jj += TILE_SIZE) {
            for (int kk = 0; kk < N; kk += TILE_SIZE) {
                // Perform multiplication on the tile
                for (int i = ii; i < std::min(ii + TILE_SIZE, N); ++i) {
                    for (int j = jj; j < std::min(jj + TILE_SIZE, N); ++j) {
                        double sum = C[i][j]; // Load previous sum
                        for (int k = kk; k < std::min(kk + TILE_SIZE, N); ++k) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] = sum; // Store final sum for the tile block
                    }
                }
            }
        }
    }
}


// Verify that the results of two matrices are the same
void verify_results(const Matrix& C1, const Matrix& C2, int N) {
    const double epsilon = 1e-5;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (std::abs(C1[i][j] - C2[i][j]) > epsilon) {
                std::cerr << "Verification FAILED at (" << i << "," << j << "): "
                          << C1[i][j] << " vs " << C2[i][j] << std::endl;
                return;
            }
        }
    }
    std::cout << "Verification PASSED" << std::endl;
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <tile_size>" << std::endl;
        return 1;
    }

    int N = std::stoi(argv[1]);
    int TILE_SIZE = std::stoi(argv[2]);

    if (N <= 0 || TILE_SIZE <= 0 || TILE_SIZE > N) {
        throw std::runtime_error("Invalid matrix or tile size.");
    }

    std::cout << "--- Matrix Multiplication Benchmark ---" << std::endl;
    std::cout << "Matrix Size (N x N): " << N << " x " << N << std::endl;
    std::cout << "Tile Size: " << TILE_SIZE << " x " << TILE_SIZE << std::endl;
#ifdef __riscv_vector
    std::cout << "Vector Instructions: Enabled" << std::endl;
#else
    std::cout << "Vector Instructions: Disabled" << std::endl;
#endif
    std::cout << "---------------------------------------" << std::endl;

    // Allocate matrices
    Matrix A(N, std::vector<double>(N));
    Matrix B(N, std::vector<double>(N));
    Matrix C_naive(N, std::vector<double>(N, 0.0));
    Matrix C_tiled(N, std::vector<double>(N, 0.0));

    // Initialize with random data
    init_matrix(A, N);
    init_matrix(B, N);

    // --- Benchmark Naive Implementation ---
    auto start_naive = std::chrono::high_resolution_clock::now();
    matmul_naive(A, B, C_naive, N);
    auto end_naive = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_naive = end_naive - start_naive;
    double gflops_naive = (2.0 * N * N * N) / duration_naive.count() / 1e9;
    
    std::cout << "Naive Implementation:" << std::endl;
    std::cout << "  Time: " << duration_naive.count() << " seconds" << std::endl;
    std::cout << "  Performance: " << gflops_naive << " GFLOPS" << std::endl;
    std::cout << std::endl;

    // --- Benchmark Tiled Implementation ---
    auto start_tiled = std::chrono::high_resolution_clock::now();
    matmul_tiled(A, B, C_tiled, N, TILE_SIZE);
    auto end_tiled = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_tiled = end_tiled - start_tiled;
    double gflops_tiled = (2.0 * N * N * N) / duration_tiled.count() / 1e9;

    std::cout << "Tiled Implementation:" << std::endl;
    std::cout << "  Time: " << duration_tiled.count() << " seconds" << std::endl;
    std::cout << "  Performance: " << gflops_tiled << " GFLOPS" << std::endl;
    std::cout << "---------------------------------------" << std::endl;

    // --- Verify correctness ---
    verify_results(C_naive, C_tiled, N);

    return 0;
}
