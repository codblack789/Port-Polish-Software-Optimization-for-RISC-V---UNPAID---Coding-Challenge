#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

#define BLOCK_SIZE 32  //for now 

using Matrix = vector<vector<float>>;

void init_matrix(Matrix &mat, int N) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            mat[i][j] = static_cast<float>(dis(gen));
}

void matmul_naive(const Matrix &A, const Matrix &B, Matrix &C, int N) {
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k)
            for (int j = 0; j < N; ++j)
                C[i][j] += A[i][k] * B[k][j];
}

void matmul_blocked(const Matrix &A, const Matrix &B, Matrix &C, int N) {
    for (int ii = 0; ii < N; ii += BLOCK_SIZE)
        for (int jj = 0; jj < N; jj += BLOCK_SIZE)
            for (int kk = 0; kk < N; kk += BLOCK_SIZE)
                for (int i = ii; i < min(ii + BLOCK_SIZE, N); ++i)
                    for (int k = kk; k < min(kk + BLOCK_SIZE, N); ++k)
                        for (int j = jj; j < min(jj + BLOCK_SIZE, N); ++j)
                            C[i][j] += A[i][k] * B[k][j];
}

int main(int argc, char *argv[]) {
    int N = 1024;
    bool use_block = false;

    if (argc >= 2)
        N = atoi(argv[1]);
    if (argc >= 3 && strcmp(argv[2], "block") == 0)
        use_block = true;

    Matrix A(N, vector<float>(N));
    Matrix B(N, vector<float>(N));
    Matrix C(N, vector<float>(N, 0.0f));

    init_matrix(A, N);
    init_matrix(B, N);

    auto start = high_resolution_clock::now();
    if (use_block)
        matmul_blocked(A, B, C, N);
    else
        matmul_naive(A, B, C, N);
    auto end = high_resolution_clock::now();

    double time_sec = duration_cast<duration<double>>(end - start).count();
    cout << (use_block ? "Blocked" : "Naive") << " matrix multiplication time: " << time_sec << " seconds\n";

#ifdef VECTOR_INSTRUCTIONS
    cout << "Compiled with vector instructions.\n";
#else
    cout << "Compiled without vector instructions.\n";
#endif

    return 0;
}
