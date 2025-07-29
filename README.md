# Port-Polish-Software-Optimization-for-RISC-V---UNPAID---Coding-Challenge

Software Optimization for RISC-V: Matrix Multiplication

This project demonstrates software optimization techniques for matrix multiplication on the RISC-V architecture. It provides implementations for naive, cache-aware (tiled), and RISC-V 'V' extension (vectorized) matrix multiplication, along with performance benchmarks.

The primary goal is to showcase the significant performance improvements gained by moving from a simple algorithm to one optimized for the underlying hardware architecture.


Features

    Naive Implementation: A standard (i, k, j) three-loop matrix multiplication.

    Cache-Aware (Tiled) Implementation: A blocked algorithm that processes matrices in smaller chunks (tiles) to improve data locality and cache usage.

    Vectorized Implementation: A highly optimized version that uses the RISC-V 'V' Vector Extension via compiler intrinsics for massive parallelism.

    Performance Benchmarking: Measures and prints the execution time for all implemented methods.

    Correctness Verification: Ensures that the optimized versions produce the same results as the naive implementation.


Prerequisites

To build and run this project, you will need:

    A C++17 compliant compiler (e.g., g++).

    The make build automation tool.

    A RISC-V GNU Toolchain that supports the 'V' Vector Extension (-march=rv64gcv).


How to Build

A Makefile is included to simplify the compilation process.

    To build both the scalar and vectorized versions:

    ```make all```

    This command will produce two executables:

    matmul_scalar: An optimized binary without vector instructions.

    matmul_vector: An optimized binary that utilizes the RISC-V Vector Extension.


    2. build only the scalar version
      ```make scalar```

    3.To build only the vector version
      ```make vector```

    4. To clean up generated files
    ```make clean```
How to Run

Run the executables from your terminal, providing the square matrix size as a command-line argument
```./<executable_name> <matrix_size>```
