# Compiler settings
CXX = g++
CXXFLAGS = -O3 -Wall -std=c++17

# RISC-V specific flags for vector extension
RISCV_V_FLAGS = -march=rv64gcv -menable-rvv

# Target names
TARGET_SCALAR = matmul_scalar
TARGET_VECTOR = matmul_vector

# Default target
all: $(TARGET_SCALAR) $(TARGET_VECTOR)

# Rule to build the scalar (non-vector) version
$(TARGET_SCALAR): matrix_mul.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET_SCALAR) matrix_mul.cpp

# Rule to build the RISC-V vector version
$(TARGET_VECTOR): matrix_mul.cpp
	$(CXX) $(CXXFLAGS) $(RISCV_V_FLAGS) -o $(TARGET_VECTOR) matrix_mul.cpp

# Clean up generated files
clean:
	rm -f $(TARGET_SCALAR) $(TARGET_VECTOR)

.PHONY: all clean
