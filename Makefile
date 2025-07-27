CXX = riscv64-unknown-elf-g++
CXXFLAGS = -O3 -march=rv64g -mabi=lp64
TARGET = matmul

all: no_vector vector

no_vector:
	$(CXX) $(CXXFLAGS) matmul.cpp -o $(TARGET)_no_vector

vector:
	$(CXX) $(CXXFLAGS) -DVECTOR_INSTRUCTIONS matmul.cpp -o $(TARGET)_vector

clean:
	rm -f matmul_* benchmark_results.txt
