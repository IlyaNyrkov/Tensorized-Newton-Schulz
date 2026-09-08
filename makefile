.PHONY: all build run_benchmarks clean

# The directory where CMake will generate build files and binaries
BUILD_DIR = build

all: build run_benchmarks

build:
	@echo "--- Configuring and building the project ---"
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --parallel

run_benchmarks: build
	@echo "--- Running Phase 1 Benchmark ---"
	./$(BUILD_DIR)/bin/bench_01_nested_polynomial

clean:
	@echo "--- Cleaning build directory ---"
	rm -rf $(BUILD_DIR)