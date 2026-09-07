# ============================================================================
# Makefile — SLM Inference Engine
#
# Targets:
#   make            Build with native compiler (g++ / c++)
#   make cross      Cross-compile for aarch64-linux-android (Termux)
#   make clean      Remove build artifacts
#
# Override CC/CXX for cross-compilation:
#   make CC=aarch64-linux-android-clang CXX=aarch64-linux-android-clang++
# ============================================================================

# --- Native (default) ---
CC       ?= gcc
CXX      ?= g++

# --- Cross-compilation (Termux / Android NDK) ---
CROSS_CC  = aarch64-linux-android-clang
CROSS_CXX = aarch64-linux-android-clang++

# --- Compiler flags ---
COMMON_FLAGS  = -O3 -ffast-math -funroll-loops -std=c++17
WARN_FLAGS    = -Wall -Wextra -Wpedantic
INCLUDE_FLAGS = -I.

# --- Output ---
BUILD_DIR = build
TARGET    = $(BUILD_DIR)/slm

# ============================================================================
# Default target — build with host compiler
# ============================================================================

.PHONY: all clean model

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): cpp/main.cpp cpp/slm_engine.hpp | $(BUILD_DIR)
	$(CXX) $(COMMON_FLAGS) $(WARN_FLAGS) $(INCLUDE_FLAGS) \
		-o $(TARGET) cpp/main.cpp

# ============================================================================
# Cross-compile for aarch64 (Android / Termux)
# ============================================================================

.PHONY: cross

cross: | $(BUILD_DIR)
	$(CROSS_CXX) $(COMMON_FLAGS) $(WARN_FLAGS) $(INCLUDE_FLAGS) \
		-o $(TARGET) cpp/main.cpp

# ============================================================================
# Generate a dummy model.slm
# ============================================================================

model: $(TARGET)
	python3 generate_model.py -o model.slm \
		--n-layers 4 --n-heads 8 --n-kv-heads 2 \
		--d-model 256 --d-ff 1024 --vocab-size 1000

# ============================================================================
# Bench shortcut
# ============================================================================

.PHONY: bench

bench: all model
	./$(TARGET) bench -m model.slm -n 128

# ============================================================================
# Clean
# ============================================================================

clean:
	rm -rf $(BUILD_DIR)
	rm -f model.slm
