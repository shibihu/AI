CXX ?= g++
CXXFLAGS = -O3 -ffast-math -funroll-loops -std=c++17 -Wall -Wextra -Icpp
TARGET = build/slm
SRC = cpp/main.cpp
HEADERS = cpp/slm_engine.hpp

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -rf build model.slm

.PHONY: all clean
