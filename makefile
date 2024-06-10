# CXX := g++
# CXXFLAGS := -std=c++11 -O3
# NVFLAGS := $(CXXFLAGS) -arch=sm_61 -Xcompiler -fopenmp
# TARGET := tsp
# SEQUENTIAL := nbody


# .PHONY: all
# all: $(TARGET)

# .PHONY: tsp
# tsp: tsp.cu
# 	nvcc $(NVFLAGS) -o tsp tsp.cu
# .PHONY: seq
# seq: nbody.cc
# 	$(CXX) $(CXXFLAGS) -o nbody nbody.cc

# .PHONY: clean
# clean:
# 	rm -f $(TARGET) $(SEQUENTIAL)

# -----------------------------------------------------

# Variables
NVCC = nvcc
CXXFLAGS = -std=c++11 -O3 -arch=sm_61 -Xcompiler -fopenmp
TARGET ?= tsp

# Default target
all: $(TARGET)

# Target rule
$(TARGET): $(TARGET).cu
	$(NVCC) $(CXXFLAGS) -o $@ $<

# Clean rule
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean
