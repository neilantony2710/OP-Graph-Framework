 Makefile for Parallel Graph Optimization Framework

CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -march=native
DEBUG_FLAGS = -g -O0 -DDEBUG
INCLUDES = -I.

# Source files
HEADERS = graph_framework.hpp
TEST_SRC = test_graph_framework.cpp
EXAMPLE_SRC = example_social_network.cpp

# Targets
TARGETS = test_graph_framework example_social_network

# Default target
all: $(TARGETS)

# Test executable
test_graph_framework: $(TEST_SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(TEST_SRC) -o $@

# Debug build
debug: CXXFLAGS = $(DEBUG_FLAGS) -std=c++20 -Wall -Wextra
debug: clean $(TARGETS)

# Run tests
test: test_graph_framework
	./test_graph_framework

# Run example
example: example_social_network
	./example_social_network

# Benchmarks only
benchmark: test_graph_framework
	./test_graph_framework | grep -A 1000 "Running Benchmarks"

# Clean
clean:
	rm -f $(TARGETS) *.o
