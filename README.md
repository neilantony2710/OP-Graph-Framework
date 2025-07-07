# OP-Graph-Framework
An O.P. Graph Framework written in C++. Optimized, Parallelized.

A high-performance C++ graph processing library designed to demonstrate advanced programming skills and algorithm implementation. This project showcases modern C++ features, clean architecture, and efficient algorithm design.

## Overview

This framework provides:
- **Flexible graph representations** supporting directed/undirected and weighted graphs
- **Classic graph algorithms** with clean, extensible implementations
- **Modern C++ design** using templates, concepts, and RAII principles
- **Benchmarking utilities** to measure and compare performance
- **Comprehensive test suite** demonstrating correctness and usage

## Features

### Phase 1 (Current Implementation)
- ✅ Core graph data structure with adjacency list representation
- ✅ Basic graph operations (add/remove vertices and edges)
- ✅ Graph traversal algorithms (BFS, DFS)
- ✅ Shortest path algorithms (Dijkstra, Bellman-Ford)
- ✅ Minimum spanning tree (Kruskal's algorithm)
- ✅ Connected components detection
- ✅ Topological sorting for DAGs
- ✅ Graph generators for testing
- ✅ Benchmarking framework

### Upcoming Phases
- Phase 2: CPU parallelization with thread pools and OpenMP
- Phase 3: GPU acceleration using CUDA
- Phase 4: Advanced optimizations and memory-efficient representations
- Phase 5: Machine learning integration for algorithm selection
- Phase 6: Real-world applications and visualization

## Requirements

- C++20 compatible compiler (GCC 10+, Clang 12+, or MSVC 2019+)
- CMake 3.16+ (optional)
- Make (for using the provided Makefile)

## Building

### Using Make
```bash
# Build the test executable
make

# Build with debug symbols
make debug

# Run tests
make test

# Run benchmarks only
make benchmark

# Run social network example
make example

# Clean build artifacts
make clean
```

### Manual Compilation
```bash
g++ -std=c++20 -O3 test_graph_framework.cpp -o test_graph_framework
./test_graph_framework
```

## Usage Example

```cpp
#include "graph_framework.hpp"
using namespace pgof;

// Create a weighted directed graph
Graph<int, double> graph(Graph<int, double>::GraphType::DIRECTED);

// Add vertices
graph.addVertex(1);
graph.addVertex(2);
graph.addVertex(3);

// Add weighted edges
graph.addEdge(1, 2, 5.0);
graph.addEdge(2, 3, 3.0);
graph.addEdge(1, 3, 10.0);

// Run Dijkstra's algorithm
auto result = GraphAlgorithms<int, double>::dijkstra(graph, 1);

// Get shortest path to vertex 3
auto path = result.getPath(3);
std::cout << "Shortest distance: " << result.distances[3] << std::endl;
```

### Real-World Example: Social Network Analysis

The repository includes a complete example (`example_social_network.cpp`) demonstrating:
- Friend recommendation system
- Finding mutual friends
- Calculating degrees of separation
- Identifying friend groups (communities)
- Network statistics and analysis

Run it with: `make example`

## Project Structure

```
.
├── graph_framework.hpp          # Main header-only library
├── test_graph_framework.cpp     # Test suite and benchmarks
├── example_social_network.cpp   # Social network analysis example
├── Makefile                     # Build configuration
└── README.md                    # This file
```

## API Documentation

### Graph Class
```cpp
template<typename VertexId, typename EdgeWeight>
class Graph {
    // Construction
    Graph(GraphType type = GraphType::DIRECTED);
    
    // Vertex operations
    void addVertex(VertexId vertex);
    bool hasVertex(VertexId vertex) const;
    
    // Edge operations
    void addEdge(VertexId from, VertexId to, EdgeWeight weight = 1);
    bool hasEdge(VertexId from, VertexId to) const;
    
    // Graph properties
    size_t numVertices() const;
    size_t numEdges() const;
    size_t degree(VertexId vertex) const;
};
```

### Algorithms
- **BFS/DFS**: Graph traversal algorithms
- **Dijkstra**: Single-source shortest paths (non-negative weights)
- **Bellman-Ford**: Single-source shortest paths (handles negative weights)
- **Kruskal**: Minimum spanning tree for undirected graphs
- **Connected Components**: Find disconnected subgraphs
- **Topological Sort**: Linear ordering of vertices in a DAG

## Performance

Current benchmarks on a modern CPU (single-threaded):

| Algorithm | 1000 vertices | 2000 vertices | Time Complexity |
|-----------|---------------|---------------|-----------------|
| BFS       | ~2 ms         | ~8 ms         | O(V + E)        |
| DFS       | ~2 ms         | ~8 ms         | O(V + E)        |
| Dijkstra  | ~15 ms        | ~65 ms        | O((V + E) log V)|
| Kruskal   | ~5 ms         | ~20 ms        | O(E log E)      |

## Design Decisions

1. **Header-only library**: Simplifies integration and showcases template programming
2. **Modern C++ features**: Demonstrates knowledge of C++20 concepts, std::optional, etc.
3. **Generic programming**: Supports any hashable vertex type and numeric edge weights
4. **Clean interfaces**: Focuses on readability and ease of use
5. **Extensible architecture**: Easy to add new algorithms and representations

### Key Design Patterns Used

- **Template specialization** for type-safe generic programming
- **RAII** for automatic resource management
- **Iterator patterns** for graph traversal
- **Strategy pattern** potential for algorithm selection (Phase 5)
- **Factory pattern** potential for graph representations (Phase 4)

## Future Enhancements

- [ ] Parallel execution using std::execution policies
- [ ] CUDA kernels for GPU acceleration
- [ ] Memory-mapped graphs for out-of-core processing
- [ ] Advanced algorithms (A*, network flow, graph coloring)
- [ ] Python bindings for easier prototyping
- [ ] Visualization tools for debugging

## Extending the Framework

Adding a new algorithm is straightforward:

```cpp
template<typename VertexId, typename EdgeWeight>
static YourResultType yourAlgorithm(const GraphType& graph, VertexId start) {
    // Your implementation here
}
```



