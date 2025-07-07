// test_graph_framework.cpp
#include "graph_framework.hpp"
#include <iostream>
#include <cassert>
#include <iomanip>

using namespace pgof;

// Simple test framework
class TestRunner {
public:
    static void run(const std::string& name, std::function<void()> test) {
        try {
            test();
            std::cout << "[PASS] " << name << std::endl;
            passed_++;
        } catch (const std::exception& e) {
            std::cout << "[FAIL] " << name << ": " << e.what() << std::endl;
            failed_++;
        }
    }
    
    static void summary() {
        std::cout << "\n=== Test Summary ===\n";
        std::cout << "Passed: " << passed_ << "\n";
        std::cout << "Failed: " << failed_ << "\n";
        std::cout << "Total:  " << (passed_ + failed_) << "\n";
    }
    
private:
    static inline int passed_ = 0;
    static inline int failed_ = 0;
};

// Test basic graph operations
void testBasicOperations() {
    Graph<int, double> graph(Graph<int, double>::GraphType::DIRECTED);
    
    // Test vertex operations
    graph.addVertex(1);
    graph.addVertex(2);
    graph.addVertex(3);
    
    assert(graph.numVertices() == 3);
    assert(graph.hasVertex(1));
    assert(graph.hasVertex(2));
    assert(graph.hasVertex(3));
    assert(!graph.hasVertex(4));
    
    // Test edge operations
    graph.addEdge(1, 2, 1.5);
    graph.addEdge(2, 3, 2.0);
    graph.addEdge(1, 3, 3.5);
    
    assert(graph.numEdges() == 3);
    assert(graph.hasEdge(1, 2));
    assert(graph.hasEdge(2, 3));
    assert(!graph.hasEdge(3, 1));
    
    // Test edge weights
    auto weight = graph.getEdgeWeight(1, 2);
    assert(weight.has_value() && *weight == 1.5);
    
    // Test neighbors
    auto neighbors = graph.getNeighbors(1);
    assert(neighbors.size() == 2);
    
    // Test degree
    assert(graph.degree(1) == 2);
    assert(graph.degree(2) == 1);
    assert(graph.degree(3) == 0);
}

// Test undirected graph
void testUndirectedGraph() {
    Graph<int, double> graph(Graph<int, double>::GraphType::UNDIRECTED);
    
    graph.addEdge(1, 2, 1.0);
    graph.addEdge(2, 3, 2.0);
    
    assert(graph.numEdges() == 2);  // Undirected counts each edge once
    assert(graph.hasEdge(1, 2));
    assert(graph.hasEdge(2, 1));    // Reverse edge should exist
    assert(graph.hasEdge(2, 3));
    assert(graph.hasEdge(3, 2));    // Reverse edge should exist
    
    assert(graph.degree(2) == 2);   // Connected to both 1 and 3
}

// Test BFS
void testBFS() {
    Graph<int, double> graph(Graph<int, double>::GraphType::DIRECTED);
    
    // Create a simple graph: 1 -> 2 -> 3
    //                        \-> 4
    graph.addEdge(1, 2);
    graph.addEdge(1, 4);
    graph.addEdge(2, 3);
    
    auto bfsResult = GraphAlgorithms<int, double>::bfs(graph, 1);
    assert(bfsResult.size() == 4);
    assert(bfsResult[0] == 1);  // Start vertex
    
    // Check that 2 and 4 are visited before 3
    auto pos2 = std::find(bfsResult.begin(), bfsResult.end(), 2) - bfsResult.begin();
    auto pos3 = std::find(bfsResult.begin(), bfsResult.end(), 3) - bfsResult.begin();
    auto pos4 = std::find(bfsResult.begin(), bfsResult.end(), 4) - bfsResult.begin();
    
    assert(pos2 < pos3);  // 2 is visited before 3
}

// Test DFS
void testDFS() {
    Graph<int, double> graph(Graph<int, double>::GraphType::DIRECTED);
    
    graph.addEdge(1, 2);
    graph.addEdge(1, 4);
    graph.addEdge(2, 3);
    
    auto dfsResult = GraphAlgorithms<int, double>::dfs(graph, 1);
    assert(dfsResult.size() == 4);
    assert(dfsResult[0] == 1);  // Start vertex
}

// Test Dijkstra's algorithm
void testDijkstra() {
    Graph<int, double> graph(Graph<int, double>::GraphType::DIRECTED);
    
    // Create a weighted graph
    graph.addEdge(1, 2, 7.0);
    graph.addEdge(1, 3, 9.0);
    graph.addEdge(1, 6, 14.0);
    graph.addEdge(2, 3, 10.0);
    graph.addEdge(2, 4, 15.0);
    graph.addEdge(3, 4, 11.0);
    graph.addEdge(3, 6, 2.0);
    graph.addEdge(4, 5, 6.0);
    graph.addEdge(5, 6, 9.0);
    
    auto result = GraphAlgorithms<int, double>::dijkstra(graph, 1);
    
    // Check distances
    assert(result.distances[1] == 0.0);
    assert(result.distances[2] == 7.0);
    assert(result.distances[3] == 9.0);
    assert(result.distances[4] == 20.0);
    assert(result.distances[5] == 26.0);
    assert(result.distances[6] == 11.0);
    
    // Check path reconstruction
    auto pathTo5 = result.getPath(5);
    assert(pathTo5.size() == 4);  // 1 -> 3 -> 4 -> 5
    assert(pathTo5[0] == 1);
    assert(pathTo5[3] == 5);
}

// Test Bellman-Ford with negative weights
void testBellmanFord() {
    Graph<int, double> graph(Graph<int, double>::GraphType::DIRECTED);
    
    graph.addEdge(1, 2, 4.0);
    graph.addEdge(1, 3, 3.0);
    graph.addEdge(2, 3, -2.0);  // Negative edge
    graph.addEdge(3, 4, 1.0);
    
    auto result = GraphAlgorithms<int, double>::bellmanFord(graph, 1);
    
    assert(!result.hasNegativeCycle);
    assert(result.distances[1] == 0.0);
    assert(result.distances[2] == 4.0);
    assert(result.distances[3] == 2.0);   // Should be 2, not 3, due to path through 2
    assert(result.distances[4] == 3.0);
}

// Test Kruskal's MST
void testKruskal() {
    Graph<int, double> graph(Graph<int, double>::GraphType::UNDIRECTED);
    
    // Create a weighted undirected graph
    graph.addEdge(1, 2, 4.0);
    graph.addEdge(1, 3, 8.0);
    graph.addEdge(2, 3, 11.0);
    graph.addEdge(2, 4, 8.0);
    graph.addEdge(3, 5, 7.0);
    graph.addEdge(3, 4, 2.0);
    graph.addEdge(4, 5, 6.0);
    
    auto mst = GraphAlgorithms<int, double>::kruskal(graph);
    
    // MST should have V-1 edges
    assert(mst.size() == graph.numVertices() - 1);
    
    // Calculate total weight
    double totalWeight = 0.0;
    for (const auto& edge : mst) {
        totalWeight += edge.weight;
    }
    
    // For this graph, MST weight should be 19
    assert(totalWeight == 19.0);
}

// Test connected components
void testConnectedComponents() {
    Graph<int, double> graph(Graph<int, double>::GraphType::UNDIRECTED);
    
    // Create two disconnected components
    // Component 1: 1-2-3
    graph.addEdge(1, 2);
    graph.addEdge(2, 3);
    
    // Component 2: 4-5
    graph.addEdge(4, 5);
    
    // Isolated vertex
    graph.addVertex(6);
    
    auto components = GraphAlgorithms<int, double>::connectedComponents(graph);
    
    assert(components.size() == 3);  // Three components
    
    // Find sizes of components
    std::vector<size_t> sizes;
    for (const auto& comp : components) {
        sizes.push_back(comp.size());
    }
    std::sort(sizes.begin(), sizes.end());
    
    assert(sizes[0] == 1);  // Isolated vertex
    assert(sizes[1] == 2);  // Component with 2 vertices
    assert(sizes[2] == 3);  // Component with 3 vertices
}

// Test topological sort
void testTopologicalSort() {
    Graph<int, double> graph(Graph<int, double>::GraphType::DIRECTED);
    
    // Create a DAG
    graph.addEdge(5, 2);
    graph.addEdge(5, 0);
    graph.addEdge(4, 0);
    graph.addEdge(4, 1);
    graph.addEdge(2, 3);
    graph.addEdge(3, 1);
    
    auto topoSort = GraphAlgorithms<int, double>::topologicalSort(graph);
    
    assert(topoSort.has_value());
    assert(topoSort->size() == 6);
    
    // Verify topological ordering
    auto& order = *topoSort;
    std::unordered_map<int, size_t> position;
    for (size_t i = 0; i < order.size(); ++i) {
        position[order[i]] = i;
    }
    
    // Check that all edges go from earlier to later in the ordering
    for (const auto& edge : graph.edges()) {
        assert(position[edge.from] < position[edge.to]);
    }
}

// Benchmark different algorithms
void runBenchmarks() {
    std::cout << "\n=== Running Benchmarks ===\n";
    
    // Test on different graph sizes
    std::vector<size_t> sizes = {100, 500, 1000, 2000};
    
    for (size_t size : sizes) {
        std::cout << "\nGraph size: " << size << " vertices\n";
        
        // Generate random graph
        auto graph = GraphGenerator::generateRandomGraph<int, double>(
            size, 0.1, 1.0, 10.0, Graph<int, double>::GraphType::DIRECTED);
        
        std::cout << "Edges: " << graph.numEdges() << "\n";
        
        // Benchmark BFS
        Benchmark::start("BFS_" + std::to_string(size));
        auto bfsResult = GraphAlgorithms<int, double>::bfs(graph, 0);
        double bfsTime = Benchmark::end("BFS_" + std::to_string(size));
        std::cout << "BFS: " << std::fixed << std::setprecision(2) << bfsTime << " ms\n";
        
        // Benchmark DFS
        Benchmark::start("DFS_" + std::to_string(size));
        auto dfsResult = GraphAlgorithms<int, double>::dfs(graph, 0);
        double dfsTime = Benchmark::end("DFS_" + std::to_string(size));
        std::cout << "DFS: " << dfsTime << " ms\n";
        
        // Benchmark Dijkstra
        Benchmark::start("Dijkstra_" + std::to_string(size));
        auto dijkstraResult = GraphAlgorithms<int, double>::dijkstra(graph, 0);
        double dijkstraTime = Benchmark::end("Dijkstra_" + std::to_string(size));
        std::cout << "Dijkstra: " << dijkstraTime << " ms\n";
        
        // Benchmark Bellman-Ford (only for smaller graphs due to O(VE) complexity)
        if (size <= 500) {
            Benchmark::start("BellmanFord_" + std::to_string(size));
            auto bellmanResult = GraphAlgorithms<int, double>::bellmanFord(graph, 0);
            double bellmanTime = Benchmark::end("BellmanFord_" + std::to_string(size));
            std::cout << "Bellman-Ford: " << bellmanTime << " ms\n";
        }
    }
    
    // Test MST on undirected graphs
    std::cout << "\n=== MST Benchmarks ===\n";
    for (size_t size : {100, 200, 500}) {
        auto graph = GraphGenerator::generateRandomGraph<int, double>(
            size, 0.3, 1.0, 10.0, Graph<int, double>::GraphType::UNDIRECTED);
        
        std::cout << "\nGraph size: " << size << " vertices, " << graph.numEdges() << " edges\n";
        
        Benchmark::start("Kruskal_" + std::to_string(size));
        auto mst = GraphAlgorithms<int, double>::kruskal(graph);
        double kruskalTime = Benchmark::end("Kruskal_" + std::to_string(size));
        std::cout << "Kruskal: " << kruskalTime << " ms (MST edges: " << mst.size() << ")\n";
    }
}

// Example usage demonstrating the framework
void demonstrateUsage() {
    std::cout << "\n=== Framework Usage Example ===\n";
    
    // Create a graph representing a small road network
    Graph<std::string, double> roadNetwork(Graph<std::string, double>::GraphType::UNDIRECTED);
    
    // Add cities (vertices)
    roadNetwork.addVertex("San Francisco");
    roadNetwork.addVertex("Los Angeles");
    roadNetwork.addVertex("Las Vegas");
    roadNetwork.addVertex("Phoenix");
    roadNetwork.addVertex("Denver");
    roadNetwork.addVertex("Seattle");
    
    // Add roads (edges) with distances
    roadNetwork.addEdge("San Francisco", "Los Angeles", 383.0);
    roadNetwork.addEdge("San Francisco", "Seattle", 807.0);
    roadNetwork.addEdge("Los Angeles", "Las Vegas", 270.0);
    roadNetwork.addEdge("Los Angeles", "Phoenix", 372.0);
    roadNetwork.addEdge("Las Vegas", "Phoenix", 297.0);
    roadNetwork.addEdge("Las Vegas", "Denver", 748.0);
    roadNetwork.addEdge("Phoenix", "Denver", 836.0);
    roadNetwork.addEdge("Seattle", "Denver", 1315.0);
    
    std::cout << "\nRoad Network:\n";
    roadNetwork.printGraph();
    
    // Find shortest path from SF to Denver
    std::cout << "\n--- Shortest Paths from San Francisco ---\n";
    auto paths = GraphAlgorithms<std::string, double>::dijkstra(roadNetwork, "San Francisco");
    
    for (const auto& [city, distance] : paths.distances) {
        if (distance != std::numeric_limits<double>::infinity()) {
            std::cout << "To " << city << ": " << distance << " miles\n";
            
            // Show the path
            auto path = paths.getPath(city);
            std::cout << "  Path: ";
            for (size_t i = 0; i < path.size(); ++i) {
                std::cout << path[i];
                if (i < path.size() - 1) std::cout << " -> ";
            }
            std::cout << "\n";
        }
    }
    
    // Find minimum spanning tree (e.g., for laying fiber optic cables)
    std::cout << "\n--- Minimum Spanning Tree (Fiber Network) ---\n";
    auto mst = GraphAlgorithms<std::string, double>::kruskal(roadNetwork);
    
    double totalCost = 0.0;
    for (const auto& edge : mst) {
        std::cout << edge.from << " <-> " << edge.to << ": " << edge.weight << " miles\n";
        totalCost += edge.weight;
    }
    std::cout << "Total distance: " << totalCost << " miles\n";
}

int main() {
    std::cout << "=== Parallel Graph Optimization Framework - Phase 1 Tests ===\n\n";
    
    // Run unit tests
    TestRunner::run("Basic Operations", testBasicOperations);
    TestRunner::run("Undirected Graph", testUndirectedGraph);
    TestRunner::run("BFS", testBFS);
    TestRunner::run("DFS", testDFS);
    TestRunner::run("Dijkstra's Algorithm", testDijkstra);
    TestRunner::run("Bellman-Ford Algorithm", testBellmanFord);
    TestRunner::run("Kruskal's MST", testKruskal);
    TestRunner::run("Connected Components", testConnectedComponents);
    TestRunner::run("Topological Sort", testTopologicalSort);
    
    TestRunner::summary();
    
    // Run benchmarks
    runBenchmarks();
    
    // Show detailed benchmark results
    Benchmark::printResults();
    
    // Demonstrate practical usage
    demonstrateUsage();
    
    return 0;
}
