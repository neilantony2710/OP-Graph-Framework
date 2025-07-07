// graph_framework.hpp
#ifndef GRAPH_FRAMEWORK_HPP
#define GRAPH_FRAMEWORK_HPP

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <stack>
#include <limits>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <memory>
#include <optional>
#include <functional>
#include <concepts>
#include <type_traits>

namespace pgof { // Parallel Graph Optimization Framework

// Forward declarations
template<typename VertexId, typename EdgeWeight> class Graph;
template<typename VertexId, typename EdgeWeight> class GraphAlgorithms;

// Concepts for template constraints (C++20)
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<typename T>
concept Hashable = requires(T t) {
    { std::hash<T>{}(t) } -> std::convertible_to<std::size_t>;
};

// Edge structure
template<typename VertexId, typename Weight = double>
struct Edge {
    VertexId from;
    VertexId to;
    Weight weight;
    
    Edge(VertexId f, VertexId t, Weight w = Weight{1}) 
        : from(f), to(t), weight(w) {}
    
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

// Vertex properties that can be extended
template<typename VertexId>
struct VertexProperties {
    VertexId id;
    std::string label;
    
    VertexProperties(VertexId id, const std::string& label = "") 
        : id(id), label(label) {}
};

// Graph representation with adjacency list
template<typename VertexId = int, typename EdgeWeight = double>
    requires Hashable<VertexId> && Numeric<EdgeWeight>
class Graph {
public:
    using EdgeType = Edge<VertexId, EdgeWeight>;
    using AdjacencyList = std::unordered_map<VertexId, std::vector<std::pair<VertexId, EdgeWeight>>>;
    using VertexSet = std::unordered_set<VertexId>;
    
    enum class GraphType { DIRECTED, UNDIRECTED };
    
    // Constructors
    explicit Graph(GraphType type = GraphType::DIRECTED) : type_(type) {}
    
    // Vertex operations
    void addVertex(VertexId vertex) {
        if (adjacencyList_.find(vertex) == adjacencyList_.end()) {
            adjacencyList_[vertex] = std::vector<std::pair<VertexId, EdgeWeight>>();
            vertices_.insert(vertex);
        }
    }
    
    bool hasVertex(VertexId vertex) const {
        return vertices_.find(vertex) != vertices_.end();
    }
    
    // Edge operations
    void addEdge(VertexId from, VertexId to, EdgeWeight weight = EdgeWeight{1}) {
        addVertex(from);
        addVertex(to);
        
        adjacencyList_[from].emplace_back(to, weight);
        edges_.emplace_back(from, to, weight);
        
        if (type_ == GraphType::UNDIRECTED) {
            adjacencyList_[to].emplace_back(from, weight);
            edges_.emplace_back(to, from, weight);
        }
    }
    
    bool hasEdge(VertexId from, VertexId to) const {
        auto it = adjacencyList_.find(from);
        if (it == adjacencyList_.end()) return false;
        
        const auto& neighbors = it->second;
        return std::any_of(neighbors.begin(), neighbors.end(),
            [&to](const auto& pair) { return pair.first == to; });
    }
    
    std::optional<EdgeWeight> getEdgeWeight(VertexId from, VertexId to) const {
        auto it = adjacencyList_.find(from);
        if (it == adjacencyList_.end()) return std::nullopt;
        
        const auto& neighbors = it->second;
        auto edge_it = std::find_if(neighbors.begin(), neighbors.end(),
            [&to](const auto& pair) { return pair.first == to; });
            
        if (edge_it != neighbors.end()) {
            return edge_it->second;
        }
        return std::nullopt;
    }
    
    // Graph properties
    size_t numVertices() const { return vertices_.size(); }
    size_t numEdges() const { 
        return type_ == GraphType::DIRECTED ? edges_.size() : edges_.size() / 2;
    }
    
    GraphType getType() const { return type_; }
    
    // Accessors
    const VertexSet& vertices() const { return vertices_; }
    const std::vector<EdgeType>& edges() const { return edges_; }
    const AdjacencyList& adjacencyList() const { return adjacencyList_; }
    
    // Neighbor access
    std::vector<std::pair<VertexId, EdgeWeight>> getNeighbors(VertexId vertex) const {
        auto it = adjacencyList_.find(vertex);
        if (it != adjacencyList_.end()) {
            return it->second;
        }
        return {};
    }
    
    // Degree functions
    size_t degree(VertexId vertex) const {
        auto it = adjacencyList_.find(vertex);
        return (it != adjacencyList_.end()) ? it->second.size() : 0;
    }
    
    size_t inDegree(VertexId vertex) const {
        if (type_ == GraphType::UNDIRECTED) return degree(vertex);
        
        size_t count = 0;
        for (const auto& [v, neighbors] : adjacencyList_) {
            count += std::count_if(neighbors.begin(), neighbors.end(),
                [&vertex](const auto& pair) { return pair.first == vertex; });
        }
        return count;
    }
    
    // Utility functions
    void clear() {
        adjacencyList_.clear();
        vertices_.clear();
        edges_.clear();
    }
    
    void printGraph() const {
        std::cout << "Graph (" << (type_ == GraphType::DIRECTED ? "Directed" : "Undirected") 
                  << ") with " << numVertices() << " vertices and " << numEdges() << " edges:\n";
        
        for (const auto& [vertex, neighbors] : adjacencyList_) {
            std::cout << vertex << " -> ";
            for (const auto& [neighbor, weight] : neighbors) {
                std::cout << "(" << neighbor << ", " << weight << ") ";
            }
            std::cout << "\n";
        }
    }
    
private:
    GraphType type_;
    AdjacencyList adjacencyList_;
    VertexSet vertices_;
    std::vector<EdgeType> edges_;
};

// Path result structure
template<typename VertexId, typename Distance>
struct PathResult {
    std::unordered_map<VertexId, Distance> distances;
    std::unordered_map<VertexId, VertexId> predecessors;
    bool hasNegativeCycle = false;
    
    std::vector<VertexId> getPath(VertexId destination) const {
        std::vector<VertexId> path;
        if (predecessors.find(destination) == predecessors.end()) {
            return path; // No path exists
        }
        
        VertexId current = destination;
        while (predecessors.find(current) != predecessors.end()) {
            path.push_back(current);
            auto it = predecessors.find(current);
            if (it->second == current) break; // Source vertex
            current = it->second;
        }
        path.push_back(current);
        std::reverse(path.begin(), path.end());
        return path;
    }
};

// Basic graph algorithms (sequential implementation for Phase 1)
template<typename VertexId = int, typename EdgeWeight = double>
class GraphAlgorithms {
public:
    using GraphType = Graph<VertexId, EdgeWeight>;
    using PathResultType = PathResult<VertexId, EdgeWeight>;
    
    // Breadth-First Search
    static std::vector<VertexId> bfs(const GraphType& graph, VertexId start) {
        std::vector<VertexId> visited_order;
        std::unordered_set<VertexId> visited;
        std::queue<VertexId> queue;
        
        if (!graph.hasVertex(start)) return visited_order;
        
        queue.push(start);
        visited.insert(start);
        
        while (!queue.empty()) {
            VertexId current = queue.front();
            queue.pop();
            visited_order.push_back(current);
            
            for (const auto& [neighbor, weight] : graph.getNeighbors(current)) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    queue.push(neighbor);
                }
            }
        }
        
        return visited_order;
    }
    
    // Depth-First Search
    static std::vector<VertexId> dfs(const GraphType& graph, VertexId start) {
        std::vector<VertexId> visited_order;
        std::unordered_set<VertexId> visited;
        std::stack<VertexId> stack;
        
        if (!graph.hasVertex(start)) return visited_order;
        
        stack.push(start);
        
        while (!stack.empty()) {
            VertexId current = stack.top();
            stack.pop();
            
            if (visited.find(current) != visited.end()) continue;
            
            visited.insert(current);
            visited_order.push_back(current);
            
            for (const auto& [neighbor, weight] : graph.getNeighbors(current)) {
                if (visited.find(neighbor) == visited.end()) {
                    stack.push(neighbor);
                }
            }
        }
        
        return visited_order;
    }
    
    // Dijkstra's Algorithm
    static PathResultType dijkstra(const GraphType& graph, VertexId source) {
        PathResultType result;
        
        // Initialize distances
        for (const auto& vertex : graph.vertices()) {
            result.distances[vertex] = std::numeric_limits<EdgeWeight>::infinity();
        }
        result.distances[source] = EdgeWeight{0};
        result.predecessors[source] = source;
        
        // Priority queue: (distance, vertex)
        using PQElement = std::pair<EdgeWeight, VertexId>;
        std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> pq;
        
        pq.emplace(EdgeWeight{0}, source);
        std::unordered_set<VertexId> processed;
        
        while (!pq.empty()) {
            auto [dist, current] = pq.top();
            pq.pop();
            
            if (processed.find(current) != processed.end()) continue;
            processed.insert(current);
            
            for (const auto& [neighbor, weight] : graph.getNeighbors(current)) {
                EdgeWeight newDist = result.distances[current] + weight;
                
                if (newDist < result.distances[neighbor]) {
                    result.distances[neighbor] = newDist;
                    result.predecessors[neighbor] = current;
                    pq.emplace(newDist, neighbor);
                }
            }
        }
        
        return result;
    }
    
    // Bellman-Ford Algorithm
    static PathResultType bellmanFord(const GraphType& graph, VertexId source) {
        PathResultType result;
        
        // Initialize distances
        for (const auto& vertex : graph.vertices()) {
            result.distances[vertex] = std::numeric_limits<EdgeWeight>::infinity();
        }
        result.distances[source] = EdgeWeight{0};
        result.predecessors[source] = source;
        
        // Relax edges |V| - 1 times
        for (size_t i = 0; i < graph.numVertices() - 1; ++i) {
            bool updated = false;
            
            for (const auto& edge : graph.edges()) {
                if (result.distances[edge.from] != std::numeric_limits<EdgeWeight>::infinity()) {
                    EdgeWeight newDist = result.distances[edge.from] + edge.weight;
                    
                    if (newDist < result.distances[edge.to]) {
                        result.distances[edge.to] = newDist;
                        result.predecessors[edge.to] = edge.from;
                        updated = true;
                    }
                }
            }
            
            if (!updated) break; // Early termination
        }
        
        // Check for negative cycles
        for (const auto& edge : graph.edges()) {
            if (result.distances[edge.from] != std::numeric_limits<EdgeWeight>::infinity()) {
                if (result.distances[edge.from] + edge.weight < result.distances[edge.to]) {
                    result.hasNegativeCycle = true;
                    break;
                }
            }
        }
        
        return result;
    }
    
    // Kruskal's Algorithm for Minimum Spanning Tree
    static std::vector<typename GraphType::EdgeType> kruskal(const GraphType& graph) {
        std::vector<typename GraphType::EdgeType> mst;
        
        if (graph.getType() == GraphType::GraphType::DIRECTED) {
            throw std::runtime_error("Kruskal's algorithm requires an undirected graph");
        }
        
        // Get all edges and sort by weight
        auto edges = graph.edges();
        std::sort(edges.begin(), edges.end());
        
        // Union-Find data structure
        std::unordered_map<VertexId, VertexId> parent;
        std::unordered_map<VertexId, int> rank;
        
        // Initialize Union-Find
        for (const auto& vertex : graph.vertices()) {
            parent[vertex] = vertex;
            rank[vertex] = 0;
        }
        
        // Find with path compression
        std::function<VertexId(VertexId)> find = [&](VertexId x) -> VertexId {
            if (parent[x] != x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        };
        
        // Union by rank
        auto unite = [&](VertexId x, VertexId y) -> bool {
            VertexId px = find(x);
            VertexId py = find(y);
            
            if (px == py) return false;
            
            if (rank[px] < rank[py]) {
                parent[px] = py;
            } else if (rank[px] > rank[py]) {
                parent[py] = px;
            } else {
                parent[py] = px;
                rank[px]++;
            }
            return true;
        };
        
        // Process edges in order of weight
        for (const auto& edge : edges) {
            if (unite(edge.from, edge.to)) {
                mst.push_back(edge);
                if (mst.size() == graph.numVertices() - 1) {
                    break;
                }
            }
        }
        
        return mst;
    }
    
    // Connected Components (for undirected graphs)
    static std::vector<std::vector<VertexId>> connectedComponents(const GraphType& graph) {
        std::vector<std::vector<VertexId>> components;
        std::unordered_set<VertexId> visited;
        
        for (const auto& vertex : graph.vertices()) {
            if (visited.find(vertex) == visited.end()) {
                auto component = bfs(graph, vertex);
                for (const auto& v : component) {
                    visited.insert(v);
                }
                components.push_back(std::move(component));
            }
        }
        
        return components;
    }
    
    // Topological Sort (for DAGs)
    static std::optional<std::vector<VertexId>> topologicalSort(const GraphType& graph) {
        if (graph.getType() != GraphType::GraphType::DIRECTED) {
            return std::nullopt;
        }
        
        std::unordered_map<VertexId, int> inDegree;
        for (const auto& vertex : graph.vertices()) {
            inDegree[vertex] = graph.inDegree(vertex);
        }
        
        std::queue<VertexId> queue;
        for (const auto& [vertex, degree] : inDegree) {
            if (degree == 0) {
                queue.push(vertex);
            }
        }
        
        std::vector<VertexId> result;
        while (!queue.empty()) {
            VertexId current = queue.front();
            queue.pop();
            result.push_back(current);
            
            for (const auto& [neighbor, weight] : graph.getNeighbors(current)) {
                inDegree[neighbor]--;
                if (inDegree[neighbor] == 0) {
                    queue.push(neighbor);
                }
            }
        }
        
        if (result.size() != graph.numVertices()) {
            return std::nullopt; // Graph has a cycle
        }
        
        return result;
    }
};

// Simple benchmarking utility
class Benchmark {
public:
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;
    
    static void start(const std::string& name) {
        getInstance().startTimes_[name] = Clock::now();
    }
    
    static double end(const std::string& name) {
        auto endTime = Clock::now();
        auto& instance = getInstance();
        
        auto it = instance.startTimes_.find(name);
        if (it != instance.startTimes_.end()) {
            Duration duration = endTime - it->second;
            instance.results_[name].push_back(duration.count());
            instance.startTimes_.erase(it);
            return duration.count();
        }
        return -1.0;
    }
    
    static void printResults() {
        auto& instance = getInstance();
        std::cout << "\n=== Benchmark Results ===\n";
        
        for (const auto& [name, times] : instance.results_) {
            double avg = 0.0;
            double min = std::numeric_limits<double>::max();
            double max = std::numeric_limits<double>::min();
            
            for (double time : times) {
                avg += time;
                min = std::min(min, time);
                max = std::max(max, time);
            }
            avg /= times.size();
            
            std::cout << name << ":\n";
            std::cout << "  Runs: " << times.size() << "\n";
            std::cout << "  Avg: " << avg << " ms\n";
            std::cout << "  Min: " << min << " ms\n";
            std::cout << "  Max: " << max << " ms\n\n";
        }
    }
    
    static void clear() {
        getInstance().results_.clear();
        getInstance().startTimes_.clear();
    }
    
private:
    std::unordered_map<std::string, Clock::time_point> startTimes_;
    std::unordered_map<std::string, std::vector<double>> results_;
    
    static Benchmark& getInstance() {
        static Benchmark instance;
        return instance;
    }
    
    Benchmark() = default;
};

// Graph generator utilities for testing
class GraphGenerator {
public:
    template<typename VertexId = int, typename EdgeWeight = double>
    static Graph<VertexId, EdgeWeight> generateRandomGraph(
        size_t numVertices, 
        double edgeProbability, 
        EdgeWeight minWeight = 1.0, 
        EdgeWeight maxWeight = 10.0,
        typename Graph<VertexId, EdgeWeight>::GraphType type = Graph<VertexId, EdgeWeight>::GraphType::DIRECTED) {
        
        Graph<VertexId, EdgeWeight> graph(type);
        
        // Add vertices
        for (size_t i = 0; i < numVertices; ++i) {
            graph.addVertex(static_cast<VertexId>(i));
        }
        
        // Add edges randomly
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> probDist(0.0, 1.0);
        std::uniform_real_distribution<double> weightDist(
            static_cast<double>(minWeight), 
            static_cast<double>(maxWeight)
        );
        
        for (size_t i = 0; i < numVertices; ++i) {
            for (size_t j = (type == Graph<VertexId, EdgeWeight>::GraphType::UNDIRECTED ? i + 1 : 0); 
                 j < numVertices; ++j) {
                if (i != j && probDist(gen) < edgeProbability) {
                    graph.addEdge(
                        static_cast<VertexId>(i), 
                        static_cast<VertexId>(j), 
                        static_cast<EdgeWeight>(weightDist(gen))
                    );
                }
            }
        }
        
        return graph;
    }
    
    template<typename VertexId = int, typename EdgeWeight = double>
    static Graph<VertexId, EdgeWeight> generateCompleteGraph(
        size_t numVertices,
        EdgeWeight weight = 1.0,
        typename Graph<VertexId, EdgeWeight>::GraphType type = Graph<VertexId, EdgeWeight>::GraphType::DIRECTED) {
        
        Graph<VertexId, EdgeWeight> graph(type);
        
        for (size_t i = 0; i < numVertices; ++i) {
            graph.addVertex(static_cast<VertexId>(i));
        }
        
        for (size_t i = 0; i < numVertices; ++i) {
            for (size_t j = 0; j < numVertices; ++j) {
                if (i != j) {
                    graph.addEdge(
                        static_cast<VertexId>(i), 
                        static_cast<VertexId>(j), 
                        weight
                    );
                }
            }
        }
        
        return graph;
    }
    
    template<typename VertexId = int, typename EdgeWeight = double>
    static Graph<VertexId, EdgeWeight> generateGridGraph(
        size_t rows, 
        size_t cols,
        EdgeWeight weight = 1.0) {
        
        Graph<VertexId, EdgeWeight> graph(Graph<VertexId, EdgeWeight>::GraphType::UNDIRECTED);
        
        auto getVertexId = [cols](size_t row, size_t col) -> VertexId {
            return static_cast<VertexId>(row * cols + col);
        };
        
        // Add vertices
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                graph.addVertex(getVertexId(i, j));
            }
        }
        
        // Add edges
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                // Right edge
                if (j + 1 < cols) {
                    graph.addEdge(getVertexId(i, j), getVertexId(i, j + 1), weight);
                }
                // Down edge
                if (i + 1 < rows) {
                    graph.addEdge(getVertexId(i, j), getVertexId(i + 1, j), weight);
                }
            }
        }
        
        return graph;
    }
};

} // namespace pgof

#endif // GRAPH_FRAMEWORK_HPP
