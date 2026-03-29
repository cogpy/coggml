// cog/lux/lux.hpp — Cognitive Node Graph
// Typed nodes/edges, BFS/DFS, PageRank, DOT export
// Header-only, C++11, zero external dependencies
// SPDX-License-Identifier: MIT
#ifndef COG_LUX_HPP
#define COG_LUX_HPP

#include "../core/core.hpp"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <functional>
#include <algorithm>
#include <sstream>
#include <cassert>
#include <cmath>

namespace cog { namespace lux {

// ─────────────────────────────────────────────────────────────────────────────
// Node & Edge Types
// ─────────────────────────────────────────────────────────────────────────────
enum class NodeType : uint16_t {
    CONCEPT      = 0,
    PREDICATE    = 1,
    SCHEMA       = 2,
    PROCEDURE    = 3,
    VARIABLE     = 4,
    EMBEDDING    = 5,
    CONTEXT      = 6,
    GOAL         = 7,
    BELIEF       = 8,
    MEMORY       = 9,
    PERCEPTION   = 10,
    ACTION       = 11,
    COUNT        = 12
};

enum class EdgeType : uint16_t {
    INHERITANCE    = 0,
    SIMILARITY     = 1,
    IMPLICATION    = 2,
    EVALUATION     = 3,
    EXECUTION      = 4,
    MEMBER         = 5,
    CONTEXT        = 6,
    TEMPORAL       = 7,
    CAUSAL         = 8,
    ANALOGY        = 9,
    ATTENTION      = 10,
    COUNT          = 11
};

inline const char* node_type_name(NodeType t) {
    static const char* names[] = {
        "Concept","Predicate","Schema","Procedure","Variable",
        "Embedding","Context","Goal","Belief","Memory","Perception","Action"
    };
    size_t idx = static_cast<size_t>(t);
    return (idx < 12) ? names[idx] : "Unknown";
}

inline const char* edge_type_name(EdgeType t) {
    static const char* names[] = {
        "Inheritance","Similarity","Implication","Evaluation","Execution",
        "Member","Context","Temporal","Causal","Analogy","Attention"
    };
    size_t idx = static_cast<size_t>(t);
    return (idx < 11) ? names[idx] : "Unknown";
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph Node
// ─────────────────────────────────────────────────────────────────────────────
using NodeId = uint32_t;
using EdgeId = uint32_t;

static const NodeId INVALID_NODE = 0;
static const EdgeId INVALID_EDGE = 0;

struct LuxNode {
    NodeId      id;
    NodeType    type;
    std::string label;
    float       strength;    // truth value strength [0,1]
    float       confidence;  // truth value confidence [0,1]
    float       sti;         // short-term importance (attention)
    float       lti;         // long-term importance
    std::unordered_map<std::string, std::string> attrs;

    LuxNode() : id(INVALID_NODE), type(NodeType::CONCEPT),
                strength(1.0f), confidence(0.9f),
                sti(0.0f), lti(0.0f) {}

    LuxNode(NodeId i, NodeType t, const std::string& l)
        : id(i), type(t), label(l),
          strength(1.0f), confidence(0.9f),
          sti(0.0f), lti(0.0f) {}

    void set_attr(const std::string& k, const std::string& v) {
        attrs[k] = v;
    }

    std::string get_attr(const std::string& k, const std::string& def = "") const {
        auto it = attrs.find(k);
        return (it != attrs.end()) ? it->second : def;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Graph Edge
// ─────────────────────────────────────────────────────────────────────────────
struct LuxEdge {
    EdgeId      id;
    EdgeType    type;
    NodeId      source;
    NodeId      target;
    float       weight;
    float       strength;
    float       confidence;
    std::string label;

    LuxEdge() : id(INVALID_EDGE), type(EdgeType::INHERITANCE),
                source(INVALID_NODE), target(INVALID_NODE),
                weight(1.0f), strength(1.0f), confidence(0.9f) {}

    LuxEdge(EdgeId i, EdgeType t, NodeId s, NodeId tgt, float w = 1.0f)
        : id(i), type(t), source(s), target(tgt), weight(w),
          strength(1.0f), confidence(0.9f) {}
};

// ─────────────────────────────────────────────────────────────────────────────
// LuxGraph — Cognitive Node Graph
// ─────────────────────────────────────────────────────────────────────────────
class LuxGraph {
public:
    LuxGraph() : next_node_(1), next_edge_(1) {}

    // ── Node operations ──────────────────────────────────────────────────────

    NodeId add_node(NodeType type, const std::string& label) {
        NodeId id = next_node_++;
        LuxNode n(id, type, label);
        nodes_[id] = n;
        label_to_id_[label] = id;
        adj_out_[id];  // ensure adjacency entry exists
        adj_in_[id];
        return id;
    }

    LuxNode* node(NodeId id) {
        auto it = nodes_.find(id);
        return (it != nodes_.end()) ? &it->second : nullptr;
    }

    const LuxNode* node(NodeId id) const {
        auto it = nodes_.find(id);
        return (it != nodes_.end()) ? &it->second : nullptr;
    }

    NodeId find_node(const std::string& label) const {
        auto it = label_to_id_.find(label);
        return (it != label_to_id_.end()) ? it->second : INVALID_NODE;
    }

    bool has_node(NodeId id) const {
        return nodes_.find(id) != nodes_.end();
    }

    size_t node_count() const { return nodes_.size(); }

    // ── Edge operations ──────────────────────────────────────────────────────

    EdgeId add_edge(EdgeType type, NodeId src, NodeId tgt, float weight = 1.0f) {
        assert(has_node(src) && has_node(tgt));
        EdgeId id = next_edge_++;
        LuxEdge e(id, type, src, tgt, weight);
        edges_[id] = e;
        adj_out_[src].push_back(id);
        adj_in_[tgt].push_back(id);
        return id;
    }

    LuxEdge* edge(EdgeId id) {
        auto it = edges_.find(id);
        return (it != edges_.end()) ? &it->second : nullptr;
    }

    const LuxEdge* edge(EdgeId id) const {
        auto it = edges_.find(id);
        return (it != edges_.end()) ? &it->second : nullptr;
    }

    size_t edge_count() const { return edges_.size(); }

    const std::vector<EdgeId>& out_edges(NodeId id) const {
        static const std::vector<EdgeId> empty;
        auto it = adj_out_.find(id);
        return (it != adj_out_.end()) ? it->second : empty;
    }

    const std::vector<EdgeId>& in_edges(NodeId id) const {
        static const std::vector<EdgeId> empty;
        auto it = adj_in_.find(id);
        return (it != adj_in_.end()) ? it->second : empty;
    }

    // ── Traversal ────────────────────────────────────────────────────────────

    // BFS from source, up to max_depth hops; returns visited nodes in BFS order
    std::vector<NodeId> bfs(NodeId src, size_t max_depth = SIZE_MAX) const {
        std::vector<NodeId> result;
        if (!has_node(src)) return result;
        std::unordered_set<NodeId> visited;
        std::queue<std::pair<NodeId, size_t>> q;
        q.push({src, 0});
        visited.insert(src);
        while (!q.empty()) {
            auto front = q.front(); q.pop();
            NodeId cur = front.first;
            size_t depth = front.second;
            result.push_back(cur);
            if (depth >= max_depth) continue;
            for (EdgeId eid : out_edges(cur)) {
                const auto* e = edge(eid);
                if (e && visited.find(e->target) == visited.end()) {
                    visited.insert(e->target);
                    q.push({e->target, depth + 1});
                }
            }
        }
        return result;
    }

    // DFS from source; returns visited nodes in DFS order
    std::vector<NodeId> dfs(NodeId src) const {
        std::vector<NodeId> result;
        if (!has_node(src)) return result;
        std::unordered_set<NodeId> visited;
        dfs_helper(src, visited, result);
        return result;
    }

    // ── PageRank ─────────────────────────────────────────────────────────────
    // Returns map of node_id → pagerank score
    std::unordered_map<NodeId, float> pagerank(
            float damping = 0.85f, int max_iters = 100,
            float tol = 1e-6f) const {
        size_t N = nodes_.size();
        if (N == 0) return {};

        // Initialize uniform ranks
        std::unordered_map<NodeId, float> rank;
        float init = 1.0f / (float)N;
        for (auto& kv : nodes_) rank[kv.first] = init;

        for (int iter = 0; iter < max_iters; ++iter) {
            std::unordered_map<NodeId, float> new_rank;
            float dangling_sum = 0.0f;

            for (auto& kv : nodes_) {
                new_rank[kv.first] = 0.0f;
                const auto& outs = out_edges(kv.first);
                if (outs.empty()) dangling_sum += rank[kv.first];
            }

            for (auto& kv : nodes_) {
                NodeId v = kv.first;
                const auto& ins = in_edges(v);
                float incoming = 0.0f;
                for (EdgeId eid : ins) {
                    const auto* e = edge(eid);
                    if (!e) continue;
                    NodeId u = e->source;
                    size_t out_deg = out_edges(u).size();
                    if (out_deg > 0) {
                        incoming += rank[u] / (float)out_deg;
                    }
                }
                new_rank[v] = (1.0f - damping) / (float)N
                            + damping * (incoming + dangling_sum / (float)N);
            }

            // Check convergence
            float diff = 0.0f;
            for (auto& kv : nodes_) {
                diff += std::fabs(new_rank[kv.first] - rank[kv.first]);
            }
            rank = new_rank;
            if (diff < tol) break;
        }
        return rank;
    }

    // ── DOT Export ───────────────────────────────────────────────────────────
    std::string to_dot(const std::string& graph_name = "CogGraph") const {
        std::ostringstream ss;
        ss << "digraph " << graph_name << " {\n";
        ss << "  rankdir=LR;\n";
        ss << "  node [shape=ellipse];\n";

        for (auto& kv : nodes_) {
            const LuxNode& n = kv.second;
            ss << "  n" << n.id << " [label=\"" << n.label
               << "\" type=\"" << node_type_name(n.type) << "\""
               << " strength=" << n.strength << "];\n";
        }

        for (auto& kv : edges_) {
            const LuxEdge& e = kv.second;
            ss << "  n" << e.source << " -> n" << e.target
               << " [label=\"" << edge_type_name(e.type) << "\""
               << " weight=" << e.weight << "];\n";
        }
        ss << "}\n";
        return ss.str();
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    std::vector<NodeId> all_nodes() const {
        std::vector<NodeId> ids;
        ids.reserve(nodes_.size());
        for (auto& kv : nodes_) ids.push_back(kv.first);
        return ids;
    }

    std::vector<EdgeId> all_edges() const {
        std::vector<EdgeId> ids;
        ids.reserve(edges_.size());
        for (auto& kv : edges_) ids.push_back(kv.first);
        return ids;
    }

private:
    NodeId next_node_;
    EdgeId next_edge_;
    std::unordered_map<NodeId, LuxNode>         nodes_;
    std::unordered_map<EdgeId, LuxEdge>         edges_;
    std::unordered_map<std::string, NodeId>     label_to_id_;
    std::unordered_map<NodeId, std::vector<EdgeId>> adj_out_;
    std::unordered_map<NodeId, std::vector<EdgeId>> adj_in_;

    void dfs_helper(NodeId cur, std::unordered_set<NodeId>& visited,
                    std::vector<NodeId>& result) const {
        visited.insert(cur);
        result.push_back(cur);
        for (EdgeId eid : out_edges(cur)) {
            const auto* e = edge(eid);
            if (e && visited.find(e->target) == visited.end()) {
                dfs_helper(e->target, visited, result);
            }
        }
    }
};

}} // namespace cog::lux

#endif // COG_LUX_HPP
