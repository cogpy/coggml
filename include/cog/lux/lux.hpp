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

// ─────────────────────────────────────────────────────────────────────────────
// Graph Attention Network (GAT)
// Computes node embeddings using multi-head attention over neighbour edges.
// Each head learns separate attention coefficients for source/target features.
// ─────────────────────────────────────────────────────────────────────────────

// Single-head attention layer: dim_in -> dim_out
struct GATLayer {
    int dim_in;   // input feature dimension
    int dim_out;  // output feature dimension per head

    // Learnable weight matrices stored row-major (dim_out x dim_in)
    std::vector<float> W_src;  // source transform
    std::vector<float> W_dst;  // destination transform
    std::vector<float> a_src;  // attention source vector (dim_out)
    std::vector<float> a_dst;  // attention destination vector (dim_out)

    GATLayer() : dim_in(0), dim_out(0) {}

    GATLayer(int in, int out) : dim_in(in), dim_out(out) {
        W_src.assign(static_cast<size_t>(out * in), 0.0f);
        W_dst.assign(static_cast<size_t>(out * in), 0.0f);
        a_src.assign(static_cast<size_t>(out), 0.0f);
        a_dst.assign(static_cast<size_t>(out), 0.0f);
        // Xavier initialisation (variance = 1/fan_in)
        float scale = 1.0f / std::sqrt(static_cast<float>(in));
        // Deterministic seed-free initialisation for reproducibility
        for (int i = 0; i < out * in; ++i) {
            float v = scale * (((i * 6364136223846793005ULL + 1442695040888963407ULL) & 0xFFFF)
                               / 32768.0f - 1.0f);
            W_src[static_cast<size_t>(i)] = v;
            W_dst[static_cast<size_t>(i)] = -v;  // slight asymmetry
        }
        for (int i = 0; i < out; ++i) {
            a_src[static_cast<size_t>(i)] = scale;
            a_dst[static_cast<size_t>(i)] = scale;
        }
    }

    // Linear transform: out = W * in_vec  (dim_out result)
    std::vector<float> transform(const std::vector<float>& W,
                                 const std::vector<float>& x) const {
        std::vector<float> out(static_cast<size_t>(dim_out), 0.0f);
        for (int r = 0; r < dim_out; ++r) {
            float s = 0.0f;
            for (int c = 0; c < dim_in; ++c)
                s += W[static_cast<size_t>(r * dim_in + c)] * x[static_cast<size_t>(c)];
            out[static_cast<size_t>(r)] = s;
        }
        return out;
    }

    // Dot product of a with x
    float dot(const std::vector<float>& a, const std::vector<float>& x) const {
        float s = 0.0f;
        for (int i = 0; i < dim_out; ++i)
            s += a[static_cast<size_t>(i)] * x[static_cast<size_t>(i)];
        return s;
    }

    // LeakyReLU with negative slope 0.2
    static float leaky_relu(float x) { return x >= 0.0f ? x : 0.2f * x; }

    // Forward pass for one node: returns new embedding for node `v`
    // node_feats: map from NodeId -> feature vector of length dim_in
    // neighbours: NodeId list of in-neighbours of v
    std::vector<float> forward(
            NodeId v,
            const std::unordered_map<NodeId, std::vector<float>>& node_feats,
            const std::vector<NodeId>& neighbours) const {
        auto fv_it = node_feats.find(v);
        if (fv_it == node_feats.end())
            return std::vector<float>(static_cast<size_t>(dim_out), 0.0f);

        std::vector<float> h_v = transform(W_dst, fv_it->second);
        float attn_self = leaky_relu(dot(a_dst, h_v) + dot(a_src, h_v));

        std::vector<float> agg(static_cast<size_t>(dim_out), 0.0f);
        float attn_sum = std::exp(attn_self);

        for (NodeId u : neighbours) {
            auto fu_it = node_feats.find(u);
            if (fu_it == node_feats.end()) continue;
            std::vector<float> h_u = transform(W_src, fu_it->second);
            float attn_u = leaky_relu(dot(a_src, h_u) + dot(a_dst, h_v));
            float exp_u  = std::exp(attn_u);
            attn_sum += exp_u;
            for (int d = 0; d < dim_out; ++d)
                agg[static_cast<size_t>(d)] += exp_u * h_u[static_cast<size_t>(d)];
        }
        // Normalise + add self
        std::vector<float> result(static_cast<size_t>(dim_out), 0.0f);
        float inv_sum = (attn_sum > 1e-9f) ? (1.0f / attn_sum) : 0.0f;
        for (int d = 0; d < dim_out; ++d) {
            float s = std::exp(attn_self) * h_v[static_cast<size_t>(d)] + agg[static_cast<size_t>(d)];
            // ELU activation
            float z = s * inv_sum;
            result[static_cast<size_t>(d)] = (z >= 0.0f) ? z : (std::exp(z) - 1.0f);
        }
        return result;
    }
};

// Multi-layer GAT classifier for knowledge graph node embedding / classification
class GATClassifier {
public:
    struct Config {
        int input_dim;    // raw feature dimension
        int hidden_dim;   // hidden layer dimension
        int output_dim;   // embedding / class dimension
        int num_heads;    // attention heads (outputs concatenated)
        Config() : input_dim(16), hidden_dim(32), output_dim(16), num_heads(4) {}
    };

    explicit GATClassifier(const Config& cfg = Config()) : cfg_(cfg) {
        // Layer 0: input_dim -> hidden_dim (num_heads heads)
        layer0_.reserve(static_cast<size_t>(cfg.num_heads));
        for (int h = 0; h < cfg.num_heads; ++h)
            layer0_.emplace_back(cfg.input_dim, cfg.hidden_dim);
        // Layer 1: (hidden_dim * num_heads) -> output_dim (1 head)
        layer1_.emplace_back(cfg.hidden_dim * cfg.num_heads, cfg.output_dim);
    }

    // Compute embeddings for all nodes in the graph.
    // Returns map from NodeId -> embedding vector of length output_dim.
    // node_feats: initial node feature vectors (length input_dim).
    //   Missing nodes get zero-initialised features.
    std::unordered_map<NodeId, std::vector<float>> forward(
            const LuxGraph& g,
            const std::unordered_map<NodeId, std::vector<float>>& node_feats) const {
        // Build zero-padded / truncated features
        auto all_ids = g.all_nodes();
        std::unordered_map<NodeId, std::vector<float>> feats;
        feats.reserve(all_ids.size());
        for (NodeId nid : all_ids) {
            auto it = node_feats.find(nid);
            if (it != node_feats.end()) {
                std::vector<float> f = it->second;
                f.resize(static_cast<size_t>(cfg_.input_dim), 0.0f);
                feats[nid] = f;
            } else {
                feats[nid] = std::vector<float>(static_cast<size_t>(cfg_.input_dim), 0.0f);
            }
        }

        // Layer 0 — multi-head
        std::unordered_map<NodeId, std::vector<float>> h1;
        h1.reserve(all_ids.size());
        for (NodeId v : all_ids) {
            // Collect in-neighbours
            std::vector<NodeId> nbrs;
            for (EdgeId eid : g.in_edges(v)) {
                const auto* e = g.edge(eid);
                if (e) nbrs.push_back(e->source);
            }
            // Concatenate head outputs
            std::vector<float> concat;
            concat.reserve(static_cast<size_t>(cfg_.hidden_dim * cfg_.num_heads));
            for (const auto& head : layer0_) {
                auto h = head.forward(v, feats, nbrs);
                concat.insert(concat.end(), h.begin(), h.end());
            }
            h1[v] = concat;
        }

        // Layer 1 — single head
        std::unordered_map<NodeId, std::vector<float>> h2;
        h2.reserve(all_ids.size());
        for (NodeId v : all_ids) {
            std::vector<NodeId> nbrs;
            for (EdgeId eid : g.in_edges(v)) {
                const auto* e = g.edge(eid);
                if (e) nbrs.push_back(e->source);
            }
            h2[v] = layer1_[0].forward(v, h1, nbrs);
        }
        return h2;
    }

    int output_dim() const { return cfg_.output_dim; }

private:
    Config cfg_;
    std::vector<GATLayer> layer0_;
    std::vector<GATLayer> layer1_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Evidence classification
// Maps an EdgeType to a broad evidence category string suitable for
// Unicosys knowledge-graph evidence labelling.
// ─────────────────────────────────────────────────────────────────────────────
inline const char* classify_evidence(EdgeType t) {
    switch (t) {
        case EdgeType::INHERITANCE:  return "taxonomic";
        case EdgeType::SIMILARITY:   return "semantic";
        case EdgeType::IMPLICATION:  return "logical";
        case EdgeType::EVALUATION:   return "evaluative";
        case EdgeType::EXECUTION:    return "procedural";
        case EdgeType::MEMBER:       return "membership";
        case EdgeType::CONTEXT:      return "contextual";
        case EdgeType::TEMPORAL:     return "temporal";
        case EdgeType::CAUSAL:       return "causal";
        case EdgeType::ANALOGY:      return "analogical";
        case EdgeType::ATTENTION:    return "attentional";
        default:                     return "unknown";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Subsystem index
// Groups node IDs by their NodeType for fast subsystem look-up.
// ─────────────────────────────────────────────────────────────────────────────
using SubsystemIndex = std::unordered_map<NodeType, std::vector<NodeId>>;

inline SubsystemIndex index_subsystems(const LuxGraph& g) {
    SubsystemIndex idx;
    for (NodeId nid : g.all_nodes()) {
        const LuxNode* n = g.node(nid);
        if (n) idx[n->type].push_back(nid);
    }
    return idx;
}

// ─────────────────────────────────────────────────────────────────────────────
// Link prediction
// Scores the likelihood of a directed edge u -> v using the dot product of
// their GAT embeddings. Higher score ≈ more likely edge.
// ─────────────────────────────────────────────────────────────────────────────
inline float link_predict(
        const std::unordered_map<NodeId, std::vector<float>>& embeddings,
        NodeId u,
        NodeId v) {
    auto it_u = embeddings.find(u);
    auto it_v = embeddings.find(v);
    if (it_u == embeddings.end() || it_v == embeddings.end()) return 0.0f;
    const auto& eu = it_u->second;
    const auto& ev = it_v->second;
    size_t dim = std::min(eu.size(), ev.size());
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) dot += eu[i] * ev[i];
    // Sigmoid to map to [0,1] probability
    return 1.0f / (1.0f + std::exp(-dot));
}

}} // namespace cog::lux

#endif // COG_LUX_HPP
