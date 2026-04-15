// cog/glow/glow.hpp — Neural Network Compiler
// Graph IR, optimization passes, interpreter
// Header-only, C++11, zero external dependencies
// SPDX-License-Identifier: MIT
#ifndef COG_GLOW_HPP
#define COG_GLOW_HPP

#include "../core/core.hpp"
#include "../gml/gml.hpp"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cassert>
#include <stdexcept>

namespace cog { namespace glow {

// ─────────────────────────────────────────────────────────────────────────────
// Graph IR — Operation Types
// ─────────────────────────────────────────────────────────────────────────────
enum class OpType : uint16_t {
    // Input/Output
    INPUT       = 0,
    OUTPUT      = 1,
    CONSTANT    = 2,
    // Arithmetic
    ADD         = 10,
    SUB         = 11,
    MUL         = 12,
    DIV         = 13,
    NEG         = 14,
    ABS         = 15,
    // Math
    EXP         = 20,
    LOG         = 21,
    SQRT        = 22,
    TANH        = 23,
    SIGMOID     = 24,
    RELU        = 25,
    GELU        = 26,
    // Linear algebra
    MATMUL      = 30,
    TRANSPOSE   = 31,
    RESHAPE     = 32,
    CONCAT      = 33,
    SLICE       = 34,
    // Reduction
    SUM         = 40,
    MEAN        = 41,
    MAX         = 42,
    MIN         = 43,
    SOFTMAX     = 44,
    LAYER_NORM  = 45,
    // NN layers
    LINEAR      = 50,
    EMBEDDING   = 51,
    ATTENTION   = 52,
    DROPOUT     = 53,
    // Control
    NOP         = 100
};

inline const char* op_type_name(OpType t) {
    switch (t) {
        case OpType::INPUT:      return "Input";
        case OpType::OUTPUT:     return "Output";
        case OpType::CONSTANT:   return "Constant";
        case OpType::ADD:        return "Add";
        case OpType::SUB:        return "Sub";
        case OpType::MUL:        return "Mul";
        case OpType::DIV:        return "Div";
        case OpType::NEG:        return "Neg";
        case OpType::ABS:        return "Abs";
        case OpType::EXP:        return "Exp";
        case OpType::LOG:        return "Log";
        case OpType::SQRT:       return "Sqrt";
        case OpType::TANH:       return "Tanh";
        case OpType::SIGMOID:    return "Sigmoid";
        case OpType::RELU:       return "Relu";
        case OpType::GELU:       return "Gelu";
        case OpType::MATMUL:     return "MatMul";
        case OpType::TRANSPOSE:  return "Transpose";
        case OpType::RESHAPE:    return "Reshape";
        case OpType::CONCAT:     return "Concat";
        case OpType::SLICE:      return "Slice";
        case OpType::SUM:        return "Sum";
        case OpType::MEAN:       return "Mean";
        case OpType::MAX:        return "Max";
        case OpType::MIN:        return "Min";
        case OpType::SOFTMAX:    return "Softmax";
        case OpType::LAYER_NORM: return "LayerNorm";
        case OpType::LINEAR:     return "Linear";
        case OpType::EMBEDDING:  return "Embedding";
        case OpType::ATTENTION:  return "Attention";
        case OpType::DROPOUT:    return "Dropout";
        case OpType::NOP:        return "Nop";
        default:                 return "Unknown";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IR Value (type-annotated edge in the computation graph)
// ─────────────────────────────────────────────────────────────────────────────
using ValueId = uint32_t;
static const ValueId INVALID_VALUE = 0;

struct IRType {
    gml::DType dtype;
    std::vector<size_t> shape;

    IRType() : dtype(gml::DType::F32) {}
    IRType(gml::DType dt, std::vector<size_t> sh)
        : dtype(dt), shape(std::move(sh)) {}

    size_t numel() const {
        if (shape.empty()) return 0;
        size_t n = 1;
        for (size_t s : shape) n *= s;
        return n;
    }

    std::string to_string() const {
        std::ostringstream ss;
        ss << gml::dtype_name(dtype) << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) ss << ",";
            ss << shape[i];
        }
        ss << "]";
        return ss.str();
    }

    bool operator==(const IRType& o) const {
        return dtype == o.dtype && shape == o.shape;
    }
};

struct IRValue {
    ValueId id;
    IRType  type;
    std::string name;
    std::vector<float> const_data;  // filled for CONSTANT nodes

    IRValue() : id(INVALID_VALUE) {}
    IRValue(ValueId i, const IRType& t, const std::string& n = "")
        : id(i), type(t), name(n) {}
};

// ─────────────────────────────────────────────────────────────────────────────
// IR Node (operation in the graph)
// ─────────────────────────────────────────────────────────────────────────────
using NodeId = uint32_t;
static const NodeId INVALID_NODE = 0;

struct IRNode {
    NodeId              id;
    OpType              op;
    std::string         name;
    std::vector<ValueId> inputs;
    std::vector<ValueId> outputs;
    // Op-specific attributes
    std::unordered_map<std::string, std::string> attrs;

    IRNode() : id(INVALID_NODE), op(OpType::NOP) {}
    IRNode(NodeId i, OpType t, const std::string& n = "")
        : id(i), op(t), name(n) {}

    void set_attr(const std::string& k, const std::string& v) {
        attrs[k] = v;
    }

    std::string get_attr(const std::string& k,
                         const std::string& def = "") const {
        auto it = attrs.find(k);
        return (it != attrs.end()) ? it->second : def;
    }

    int get_int_attr(const std::string& k, int def = 0) const {
        auto it = attrs.find(k);
        if (it == attrs.end()) return def;
        return std::atoi(it->second.c_str());
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// GlowGraph — IR computation graph
// ─────────────────────────────────────────────────────────────────────────────
class GlowGraph {
public:
    GlowGraph() : next_node_(1), next_value_(1) {}

    // ── Value management ─────────────────────────────────────────────────────

    ValueId add_value(const IRType& type, const std::string& name = "") {
        ValueId id = next_value_++;
        values_[id] = IRValue(id, type, name);
        return id;
    }

    IRValue* value(ValueId id) {
        auto it = values_.find(id);
        return (it != values_.end()) ? &it->second : nullptr;
    }

    const IRValue* value(ValueId id) const {
        auto it = values_.find(id);
        return (it != values_.end()) ? &it->second : nullptr;
    }

    size_t value_count() const { return values_.size(); }

    // ── Node management ──────────────────────────────────────────────────────

    NodeId add_node(OpType op, const std::string& name = "") {
        NodeId id = next_node_++;
        nodes_[id] = IRNode(id, op, name);
        return id;
    }

    IRNode* node(NodeId id) {
        auto it = nodes_.find(id);
        return (it != nodes_.end()) ? &it->second : nullptr;
    }

    const IRNode* node(NodeId id) const {
        auto it = nodes_.find(id);
        return (it != nodes_.end()) ? &it->second : nullptr;
    }

    size_t node_count() const { return nodes_.size(); }

    // Connect node output to value, and track which node produces each value
    void connect_output(NodeId nid, ValueId vid) {
        auto* n = node(nid);
        if (n) n->outputs.push_back(vid);
        value_producer_[vid] = nid;
    }

    void connect_input(NodeId nid, ValueId vid) {
        auto* n = node(nid);
        if (n) n->inputs.push_back(vid);
    }

    NodeId producer(ValueId vid) const {
        auto it = value_producer_.find(vid);
        return (it != value_producer_.end()) ? it->second : INVALID_NODE;
    }

    // ── Graph inputs/outputs ─────────────────────────────────────────────────

    void add_input(ValueId vid)  { graph_inputs_.push_back(vid);  }
    void add_output(ValueId vid) { graph_outputs_.push_back(vid); }

    const std::vector<ValueId>& inputs()  const { return graph_inputs_;  }
    const std::vector<ValueId>& outputs() const { return graph_outputs_; }

    // ── Topological sort ─────────────────────────────────────────────────────
    std::vector<NodeId> topological_order() const {
        std::unordered_map<NodeId, int> in_degree;
        std::unordered_map<NodeId, std::vector<NodeId>> adj;

        for (auto& kv : nodes_) in_degree[kv.first] = 0;

        for (auto& kv : nodes_) {
            const IRNode& n = kv.second;
            for (ValueId vid : n.inputs) {
                NodeId prod = producer(vid);
                if (prod != INVALID_NODE) {
                    adj[prod].push_back(n.id);
                    in_degree[n.id]++;
                }
            }
        }

        std::queue<NodeId> q;
        for (auto& kv : in_degree) {
            if (kv.second == 0) q.push(kv.first);
        }

        std::vector<NodeId> order;
        while (!q.empty()) {
            NodeId cur = q.front(); q.pop();
            order.push_back(cur);
            for (NodeId next : adj[cur]) {
                if (--in_degree[next] == 0) q.push(next);
            }
        }
        return order;
    }

    // ── DOT export ───────────────────────────────────────────────────────────
    std::string to_dot(const std::string& name = "GlowGraph") const {
        std::ostringstream ss;
        ss << "digraph " << name << " {\n";
        ss << "  rankdir=TB;\n";
        for (auto& kv : nodes_) {
            const IRNode& n = kv.second;
            ss << "  n" << n.id << " [label=\""
               << op_type_name(n.op);
            if (!n.name.empty()) ss << "\\n" << n.name;
            ss << "\"];\n";
        }
        for (auto& kv : nodes_) {
            const IRNode& n = kv.second;
            for (ValueId vid : n.inputs) {
                NodeId prod = producer(vid);
                if (prod != INVALID_NODE) {
                    ss << "  n" << prod << " -> n" << n.id;
                    auto* v = value(vid);
                    if (v) ss << " [label=\"" << v->type.to_string() << "\"]";
                    ss << ";\n";
                }
            }
        }
        ss << "}\n";
        return ss.str();
    }

private:
    NodeId  next_node_;
    ValueId next_value_;
    std::unordered_map<NodeId,  IRNode>   nodes_;
    std::unordered_map<ValueId, IRValue>  values_;
    std::unordered_map<ValueId, NodeId>   value_producer_;
    std::vector<ValueId> graph_inputs_;
    std::vector<ValueId> graph_outputs_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Optimization Passes
// ─────────────────────────────────────────────────────────────────────────────

// Dead code elimination: remove nodes whose outputs are not consumed
inline int pass_dce(GlowGraph& g) {
    // Build use-count for each value
    std::unordered_map<ValueId, int> use_count;
    for (auto nid : g.topological_order()) {
        const auto* n = g.node(nid);
        if (!n) continue;
        for (ValueId vid : n->inputs) use_count[vid]++;
    }
    for (ValueId vid : g.outputs()) use_count[vid]++;

    int removed = 0;
    // NOP nodes that produce only unused values
    for (auto nid : g.topological_order()) {
        auto* n = g.node(nid);
        if (!n || n->op == OpType::INPUT || n->op == OpType::OUTPUT) continue;
        bool all_unused = !n->outputs.empty();
        for (ValueId vid : n->outputs) {
            if (use_count[vid] > 0) { all_unused = false; break; }
        }
        if (all_unused) {
            n->op = OpType::NOP;
            ++removed;
        }
    }
    return removed;
}

// Constant folding: fuse consecutive constants
inline int pass_constant_fold(GlowGraph& g) {
    int folded = 0;
    for (auto nid : g.topological_order()) {
        auto* n = g.node(nid);
        if (!n || n->op != OpType::ADD) continue;
        if (n->inputs.size() < 2) continue;

        auto* v0 = g.value(n->inputs[0]);
        auto* v1 = g.value(n->inputs[1]);
        if (!v0 || !v1) continue;

        NodeId p0 = g.producer(n->inputs[0]);
        NodeId p1 = g.producer(n->inputs[1]);
        const auto* pn0 = g.node(p0);
        const auto* pn1 = g.node(p1);
        if (!pn0 || !pn1) continue;
        if (pn0->op != OpType::CONSTANT || pn1->op != OpType::CONSTANT) continue;

        // Fold: merge constant data
        if (!v0->const_data.empty() && !v1->const_data.empty() &&
            v0->const_data.size() == v1->const_data.size()) {
            IRValue* out_val = g.value(n->outputs.empty() ? INVALID_VALUE : n->outputs[0]);
            if (out_val) {
                out_val->const_data.resize(v0->const_data.size());
                for (size_t i = 0; i < v0->const_data.size(); ++i) {
                    out_val->const_data[i] = v0->const_data[i] + v1->const_data[i];
                }
                n->op = OpType::CONSTANT;
                ++folded;
            }
        }
    }
    return folded;
}

// ─────────────────────────────────────────────────────────────────────────────
// Interpreter — execute GlowGraph on float tensors
// ─────────────────────────────────────────────────────────────────────────────
class GlowInterpreter {
public:
    using TensorMap = std::unordered_map<ValueId, std::vector<float>>;

    // Execute graph with provided input tensors
    TensorMap run(const GlowGraph& g,
                  const TensorMap& inputs) {
        TensorMap env = inputs;

        // Load constants
        for (auto nid : g.topological_order()) {
            const auto* n = g.node(nid);
            if (!n || n->op != OpType::CONSTANT) continue;
            for (ValueId vid : n->outputs) {
                const auto* v = g.value(vid);
                if (v && !v->const_data.empty()) {
                    env[vid] = v->const_data;
                }
            }
        }

        // Execute in topological order
        for (auto nid : g.topological_order()) {
            const auto* n = g.node(nid);
            if (!n) continue;
            exec_node(g, *n, env);
        }

        // Collect outputs
        TensorMap result;
        for (ValueId vid : g.outputs()) {
            auto it = env.find(vid);
            if (it != env.end()) result[vid] = it->second;
        }
        return result;
    }

private:
    static std::vector<float> apply_elementwise(
            const std::vector<float>& a, const std::vector<float>& b,
            std::function<float(float, float)> fn) {
        size_t n = std::min(a.size(), b.size());
        std::vector<float> out(n);
        for (size_t i = 0; i < n; ++i) out[i] = fn(a[i], b[i]);
        return out;
    }

    static std::vector<float> apply_unary(
            const std::vector<float>& a,
            std::function<float(float)> fn) {
        std::vector<float> out(a.size());
        for (size_t i = 0; i < a.size(); ++i) out[i] = fn(a[i]);
        return out;
    }

    void exec_node(const GlowGraph& g, const IRNode& n, TensorMap& env) {
        auto get = [&](ValueId vid) -> const std::vector<float>& {
            static const std::vector<float> empty;
            auto it = env.find(vid);
            return (it != env.end()) ? it->second : empty;
        };
        auto put = [&](ValueId vid, std::vector<float> data) {
            env[vid] = std::move(data);
        };

        ValueId out_vid = n.outputs.empty() ? INVALID_VALUE : n.outputs[0];

        switch (n.op) {
            case OpType::ADD: {
                if (n.inputs.size() < 2) break;
                put(out_vid, apply_elementwise(get(n.inputs[0]), get(n.inputs[1]),
                    [](float a, float b){ return a + b; }));
                break;
            }
            case OpType::SUB: {
                if (n.inputs.size() < 2) break;
                put(out_vid, apply_elementwise(get(n.inputs[0]), get(n.inputs[1]),
                    [](float a, float b){ return a - b; }));
                break;
            }
            case OpType::MUL: {
                if (n.inputs.size() < 2) break;
                put(out_vid, apply_elementwise(get(n.inputs[0]), get(n.inputs[1]),
                    [](float a, float b){ return a * b; }));
                break;
            }
            case OpType::DIV: {
                if (n.inputs.size() < 2) break;
                put(out_vid, apply_elementwise(get(n.inputs[0]), get(n.inputs[1]),
                    [](float a, float b){ return b != 0.0f ? a/b : 0.0f; }));
                break;
            }
            case OpType::NEG: {
                if (n.inputs.empty()) break;
                put(out_vid, apply_unary(get(n.inputs[0]),
                    [](float a){ return -a; }));
                break;
            }
            case OpType::ABS: {
                if (n.inputs.empty()) break;
                put(out_vid, apply_unary(get(n.inputs[0]),
                    [](float a){ return std::fabs(a); }));
                break;
            }
            case OpType::EXP: {
                if (n.inputs.empty()) break;
                put(out_vid, apply_unary(get(n.inputs[0]),
                    [](float a){ return std::exp(a); }));
                break;
            }
            case OpType::LOG: {
                if (n.inputs.empty()) break;
                put(out_vid, apply_unary(get(n.inputs[0]),
                    [](float a){ return a > 0.0f ? std::log(a) : -1e30f; }));
                break;
            }
            case OpType::SQRT: {
                if (n.inputs.empty()) break;
                put(out_vid, apply_unary(get(n.inputs[0]),
                    [](float a){ return a >= 0.0f ? std::sqrt(a) : 0.0f; }));
                break;
            }
            case OpType::TANH: {
                if (n.inputs.empty()) break;
                put(out_vid, apply_unary(get(n.inputs[0]),
                    [](float a){ return std::tanh(a); }));
                break;
            }
            case OpType::SIGMOID: {
                if (n.inputs.empty()) break;
                put(out_vid, apply_unary(get(n.inputs[0]),
                    [](float a){ return 1.0f / (1.0f + std::exp(-a)); }));
                break;
            }
            case OpType::RELU: {
                if (n.inputs.empty()) break;
                put(out_vid, apply_unary(get(n.inputs[0]),
                    [](float a){ return a > 0.0f ? a : 0.0f; }));
                break;
            }
            case OpType::GELU: {
                if (n.inputs.empty()) break;
                put(out_vid, apply_unary(get(n.inputs[0]),
                    [](float a){ return 0.5f * a * (1.0f + std::tanh(
                        0.7978845608028654f * (a + 0.044715f * a*a*a))); }));
                break;
            }
            case OpType::SOFTMAX: {
                if (n.inputs.empty()) break;
                const auto& x = get(n.inputs[0]);
                if (x.empty()) break;
                float max_val = *std::max_element(x.begin(), x.end());
                std::vector<float> ex(x.size());
                float sum = 0.0f;
                for (size_t i = 0; i < x.size(); ++i) {
                    ex[i] = std::exp(x[i] - max_val);
                    sum += ex[i];
                }
                if (sum > 0.0f) for (auto& v : ex) v /= sum;
                put(out_vid, ex);
                break;
            }
            case OpType::SUM: {
                if (n.inputs.empty()) break;
                const auto& x = get(n.inputs[0]);
                float s = 0.0f;
                for (float v : x) s += v;
                put(out_vid, {s});
                break;
            }
            case OpType::MEAN: {
                if (n.inputs.empty()) break;
                const auto& x = get(n.inputs[0]);
                if (x.empty()) { put(out_vid, {0.0f}); break; }
                float s = 0.0f;
                for (float v : x) s += v;
                put(out_vid, {s / (float)x.size()});
                break;
            }
            case OpType::INPUT:
            case OpType::CONSTANT:
            case OpType::NOP:
                break;  // handled separately
            default:
                break;
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// GlowCompiler — graph builder helper
// ─────────────────────────────────────────────────────────────────────────────
class GlowCompiler {
public:
    GlowGraph& graph() { return g_; }

    ValueId input(const std::vector<size_t>& shape,
                  gml::DType dtype = gml::DType::F32,
                  const std::string& name = "") {
        ValueId vid = g_.add_value(IRType(dtype, shape), name);
        NodeId  nid = g_.add_node(OpType::INPUT, name);
        g_.connect_output(nid, vid);
        g_.add_input(vid);
        return vid;
    }

    ValueId constant(const std::vector<float>& data,
                     const std::vector<size_t>& shape,
                     const std::string& name = "") {
        ValueId vid = g_.add_value(IRType(gml::DType::F32, shape), name);
        auto*   v   = g_.value(vid);
        if (v) v->const_data = data;
        NodeId  nid = g_.add_node(OpType::CONSTANT, name);
        g_.connect_output(nid, vid);
        return vid;
    }

    ValueId op1(OpType op, ValueId a, const std::string& name = "") {
        IRType ty;
        if (const auto* va = g_.value(a)) ty = va->type;
        ValueId vid = g_.add_value(ty, name);
        NodeId  nid = g_.add_node(op, name);
        g_.connect_input(nid, a);
        g_.connect_output(nid, vid);
        return vid;
    }

    ValueId op2(OpType op, ValueId a, ValueId b,
                const std::string& name = "") {
        IRType ty;
        if (const auto* va = g_.value(a)) ty = va->type;
        ValueId vid = g_.add_value(ty, name);
        NodeId  nid = g_.add_node(op, name);
        g_.connect_input(nid, a);
        g_.connect_input(nid, b);
        g_.connect_output(nid, vid);
        return vid;
    }

    void mark_output(ValueId vid) { g_.add_output(vid); }

private:
    GlowGraph g_;
};

}} // namespace cog::glow

#endif // COG_GLOW_HPP
