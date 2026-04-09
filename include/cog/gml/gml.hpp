// cog/gml/gml.hpp — Tensor Library for Machine Learning
// N-dimensional tensors, computation graphs, auto-diff, quantization, optimizers
// Header-only, C++11, zero external dependencies
// SPDX-License-Identifier: MIT
#ifndef COG_GML_HPP
#define COG_GML_HPP

#include "../core/core.hpp"
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <functional>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <sstream>

namespace cog { namespace gml {

// ─────────────────────────────────────────────────────────────────────────────
// Data Types
// ─────────────────────────────────────────────────────────────────────────────
enum class DType : uint8_t {
    F32    = 0,
    F16    = 1,
    Q4_0   = 2,   // 4-bit quantization, block size 32
    Q4_1   = 3,   // 4-bit quantization with min, block size 32
    Q5_0   = 4,
    Q5_1   = 5,
    Q8_0   = 6,   // 8-bit quantization, block size 32
    I32    = 7,
    COUNT  = 8
};

inline size_t dtype_size(DType t) {
    switch (t) {
        case DType::F32:  return 4;
        case DType::F16:  return 2;
        case DType::Q4_0: return 18;  // per block of 32: 2 bytes scale + 16 bytes data
        case DType::Q4_1: return 20;  // per block of 32: 2+2 bytes + 16 bytes data
        case DType::Q5_0: return 22;
        case DType::Q5_1: return 24;
        case DType::Q8_0: return 34;  // per block of 32: 2 bytes scale + 32 bytes data
        case DType::I32:  return 4;
        default: return 0;
    }
}

inline const char* dtype_name(DType t) {
    static const char* names[] = {"f32","f16","q4_0","q4_1","q5_0","q5_1","q8_0","i32"};
    return names[static_cast<uint8_t>(t)];
}

static constexpr size_t QK = 32; // Quantization block size

// ─────────────────────────────────────────────────────────────────────────────
// Quantization blocks
// ─────────────────────────────────────────────────────────────────────────────
struct BlockQ4_0 {
    uint16_t d;          // Scale (f16)
    uint8_t  qs[QK/2];  // Quantized values (4 bits each)
};

struct BlockQ8_0 {
    uint16_t d;          // Scale (f16)
    int8_t   qs[QK];    // Quantized values
};

// ─────────────────────────────────────────────────────────────────────────────
// f16 conversion helpers
// ─────────────────────────────────────────────────────────────────────────────
inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint16_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint16_t frac = (x >> 13) & 0x03FF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (static_cast<uint16_t>(exp) << 10) | frac;
}

inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x03FF;
    if (exp == 0) {
        if (frac == 0) { float r; uint32_t v = sign; std::memcpy(&r, &v, 4); return r; }
        // Denormalized
        exp = 1;
        while (!(frac & 0x0400)) { frac <<= 1; exp--; }
        frac &= 0x03FF;
    } else if (exp == 31) {
        uint32_t v = sign | 0x7F800000 | (frac << 13);
        float r; std::memcpy(&r, &v, 4); return r;
    }
    uint32_t v = sign | ((exp + 127 - 15) << 23) | (frac << 13);
    float r; std::memcpy(&r, &v, 4);
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Quantize / Dequantize Q4_0
// ─────────────────────────────────────────────────────────────────────────────
inline void quantize_q4_0(const float* src, BlockQ4_0* dst, size_t n) {
    size_t nb = n / QK;
    for (size_t b = 0; b < nb; ++b) {
        float amax = 0;
        for (size_t i = 0; i < QK; ++i) {
            float av = std::fabs(src[b * QK + i]);
            if (av > amax) amax = av;
        }
        float d = amax / 7.0f; // 4-bit signed: -8..7
        dst[b].d = f32_to_f16(d);
        float id = (d > 1e-12f) ? 1.0f / d : 0.0f;
        for (size_t i = 0; i < QK / 2; ++i) {
            float x0 = src[b * QK + 2 * i] * id;
            float x1 = src[b * QK + 2 * i + 1] * id;
            int8_t q0 = static_cast<int8_t>(std::max(-8.0f, std::min(7.0f, std::round(x0))));
            int8_t q1 = static_cast<int8_t>(std::max(-8.0f, std::min(7.0f, std::round(x1))));
            dst[b].qs[i] = static_cast<uint8_t>((q0 + 8) | ((q1 + 8) << 4));
        }
    }
}

inline void dequantize_q4_0(const BlockQ4_0* src, float* dst, size_t n) {
    size_t nb = n / QK;
    for (size_t b = 0; b < nb; ++b) {
        float d = f16_to_f32(src[b].d);
        for (size_t i = 0; i < QK / 2; ++i) {
            int8_t q0 = static_cast<int8_t>(src[b].qs[i] & 0x0F) - 8;
            int8_t q1 = static_cast<int8_t>(src[b].qs[i] >> 4) - 8;
            dst[b * QK + 2 * i]     = static_cast<float>(q0) * d;
            dst[b * QK + 2 * i + 1] = static_cast<float>(q1) * d;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Quantize / Dequantize Q8_0
// ─────────────────────────────────────────────────────────────────────────────
inline void quantize_q8_0(const float* src, BlockQ8_0* dst, size_t n) {
    size_t nb = n / QK;
    for (size_t b = 0; b < nb; ++b) {
        float amax = 0;
        for (size_t i = 0; i < QK; ++i) {
            float av = std::fabs(src[b * QK + i]);
            if (av > amax) amax = av;
        }
        float d = amax / 127.0f;
        dst[b].d = f32_to_f16(d);
        float id = (d > 1e-12f) ? 1.0f / d : 0.0f;
        for (size_t i = 0; i < QK; ++i) {
            dst[b].qs[i] = static_cast<int8_t>(
                std::max(-128.0f, std::min(127.0f, std::round(src[b * QK + i] * id))));
        }
    }
}

inline void dequantize_q8_0(const BlockQ8_0* src, float* dst, size_t n) {
    size_t nb = n / QK;
    for (size_t b = 0; b < nb; ++b) {
        float d = f16_to_f32(src[b].d);
        for (size_t i = 0; i < QK; ++i) {
            dst[b * QK + i] = static_cast<float>(src[b].qs[i]) * d;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Context — Arena-based memory allocator for tensors
// ─────────────────────────────────────────────────────────────────────────────
class Context {
public:
    explicit Context(size_t mem_size = 1024 * 1024)
        : arena_(mem_size) {}

    void* alloc(size_t size) { return arena_.alloc(size); }
    void reset() { arena_.reset(); }
    size_t used() const { return arena_.used(); }
    size_t capacity() const { return arena_.capacity(); }

private:
    Arena arena_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Tensor — N-dimensional tensor
// ─────────────────────────────────────────────────────────────────────────────
static constexpr size_t MAX_DIMS = 4;

class Tensor {
public:
    Tensor() : data_(nullptr), grad_(nullptr), dtype_(DType::F32), ne_{0,0,0,0} {}

    // Create from context (arena-allocated)
    static Tensor create(Context& ctx, DType dtype,
                         size_t ne0, size_t ne1 = 1, size_t ne2 = 1, size_t ne3 = 1) {
        Tensor t;
        t.dtype_ = dtype;
        t.ne_[0] = ne0; t.ne_[1] = ne1; t.ne_[2] = ne2; t.ne_[3] = ne3;
        t.compute_strides();
        size_t bytes = t.nbytes();
        t.data_ = static_cast<uint8_t*>(ctx.alloc(bytes));
        if (t.data_) std::memset(t.data_, 0, bytes);
        return t;
    }

    // Create from vector (heap-allocated)
    static Tensor from_float(const std::vector<float>& data, size_t ne0, size_t ne1 = 1) {
        Tensor t;
        t.dtype_ = DType::F32;
        t.ne_[0] = ne0; t.ne_[1] = ne1; t.ne_[2] = 1; t.ne_[3] = 1;
        t.compute_strides();
        t.owned_data_.resize(t.nbytes());
        t.data_ = t.owned_data_.data();
        size_t n = std::min(data.size(), t.nelem());
        std::memcpy(t.data_, data.data(), n * sizeof(float));
        return t;
    }

    // Element access (f32)
    float& f32(size_t i) {
        assert(dtype_ == DType::F32 && i < nelem());
        return reinterpret_cast<float*>(data_)[i];
    }
    float f32(size_t i) const {
        assert(dtype_ == DType::F32 && i < nelem());
        return reinterpret_cast<const float*>(data_)[i];
    }

    float& f32(size_t r, size_t c) { return f32(r * ne_[0] + c); }
    float f32(size_t r, size_t c) const { return f32(r * ne_[0] + c); }

    // Gradient access
    float& grad_f32(size_t i) {
        assert(grad_ && i < nelem());
        return reinterpret_cast<float*>(grad_)[i];
    }

    void alloc_grad(Context& ctx) {
        if (!grad_) {
            size_t bytes = nelem() * sizeof(float);
            grad_ = static_cast<uint8_t*>(ctx.alloc(bytes));
            if (grad_) std::memset(grad_, 0, bytes);
        }
    }

    // Properties
    size_t ne(size_t d) const { return ne_[d]; }
    size_t nelem() const { return ne_[0] * ne_[1] * ne_[2] * ne_[3]; }
    size_t nbytes() const {
        if (dtype_ == DType::F32) return nelem() * 4;
        if (dtype_ == DType::F16) return nelem() * 2;
        if (dtype_ == DType::Q4_0) return (nelem() / QK) * sizeof(BlockQ4_0);
        if (dtype_ == DType::Q8_0) return (nelem() / QK) * sizeof(BlockQ8_0);
        if (dtype_ == DType::I32) return nelem() * 4;
        return nelem() * 4;
    }
    DType dtype() const { return dtype_; }
    uint8_t* data() { return data_; }
    const uint8_t* data() const { return data_; }
    bool valid() const { return data_ != nullptr; }

    std::string shape_str() const {
        std::ostringstream os;
        os << ne_[0];
        if (ne_[1] > 1) os << "x" << ne_[1];
        if (ne_[2] > 1) os << "x" << ne_[2];
        if (ne_[3] > 1) os << "x" << ne_[3];
        return os.str();
    }

private:
    uint8_t* data_;
    uint8_t* grad_;
    std::vector<uint8_t> owned_data_;
    DType dtype_;
    std::array<size_t, MAX_DIMS> ne_;
    std::array<size_t, MAX_DIMS> nb_; // Byte strides

    void compute_strides() {
        nb_[0] = (dtype_ == DType::F32) ? 4 : (dtype_ == DType::F16) ? 2 : 4;
        nb_[1] = nb_[0] * ne_[0];
        nb_[2] = nb_[1] * ne_[1];
        nb_[3] = nb_[2] * ne_[2];
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Computation Graph Operations
// ─────────────────────────────────────────────────────────────────────────────
enum class Op : uint8_t {
    NONE = 0, ADD, SUB, MUL, DIV,
    MATMUL, RELU, SIGMOID, TANH, SOFTMAX,
    SUM, MEAN, NORM, SCALE,
    OP_COUNT
};

struct CGNode {
    uint32_t id;
    Op op;
    uint32_t src0;
    uint32_t src1;
    Tensor result;
    bool has_grad;

    CGNode() : id(0), op(Op::NONE), src0(0), src1(0), has_grad(false) {}
};

class CGraph {
public:
    explicit CGraph(Context& ctx) : ctx_(ctx), next_id_(1) {}

    // Add input tensor
    uint32_t input(Tensor t) {
        CGNode n;
        n.id = next_id_++;
        n.op = Op::NONE;
        n.result = t;
        nodes_[n.id] = n;
        return n.id;
    }

    // Binary ops
    uint32_t add(uint32_t a, uint32_t b) { return binary_op(Op::ADD, a, b); }
    uint32_t sub(uint32_t a, uint32_t b) { return binary_op(Op::SUB, a, b); }
    uint32_t mul(uint32_t a, uint32_t b) { return binary_op(Op::MUL, a, b); }

    // MatMul
    uint32_t matmul(uint32_t a, uint32_t b) {
        auto& na = nodes_[a]; auto& nb = nodes_[b];
        size_t M = na.result.ne(1), K = na.result.ne(0), N = nb.result.ne(0);
        assert(K == nb.result.ne(1));
        Tensor out = Tensor::create(ctx_, DType::F32, N, M);
        CGNode n;
        n.id = next_id_++;
        n.op = Op::MATMUL;
        n.src0 = a; n.src1 = b;
        n.result = out;
        nodes_[n.id] = n;
        order_.push_back(n.id);
        return n.id;
    }

    // Unary ops
    uint32_t relu(uint32_t a) { return unary_op(Op::RELU, a); }
    uint32_t sigmoid(uint32_t a) { return unary_op(Op::SIGMOID, a); }
    uint32_t tanh_op(uint32_t a) { return unary_op(Op::TANH, a); }

    // Forward pass
    void forward() {
        for (auto id : order_) {
            auto& n = nodes_[id];
            switch (n.op) {
                case Op::ADD: exec_binary(n, [](float a, float b) { return a + b; }); break;
                case Op::SUB: exec_binary(n, [](float a, float b) { return a - b; }); break;
                case Op::MUL: exec_binary(n, [](float a, float b) { return a * b; }); break;
                case Op::RELU: exec_unary(n, [](float a) { return a > 0 ? a : 0.0f; }); break;
                case Op::SIGMOID: exec_unary(n, [](float a) { return 1.0f / (1.0f + std::exp(-a)); }); break;
                case Op::TANH: exec_unary(n, [](float a) { return std::tanh(a); }); break;
                case Op::MATMUL: exec_matmul(n); break;
                default: break;
            }
        }
    }

    // Get result tensor
    Tensor& result(uint32_t id) { return nodes_[id].result; }

    size_t size() const { return nodes_.size(); }

private:
    Context& ctx_;
    uint32_t next_id_;
    std::unordered_map<uint32_t, CGNode> nodes_;
    std::vector<uint32_t> order_;

    uint32_t binary_op(Op op, uint32_t a, uint32_t b) {
        auto& na = nodes_[a];
        Tensor out = Tensor::create(ctx_, DType::F32, na.result.ne(0), na.result.ne(1));
        CGNode n;
        n.id = next_id_++;
        n.op = op;
        n.src0 = a; n.src1 = b;
        n.result = out;
        nodes_[n.id] = n;
        order_.push_back(n.id);
        return n.id;
    }

    uint32_t unary_op(Op op, uint32_t a) {
        auto& na = nodes_[a];
        Tensor out = Tensor::create(ctx_, DType::F32, na.result.ne(0), na.result.ne(1));
        CGNode n;
        n.id = next_id_++;
        n.op = op;
        n.src0 = a; n.src1 = 0;
        n.result = out;
        nodes_[n.id] = n;
        order_.push_back(n.id);
        return n.id;
    }

    void exec_binary(CGNode& n, std::function<float(float, float)> fn) {
        auto& a = nodes_[n.src0].result;
        auto& b = nodes_[n.src1].result;
        size_t ne = a.nelem();
        for (size_t i = 0; i < ne; ++i) {
            n.result.f32(i) = fn(a.f32(i), b.f32(i % b.nelem()));
        }
    }

    void exec_unary(CGNode& n, std::function<float(float)> fn) {
        auto& a = nodes_[n.src0].result;
        size_t ne = a.nelem();
        for (size_t i = 0; i < ne; ++i) {
            n.result.f32(i) = fn(a.f32(i));
        }
    }

    void exec_matmul(CGNode& n) {
        auto& a = nodes_[n.src0].result;
        auto& b = nodes_[n.src1].result;
        size_t M = a.ne(1), K = a.ne(0), N = b.ne(0);
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0;
                for (size_t k = 0; k < K; ++k) {
                    sum += a.f32(i, k) * b.f32(k, j);
                }
                n.result.f32(i, j) = sum;
            }
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// ADAM Optimizer
// ─────────────────────────────────────────────────────────────────────────────
class Adam {
public:
    struct Params {
        float lr;
        float beta1;
        float beta2;
        float eps;
        float weight_decay;

        Params() : lr(1e-3f), beta1(0.9f), beta2(0.999f),
                   eps(1e-8f), weight_decay(0.0f) {}
    };

    explicit Adam(const Params& p = Params()) : params_(p), t_(0) {}

    // Update parameters given gradients
    void step(float* param, const float* grad, size_t n) {
        if (m_.size() < n) {
            m_.resize(n, 0.0f);
            v_.resize(n, 0.0f);
        }
        ++t_;
        float bc1 = 1.0f - std::pow(params_.beta1, static_cast<float>(t_));
        float bc2 = 1.0f - std::pow(params_.beta2, static_cast<float>(t_));

        for (size_t i = 0; i < n; ++i) {
            float g = grad[i] + params_.weight_decay * param[i];
            m_[i] = params_.beta1 * m_[i] + (1.0f - params_.beta1) * g;
            v_[i] = params_.beta2 * v_[i] + (1.0f - params_.beta2) * g * g;
            float m_hat = m_[i] / bc1;
            float v_hat = v_[i] / bc2;
            param[i] -= params_.lr * m_hat / (std::sqrt(v_hat) + params_.eps);
        }
    }

    void reset() { t_ = 0; m_.clear(); v_.clear(); }

private:
    Params params_;
    size_t t_;
    std::vector<float> m_, v_;
};

// ─────────────────────────────────────────────────────────────────────────────
// L-BFGS Optimizer (simplified)
// ─────────────────────────────────────────────────────────────────────────────
class LBFGS {
public:
    struct Params {
        size_t m;       // History size
        float lr;
        size_t max_iter;

        Params() : m(10), lr(1.0f), max_iter(20) {}
    };

    explicit LBFGS(const Params& p = Params()) : params_(p) {}

    // One step of L-BFGS (two-loop recursion)
    void step(float* param, const float* grad, size_t n,
              std::function<float(const float*)> eval_fn) {
        // Store current gradient
        std::vector<float> g(grad, grad + n);
        std::vector<float> q = g;

        // Two-loop recursion
        size_t k = s_history_.size();
        std::vector<float> alpha(k);

        for (int i = static_cast<int>(k) - 1; i >= 0; --i) {
            float rho_i = rho_[i];
            alpha[i] = rho_i * dot(s_history_[i], q);
            for (size_t j = 0; j < n; ++j) {
                q[j] -= alpha[i] * y_history_[i][j];
            }
        }

        // Scale by H0 = (s^T y) / (y^T y) if available
        std::vector<float> r = q;
        if (k > 0) {
            float ys = dot(y_history_.back(), s_history_.back());
            float yy = dot(y_history_.back(), y_history_.back());
            float gamma = ys / (yy + 1e-12f);
            for (auto& x : r) x *= gamma;
        }

        for (size_t i = 0; i < k; ++i) {
            float beta = rho_[i] * dot(y_history_[i], r);
            for (size_t j = 0; j < n; ++j) {
                r[j] += s_history_[i][j] * (alpha[i] - beta);
            }
        }

        // Update: p = p - lr * r
        std::vector<float> s(n);
        for (size_t i = 0; i < n; ++i) {
            float delta = -params_.lr * r[i];
            param[i] += delta;
            s[i] = delta;
        }

        // Store s and y for history
        if (prev_grad_.size() == n) {
            std::vector<float> y(n);
            for (size_t i = 0; i < n; ++i) y[i] = g[i] - prev_grad_[i];
            float sy = dot(s, y);
            if (sy > 1e-12f) {
                if (s_history_.size() >= params_.m) {
                    s_history_.erase(s_history_.begin());
                    y_history_.erase(y_history_.begin());
                    rho_.erase(rho_.begin());
                }
                s_history_.push_back(s);
                y_history_.push_back(y);
                rho_.push_back(1.0f / sy);
            }
        }
        prev_grad_ = g;
    }

private:
    Params params_;
    std::vector<std::vector<float>> s_history_, y_history_;
    std::vector<float> rho_;
    std::vector<float> prev_grad_;

    static float dot(const std::vector<float>& a, const std::vector<float>& b) {
        float sum = 0;
        for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
        return sum;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// SafeTensors metadata loader
//
// The SafeTensors format stores a JSON header at the start of a .safetensors
// file:  [8-byte little-endian header_len] [header_len bytes of UTF-8 JSON]
// The JSON maps tensor names to {"dtype":…, "shape":[…], "data_offsets":[…]}.
//
// This loader reads the header only (no binary weights), providing fast
// tensor metadata inspection without needing to map the full file.
// ─────────────────────────────────────────────────────────────────────────────

// Metadata for a single SafeTensors tensor entry
struct SafeTensorMeta {
    std::string          name;           // tensor name
    std::string          dtype;          // e.g. "F32", "F16", "BF16"
    std::vector<int64_t> shape;          // dimension sizes
    int64_t              data_begin;     // byte offset of data start (relative to data region)
    int64_t              data_end;       // byte offset of data end

    int64_t element_count() const {
        int64_t n = 1;
        for (int64_t s : shape) n *= s;
        return n;
    }
};

// Result of parsing a SafeTensors file header
struct SafeTensorsHeader {
    std::vector<SafeTensorMeta> tensors;  // all tensor entries (ordered by file position)
    std::string                 raw_json; // raw JSON string (for debugging / re-use)

    // Look up a tensor by name.  Returns nullptr if not found.
    const SafeTensorMeta* find(const std::string& name) const {
        for (const auto& t : tensors)
            if (t.name == name) return &t;
        return nullptr;
    }

    size_t size() const { return tensors.size(); }
};

namespace detail {

// Minimal JSON string-value extractor — no external dependencies.
// Finds the value of a JSON string key in a flat JSON object fragment.
// Returns empty string if not found.
inline std::string json_string_value(const std::string& json,
                                     const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos += needle.size();
    // skip whitespace and colon
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t')) ++pos;
    if (pos >= json.size() || json[pos] != '"') return "";
    ++pos;  // skip opening quote
    std::string val;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\') { ++pos; }  // skip escape
        val += json[pos++];
    }
    return val;
}

// Parse a JSON integer array like [1024, 768] into a vector<int64_t>
inline std::vector<int64_t> json_int_array(const std::string& json,
                                            const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return {};
    pos += needle.size();
    while (pos < json.size() && json[pos] != '[') ++pos;
    if (pos >= json.size()) return {};
    ++pos;  // skip '['
    std::vector<int64_t> vals;
    while (pos < json.size() && json[pos] != ']') {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == ',')) ++pos;
        if (pos >= json.size() || json[pos] == ']') break;
        char* end;
        int64_t v = static_cast<int64_t>(std::strtoll(&json[pos], &end, 10));
        if (end == &json[pos]) break;
        vals.push_back(v);
        pos = static_cast<size_t>(end - &json[0]);
    }
    return vals;
}

// Parse [begin, end] offset pair from "data_offsets":[begin,end]
inline std::pair<int64_t,int64_t> json_offset_pair(const std::string& json) {
    auto vec = json_int_array(json, "data_offsets");
    if (vec.size() >= 2) return {vec[0], vec[1]};
    if (vec.size() == 1) return {vec[0], vec[0]};
    return {0, 0};
}

// Split top-level JSON object keys: iterate over "key":{...} pairs in order.
// Calls callback(key, value_fragment) for each.  Not fully general but
// sufficient for the flat SafeTensors header format.
inline void json_foreach_key(const std::string& json,
                              const std::function<void(const std::string&,
                                                       const std::string&)>& cb) {
    size_t pos = 0;
    // skip leading '{'
    while (pos < json.size() && json[pos] != '{') ++pos;
    if (pos >= json.size()) return;
    ++pos;

    while (pos < json.size()) {
        // find key
        while (pos < json.size() && json[pos] != '"' && json[pos] != '}') ++pos;
        if (pos >= json.size() || json[pos] == '}') break;
        ++pos;  // skip '"'
        std::string key;
        while (pos < json.size() && json[pos] != '"') key += json[pos++];
        if (pos < json.size()) ++pos;  // skip closing '"'
        // skip ':' and whitespace
        while (pos < json.size() && (json[pos] == ':' || json[pos] == ' ')) ++pos;
        if (pos >= json.size()) break;
        // collect value (object or string or number)
        std::string val;
        if (json[pos] == '{') {
            int depth = 0;
            size_t start = pos;
            while (pos < json.size()) {
                if (json[pos] == '{') ++depth;
                else if (json[pos] == '}') { --depth; if (depth == 0) { ++pos; break; } }
                ++pos;
            }
            val = json.substr(start, pos - start);
        } else {
            size_t start = pos;
            while (pos < json.size() && json[pos] != ',' && json[pos] != '}') ++pos;
            val = json.substr(start, pos - start);
        }
        cb(key, val);
        // skip comma
        while (pos < json.size() && (json[pos] == ',' || json[pos] == ' ')) ++pos;
    }
}

} // namespace detail

// Parse SafeTensors header JSON string into a SafeTensorsHeader.
// The JSON is the raw header extracted from a .safetensors file.
inline SafeTensorsHeader parse_safetensors_json(const std::string& json) {
    SafeTensorsHeader hdr;
    hdr.raw_json = json;
    detail::json_foreach_key(json, [&](const std::string& key, const std::string& val) {
        if (key == "__metadata__") return;  // skip metadata dict
        SafeTensorMeta m;
        m.name  = key;
        m.dtype = detail::json_string_value(val, "dtype");
        m.shape = detail::json_int_array(val, "shape");
        auto off = detail::json_offset_pair(val);
        m.data_begin = off.first;
        m.data_end   = off.second;
        hdr.tensors.push_back(m);
    });
    return hdr;
}

// Read and parse the SafeTensors header from a file path.
// Returns an empty header (with empty raw_json) on read error.
// Only the header bytes are read — the weight data is not loaded.
inline SafeTensorsHeader load_safetensors_metadata(const std::string& path) {
    // We use C stdio to stay dependency-free
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return SafeTensorsHeader{};

    // Read 8-byte little-endian header length
    uint8_t len_bytes[8] = {};
    if (std::fread(len_bytes, 1, 8, f) != 8) { std::fclose(f); return SafeTensorsHeader{}; }
    uint64_t hlen = 0;
    for (int i = 0; i < 8; ++i)
        hlen |= (static_cast<uint64_t>(len_bytes[i]) << (8 * i));

    if (hlen == 0 || hlen > 64 * 1024 * 1024ULL) { // sanity cap: 64 MiB header
        std::fclose(f);
        return SafeTensorsHeader{};
    }

    std::string json(static_cast<size_t>(hlen), '\0');
    if (std::fread(&json[0], 1, static_cast<size_t>(hlen), f) != static_cast<size_t>(hlen)) {
        std::fclose(f);
        return SafeTensorsHeader{};
    }
    std::fclose(f);
    return parse_safetensors_json(json);
}

}} // namespace cog::gml

#endif // COG_GML_HPP
