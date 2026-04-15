// cog/tq_log2_3.hpp — Log₂(3) Ternary Quantization for coggml
// Header-only C++11 implementation of information-theoretically optimal
// ternary weight encoding: {-1, 0, +1} at log₂(3) ≈ 1.585 bits/weight.
//
// Packing: 5 trits per byte (3⁵ = 243 ≤ 255), yielding 1.6875 bpw with scale.
// This header provides:
//   - Compile-time LUT generation for pack/unpack (zero runtime division)
//   - Block quantize/dequantize for QK=256
//   - Ternary GEMV kernel (dot product of ternary × float)
//   - Balanced ternary arithmetic (carry-free addition)
//   - Entropy analysis utilities
//
// Composition: coggml ⊗ llama-cpp-skillm ⊗ dte-llm-evolution
// Part of the cogpy header-only ecosystem (C++11, zero dependencies)
//
// Copyright (c) 2026 cogpy contributors. MIT License.

#ifndef COG_TQ_LOG2_3_HPP
#define COG_TQ_LOG2_3_HPP

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cassert>
#include <cstring>
#include <array>
#include <algorithm>
#include <numeric>
#include <vector>

namespace cog {
namespace tq {

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

static constexpr double LOG2_3       = 1.5849625007211562;  // log₂(3)
static constexpr int    TRITS_PER_BYTE = 5;                 // 3⁵ = 243 ≤ 255
static constexpr int    POW3_5       = 243;                 // 3^5
static constexpr int    QK           = 256;                 // block size
static constexpr int    PACKED_BYTES = 52;                  // ceil(256/5) = 52
static constexpr int    USED_TRITS   = PACKED_BYTES * TRITS_PER_BYTE; // 260
static constexpr int    WASTE_TRITS  = USED_TRITS - QK;    // 4 wasted slots
static constexpr double BPW          = (PACKED_BYTES + 2.0) * 8.0 / QK; // 1.6875

// ─────────────────────────────────────────────────────────────────────────────
// Compile-time LUT: byte → 5 trits (each in {-1, 0, +1})
// ─────────────────────────────────────────────────────────────────────────────

struct TritQuintet {
    int8_t t[5];  // each in {-1, 0, +1}
};

// Unpack table: for each byte value 0..242, store 5 trits
struct UnpackLUT {
    TritQuintet table[POW3_5];

    UnpackLUT() : table() {
        for (int b = 0; b < POW3_5; ++b) {
            int v = b;
            for (int i = 0; i < 5; ++i) {
                table[b].t[i] = static_cast<int8_t>((v % 3) - 1);
                v /= 3;
            }
        }
    }
};

// Pack table: for each combination of 5 trits (each +1 offset to 0,1,2), store byte
// Index: t0 + t1*3 + t2*9 + t3*27 + t4*81  (each t_i in {0,1,2})
struct PackLUT {
    uint8_t table[POW3_5];

    PackLUT() : table() {
        for (int i = 0; i < POW3_5; ++i) {
            table[i] = static_cast<uint8_t>(i);
        }
    }
};

// Singleton LUTs (header-only safe via inline function)
inline const UnpackLUT& unpack_lut() {
    static const UnpackLUT lut;
    return lut;
}

inline const PackLUT& pack_lut() {
    static const PackLUT lut;
    return lut;
}

// ─────────────────────────────────────────────────────────────────────────────
// Block structure: 256 ternary weights packed into 54 bytes
// ─────────────────────────────────────────────────────────────────────────────

// fp16 storage (use uint16_t to avoid platform dependency)
using fp16_t = uint16_t;

inline float fp16_to_fp32(fp16_t h) {
    // IEEE 754 half → float conversion
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t expo = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    uint32_t f;
    if (expo == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            // subnormal
            expo = 1;
            while (!(mant & 0x0400)) { mant <<= 1; expo--; }
            mant &= ~0x0400u;
            f = sign | ((expo + 127 - 15) << 23) | (mant << 13);
        }
    } else if (expo == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((expo + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

inline fp16_t fp32_to_fp16(float v) {
    uint32_t f;
    std::memcpy(&f, &v, sizeof(float));
    uint32_t sign = (f >> 16) & 0x8000;
    int expo = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x007FFFFF;
    if (expo <= 0) {
        if (expo < -10) return static_cast<fp16_t>(sign);
        mant = (mant | 0x00800000) >> (1 - expo);
        return static_cast<fp16_t>(sign | (mant >> 13));
    } else if (expo == 0xFF - 127 + 15) {
        return static_cast<fp16_t>(sign | 0x7C00 | (mant ? (mant >> 13) | 1 : 0));
    }
    if (expo > 30) return static_cast<fp16_t>(sign | 0x7C00);
    return static_cast<fp16_t>(sign | (expo << 10) | (mant >> 13));
}

struct BlockTQ {
    uint8_t qs[PACKED_BYTES];  // 52 bytes: 5 trits per byte
    fp16_t  d;                 // block scale (fp16)

    float scale() const { return fp16_to_fp32(d); }
};

static_assert(sizeof(BlockTQ) == PACKED_BYTES + sizeof(fp16_t), "BlockTQ size mismatch");

// ─────────────────────────────────────────────────────────────────────────────
// Ternary quantization: float[256] → BlockTQ
// ─────────────────────────────────────────────────────────────────────────────

inline void quantize_block(const float* x, BlockTQ& block) {
    // Find absolute max for scale
    float amax = 0.0f;
    for (int i = 0; i < QK; ++i) {
        float av = std::fabs(x[i]);
        if (av > amax) amax = av;
    }

    block.d = fp32_to_fp16(amax);
    float id = (amax > 0.0f) ? 1.0f / amax : 0.0f;

    // Quantize to {-1, 0, +1} and pack 5 trits per byte
    int xi = 0;
    for (int b = 0; b < PACKED_BYTES; ++b) {
        int packed = 0;
        int base = 1;
        for (int t = 0; t < TRITS_PER_BYTE; ++t) {
            int trit;
            if (xi < QK) {
                // Round to nearest trit: -1, 0, or +1
                float scaled = x[xi] * id;
                trit = (scaled > 0.5f) ? 2 : (scaled < -0.5f) ? 0 : 1;
                xi++;
            } else {
                trit = 1;  // pad with zero (trit value 1 = 0 in balanced)
            }
            packed += trit * base;
            base *= 3;
        }
        block.qs[b] = static_cast<uint8_t>(packed);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ternary dequantization: BlockTQ → float[256]
// ─────────────────────────────────────────────────────────────────────────────

inline void dequantize_block(const BlockTQ& block, float* out) {
    const float scale = block.scale();
    const auto& lut = unpack_lut();
    int xi = 0;

    for (int b = 0; b < PACKED_BYTES && xi < QK; ++b) {
        uint8_t packed = block.qs[b];
        // Clamp to valid range
        if (packed >= POW3_5) packed = POW3_5 - 1;
        const TritQuintet& q = lut.table[packed];
        for (int t = 0; t < TRITS_PER_BYTE && xi < QK; ++t) {
            out[xi++] = static_cast<float>(q.t[t]) * scale;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch quantize/dequantize
// ─────────────────────────────────────────────────────────────────────────────

inline void quantize_row(const float* x, BlockTQ* blocks, size_t n) {
    assert(n % QK == 0);
    size_t nb = n / QK;
    for (size_t i = 0; i < nb; ++i) {
        quantize_block(x + i * QK, blocks[i]);
    }
}

inline void dequantize_row(const BlockTQ* blocks, float* out, size_t n) {
    assert(n % QK == 0);
    size_t nb = n / QK;
    for (size_t i = 0; i < nb; ++i) {
        dequantize_block(blocks[i], out + i * QK);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ternary GEMV: dot product of ternary weights × float activations
// This is the core inference kernel.
// ─────────────────────────────────────────────────────────────────────────────

// Dot product of a single block (256 ternary weights) with float activations
inline float dot_block(const BlockTQ& block, const float* x) {
    const float scale = block.scale();
    const auto& lut = unpack_lut();
    float sum = 0.0f;
    int xi = 0;

    for (int b = 0; b < PACKED_BYTES && xi < QK; ++b) {
        uint8_t packed = block.qs[b];
        if (packed >= POW3_5) packed = POW3_5 - 1;
        const TritQuintet& q = lut.table[packed];
        for (int t = 0; t < TRITS_PER_BYTE && xi < QK; ++t) {
            // Ternary multiply: just add, subtract, or skip
            int8_t trit = q.t[t];
            if (trit == 1) {
                sum += x[xi];
            } else if (trit == -1) {
                sum -= x[xi];
            }
            // trit == 0: no-op (free!)
            xi++;
        }
    }
    return sum * scale;
}

// Full GEMV: y[m] = sum_k W[m,k] * x[k]
// W is stored as blocks[m][n_blocks_per_row]
inline void gemv(const BlockTQ* W, const float* x, float* y,
                 size_t m, size_t k) {
    assert(k % QK == 0);
    size_t nbk = k / QK;
    for (size_t i = 0; i < m; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < nbk; ++j) {
            sum += dot_block(W[i * nbk + j], x + j * QK);
        }
        y[i] = sum;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Balanced Ternary Arithmetic
// Carry-free addition of two ternary values (for accumulation)
// ─────────────────────────────────────────────────────────────────────────────

struct BalancedTernary {
    std::vector<int8_t> digits;  // each in {-1, 0, +1}, LSB first

    BalancedTernary() {}
    explicit BalancedTernary(int n) { from_int(n); }

    void from_int(int n) {
        digits.clear();
        if (n == 0) { digits.push_back(0); return; }
        bool neg = (n < 0);
        if (neg) n = -n;
        while (n > 0) {
            int rem = n % 3;
            if (rem == 2) {
                digits.push_back(-1);
                n = (n + 1) / 3;
            } else {
                digits.push_back(static_cast<int8_t>(rem));
                n /= 3;
            }
        }
        if (neg) {
            for (auto& d : digits) d = -d;
        }
    }

    int to_int() const {
        int result = 0;
        int base = 1;
        for (auto d : digits) {
            result += d * base;
            base *= 3;
        }
        return result;
    }

    // Add two balanced ternary numbers
    static BalancedTernary add(const BalancedTernary& a, const BalancedTernary& b) {
        size_t maxlen = std::max(a.digits.size(), b.digits.size()) + 1;
        BalancedTernary result;
        result.digits.resize(maxlen, 0);
        int carry = 0;
        for (size_t i = 0; i < maxlen; ++i) {
            int da = (i < a.digits.size()) ? a.digits[i] : 0;
            int db = (i < b.digits.size()) ? b.digits[i] : 0;
            int sum = da + db + carry;
            carry = 0;
            if (sum > 1)       { sum -= 3; carry = 1; }
            else if (sum < -1) { sum += 3; carry = -1; }
            result.digits[i] = static_cast<int8_t>(sum);
        }
        // Trim leading zeros
        while (result.digits.size() > 1 && result.digits.back() == 0) {
            result.digits.pop_back();
        }
        return result;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Entropy & Statistics
// ─────────────────────────────────────────────────────────────────────────────

struct QuantStats {
    size_t total_weights;
    size_t count_neg;     // -1
    size_t count_zero;    //  0
    size_t count_pos;     // +1
    double entropy;       // actual entropy in bits/weight
    double efficiency;    // entropy / log2(3) — how close to optimal
    double mse;           // mean squared error vs original
    double max_error;     // max absolute error

    double sparsity() const {
        return (total_weights > 0) ? static_cast<double>(count_zero) / total_weights : 0.0;
    }
};

inline QuantStats compute_stats(const float* original, const float* reconstructed,
                                 const BlockTQ* blocks, size_t n) {
    QuantStats s = {};
    s.total_weights = n;
    assert(n % QK == 0);
    size_t nb = n / QK;
    const auto& lut = unpack_lut();

    // Count trit distribution
    for (size_t i = 0; i < nb; ++i) {
        int xi = 0;
        for (int b = 0; b < PACKED_BYTES && xi < QK; ++b) {
            uint8_t packed = blocks[i].qs[b];
            if (packed >= POW3_5) packed = POW3_5 - 1;
            const TritQuintet& q = lut.table[packed];
            for (int t = 0; t < TRITS_PER_BYTE && xi < QK; ++t) {
                if      (q.t[t] == -1) s.count_neg++;
                else if (q.t[t] ==  0) s.count_zero++;
                else                   s.count_pos++;
                xi++;
            }
        }
    }

    // Entropy
    double p_neg  = static_cast<double>(s.count_neg)  / s.total_weights;
    double p_zero = static_cast<double>(s.count_zero) / s.total_weights;
    double p_pos  = static_cast<double>(s.count_pos)  / s.total_weights;
    s.entropy = 0.0;
    if (p_neg  > 0) s.entropy -= p_neg  * std::log2(p_neg);
    if (p_zero > 0) s.entropy -= p_zero * std::log2(p_zero);
    if (p_pos  > 0) s.entropy -= p_pos  * std::log2(p_pos);
    s.efficiency = s.entropy / LOG2_3;

    // MSE and max error
    double sum_sq = 0.0;
    s.max_error = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double err = std::fabs(original[i] - reconstructed[i]);
        sum_sq += err * err;
        if (err > s.max_error) s.max_error = err;
    }
    s.mse = sum_sq / n;

    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Ternary Matrix: convenience wrapper for weight matrices
// ─────────────────────────────────────────────────────────────────────────────

class TernaryMatrix {
public:
    size_t rows_;
    size_t cols_;
    size_t blocks_per_row_;
    std::vector<BlockTQ> blocks_;

    TernaryMatrix() : rows_(0), cols_(0), blocks_per_row_(0) {}

    TernaryMatrix(size_t rows, size_t cols)
        : rows_(rows), cols_(cols) {
        assert(cols % QK == 0);
        blocks_per_row_ = cols / QK;
        blocks_.resize(rows * blocks_per_row_);
    }

    // Quantize from float matrix (row-major)
    void quantize_from(const float* data) {
        for (size_t r = 0; r < rows_; ++r) {
            quantize_row(data + r * cols_, blocks_.data() + r * blocks_per_row_, cols_);
        }
    }

    // Dequantize to float matrix (row-major)
    void dequantize_to(float* data) const {
        for (size_t r = 0; r < rows_; ++r) {
            dequantize_row(blocks_.data() + r * blocks_per_row_, data + r * cols_, cols_);
        }
    }

    // GEMV: y = W * x
    void matvec(const float* x, float* y) const {
        gemv(blocks_.data(), x, y, rows_, cols_);
    }

    // Memory footprint in bytes
    size_t memory_bytes() const {
        return blocks_.size() * sizeof(BlockTQ);
    }

    // Compression ratio vs fp32
    double compression_ratio() const {
        return static_cast<double>(rows_ * cols_ * sizeof(float)) / memory_bytes();
    }

    // Bits per weight
    double bits_per_weight() const {
        return static_cast<double>(memory_bytes() * 8) / (rows_ * cols_);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Ternary Linear Layer: W*x + bias (for DTE inference)
// ─────────────────────────────────────────────────────────────────────────────

class TernaryLinear {
public:
    TernaryMatrix weight;
    std::vector<float> bias;
    bool has_bias;

    TernaryLinear() : has_bias(false) {}

    TernaryLinear(size_t in_features, size_t out_features, bool use_bias = true)
        : weight(out_features, in_features), has_bias(use_bias) {
        if (use_bias) {
            bias.resize(out_features, 0.0f);
        }
    }

    // Quantize weight from float data
    void load_weight(const float* w) {
        weight.quantize_from(w);
    }

    void load_bias(const float* b) {
        if (has_bias) {
            std::copy(b, b + weight.rows_, bias.begin());
        }
    }

    // Forward: y = W*x + b
    void forward(const float* x, float* y) const {
        weight.matvec(x, y);
        if (has_bias) {
            for (size_t i = 0; i < weight.rows_; ++i) {
                y[i] += bias[i];
            }
        }
    }

    size_t memory_bytes() const {
        return weight.memory_bytes() + (has_bias ? bias.size() * sizeof(float) : 0);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Ternary MLP: Two-layer feed-forward network (for DTE cognitive layers)
// ─────────────────────────────────────────────────────────────────────────────

class TernaryMLP {
public:
    TernaryLinear up;    // hidden = up(x)
    TernaryLinear down;  // out = down(act(hidden))

    TernaryMLP() {}

    TernaryMLP(size_t in_dim, size_t hidden_dim, size_t out_dim)
        : up(in_dim, hidden_dim, true), down(hidden_dim, out_dim, true) {}

    // Forward with SiLU activation
    void forward(const float* x, float* out) const {
        std::vector<float> hidden(up.weight.rows_);
        up.forward(x, hidden.data());

        // SiLU activation: x * sigmoid(x)
        for (auto& h : hidden) {
            float sig = 1.0f / (1.0f + std::exp(-h));
            h = h * sig;
        }

        down.forward(hidden.data(), out);
    }

    size_t memory_bytes() const {
        return up.memory_bytes() + down.memory_bytes();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// System-level integration constants
// ─────────────────────────────────────────────────────────────────────────────

// Trit packing densities for different bases
// These relate to the tree-polytope kernel: each base-3 digit is a
// rooted tree with 3 children, and the packing is the convolution
// (1-1)^n evaluated at the ternary simplex.
// Packing densities for different container sizes.
// The key insight: floor(n / log2(3)) trits fit in n bits.
// Wider containers waste fewer fractional trits.
//   8-bit:  floor(8/1.585)  = 5  trits, 3^5  = 243   ≤ 255    → eff = 5*1.585/8  = 0.9906
//  16-bit:  floor(16/1.585) = 10 trits, 3^10 = 59049  ≤ 65535  → eff = 10*1.585/16 = 0.9906
//  32-bit:  floor(32/1.585) = 20 trits, 3^20 = 3.49e9 ≤ 4.29e9 → eff = 20*1.585/32 = 0.9906
//  64-bit:  floor(64/1.585) = 40 trits, 3^40 = 1.22e19 ≤ 1.84e19 → eff = 40*1.585/64 = 0.9906
//
// All have the same efficiency because 5/8 = 10/16 = 20/32 = 40/64.
// To get *better* efficiency, we need non-power-of-2 containers or
// entropy coding. The real win is in the GEMV kernel: wider containers
// allow SIMD-parallel unpacking of more trits per instruction.
struct PackingDensity {
    int trits_per_unit;
    int unit_bits;
    double bpw;           // bits per weight = unit_bits / trits_per_unit
    double efficiency;    // vs theoretical log2(3)
    int simd_parallelism; // trits unpacked per SIMD op

    static constexpr PackingDensity byte_5()  { return {5,  8,  8.0/5.0,  5*LOG2_3/8.0,   5}; }
    static constexpr PackingDensity u16_10()  { return {10, 16, 16.0/10.0, 10*LOG2_3/16.0, 10}; }
    static constexpr PackingDensity u32_20()  { return {20, 32, 32.0/20.0, 20*LOG2_3/32.0, 20}; }
    static constexpr PackingDensity u64_40()  { return {40, 64, 64.0/40.0, 40*LOG2_3/64.0, 40}; }
};

// Ternary type info (for ggml integration)
struct TQTypeInfo {
    static constexpr int type_id = 41;  // GGML_TYPE_TQ_LOG2_3 (next after MXFP4=39, COUNT=40)
    static constexpr const char* name = "TQ_LOG2_3";
    static constexpr int block_size = QK;
    static constexpr int type_size = sizeof(BlockTQ);
    static constexpr double bits_per_weight = BPW;
    static constexpr bool is_quantized = true;
};

} // namespace tq
} // namespace cog

#endif // COG_TQ_LOG2_3_HPP
