// cog/gml/facs_tensor.hpp — FACS Tensor Operations for MetaHuman DNA
// Quantized tensor operations for efficient AU activation computation
// Header-only, C++11, zero external dependencies
// SPDX-License-Identifier: MIT
//
// Provides SIMD-friendly tensor operations for the expression pipeline:
// - Batch hormone-to-AU mapping via matrix multiply
// - Quantized AU storage (Q8_0 format)
// - Smoothing via exponential moving average
//
#ifndef COG_GML_FACS_TENSOR_HPP
#define COG_GML_FACS_TENSOR_HPP

#include "../core/core.hpp"
#include <cstdint>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>

namespace cog { namespace gml {

// ─────────────────────────────────────────────────────────────────────────────
// Constants matching meta-echo-dna specification
// ─────────────────────────────────────────────────────────────────────────────
static constexpr size_t NUM_HORMONES = 10;
static constexpr size_t NUM_AUS      = 20;
static constexpr size_t NUM_TARGETS  = 20; // MetaHuman CTRL_ morph targets

// ─────────────────────────────────────────────────────────────────────────────
// Q8_0 Quantized Block — 32 values quantized to int8 with shared scale
// ─────────────────────────────────────────────────────────────────────────────
struct Q8Block {
    float scale;
    int8_t data[32];

    Q8Block() : scale(0.0f) {
        for (int i = 0; i < 32; ++i) data[i] = 0;
    }
};

// Quantize a float array to Q8 blocks
inline std::vector<Q8Block> quantize_q8(const float* src, size_t n) {
    size_t nblocks = (n + 31) / 32;
    std::vector<Q8Block> blocks(nblocks);
    for (size_t b = 0; b < nblocks; ++b) {
        float max_abs = 0.0f;
        size_t start = b * 32;
        size_t end = std::min(start + 32, n);
        for (size_t i = start; i < end; ++i) {
            max_abs = std::max(max_abs, std::abs(src[i]));
        }
        blocks[b].scale = (max_abs > 0.0f) ? max_abs / 127.0f : 0.0f;
        float inv_scale = (blocks[b].scale > 0.0f) ? 1.0f / blocks[b].scale : 0.0f;
        for (size_t i = start; i < end; ++i) {
            int val = static_cast<int>(std::round(src[i] * inv_scale));
            blocks[b].data[i - start] = static_cast<int8_t>(
                std::max(-127, std::min(127, val)));
        }
    }
    return blocks;
}

// Dequantize Q8 blocks back to float
inline void dequantize_q8(const std::vector<Q8Block>& blocks, float* dst, size_t n) {
    for (size_t b = 0; b < blocks.size(); ++b) {
        size_t start = b * 32;
        size_t end = std::min(start + 32, n);
        for (size_t i = start; i < end; ++i) {
            dst[i] = blocks[b].data[i - start] * blocks[b].scale;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HormoneToAUMatrix — Sparse mapping matrix [NUM_AUS x NUM_HORMONES]
// ─────────────────────────────────────────────────────────────────────────────
class HormoneToAUMatrix {
public:
    HormoneToAUMatrix() {
        mat_.fill(0.0f);
        // Populate from meta-echo-dna endocrine-expression mapping
        // Row = AU index, Col = Hormone index
        set(2, 0, 0.8f);  // AU4  ← Cortisol
        set(0, 0, 0.5f);  // AU1  ← Cortisol
        set(10, 0, 0.4f); // AU15 ← Cortisol
        set(8, 1, 0.9f);  // AU12 ← DopaminePhasic
        set(4, 1, 0.7f);  // AU6  ← DopaminePhasic
        set(8, 2, 0.3f);  // AU12 ← DopamineTonic
        set(4, 3, 0.4f);  // AU6  ← Serotonin
        set(8, 3, 0.3f);  // AU12 ← Serotonin
        set(3, 4, 0.8f);  // AU5  ← Norepinephrine
        set(5, 4, 0.5f);  // AU7  ← Norepinephrine
        set(12, 4, 0.3f); // AU20 ← Norepinephrine
        set(4, 5, 0.6f);  // AU6  ← Oxytocin
        set(8, 5, 0.5f);  // AU12 ← Oxytocin
        set(14, 5, 0.3f); // AU25 ← Oxytocin
        set(17, 6, 0.7f); // AU43 ← Melatonin
        set(5, 6, 0.4f);  // AU7  ← Melatonin
        set(2, 8, 0.5f);  // AU4  ← CytokineIL6
        set(7, 8, 0.4f);  // AU10 ← CytokineIL6
        set(4, 9, 0.5f);  // AU6  ← Anandamide
        set(14, 9, 0.3f); // AU25 ← Anandamide
    }

    // Matrix-vector multiply: au_out = mat * hormone_in
    void multiply(const float hormone_in[NUM_HORMONES],
                  float au_out[NUM_AUS]) const {
        for (size_t r = 0; r < NUM_AUS; ++r) {
            float sum = 0.0f;
            for (size_t c = 0; c < NUM_HORMONES; ++c) {
                sum += mat_[r * NUM_HORMONES + c] * hormone_in[c];
            }
            au_out[r] = sum;
        }
    }

    // Get raw matrix for inspection
    const std::array<float, NUM_AUS * NUM_HORMONES>& raw() const { return mat_; }

private:
    std::array<float, NUM_AUS * NUM_HORMONES> mat_;

    void set(size_t au_idx, size_t hormone_idx, float value) {
        if (au_idx < NUM_AUS && hormone_idx < NUM_HORMONES) {
            mat_[au_idx * NUM_HORMONES + hormone_idx] = value;
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// ExponentialMovingAverage — Temporal smoothing for morph targets
// ─────────────────────────────────────────────────────────────────────────────
class ExponentialMovingAverage {
public:
    explicit ExponentialMovingAverage(size_t dim, float alpha = 0.3f)
        : alpha_(alpha), state_(dim, 0.0f), initialized_(false) {}

    void update(const float* input, float* output, size_t n) {
        if (!initialized_) {
            for (size_t i = 0; i < n && i < state_.size(); ++i) {
                state_[i] = input[i];
            }
            initialized_ = true;
        }
        for (size_t i = 0; i < n && i < state_.size(); ++i) {
            state_[i] = alpha_ * state_[i] + (1.0f - alpha_) * input[i];
            output[i] = state_[i];
        }
    }

    void reset() {
        std::fill(state_.begin(), state_.end(), 0.0f);
        initialized_ = false;
    }

    float alpha() const { return alpha_; }
    void set_alpha(float a) { alpha_ = a; }

private:
    float alpha_;
    std::vector<float> state_;
    bool initialized_;
};

// ─────────────────────────────────────────────────────────────────────────────
// BatchExpressionPipeline — Optimized batch processing
// ─────────────────────────────────────────────────────────────────────────────
class BatchExpressionPipeline {
public:
    BatchExpressionPipeline(float smoothing = 0.3f)
        : h2au_(), ema_(NUM_AUS, smoothing) {}

    // Process one frame: hormone concentrations → smoothed AU activations
    void process(const float hormones[NUM_HORMONES],
                 float au_output[NUM_AUS]) {
        float raw_au[NUM_AUS];
        h2au_.multiply(hormones, raw_au);

        // Clamp to [0, 1]
        for (size_t i = 0; i < NUM_AUS; ++i) {
            raw_au[i] = std::max(0.0f, std::min(1.0f, raw_au[i]));
        }

        // Apply temporal smoothing
        ema_.update(raw_au, au_output, NUM_AUS);
    }

    void reset() { ema_.reset(); }

private:
    HormoneToAUMatrix h2au_;
    ExponentialMovingAverage ema_;
};

}} // namespace cog::gml

#endif // COG_GML_FACS_TENSOR_HPP
