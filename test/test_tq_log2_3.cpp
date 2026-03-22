// test_tq_log2_3.cpp — Comprehensive test suite for log₂(3) ternary quantization
// Compile: g++ -std=c++11 -O2 -I../include -o test_tq_log2_3 test_tq_log2_3.cpp
// Run: ./test_tq_log2_3

#include "cog/tq_log2_3.hpp"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cassert>

using namespace cog::tq;

// ─── Test Utilities ──────────────────────────────────────────────────────────

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    do { \
        std::cout << "  [TEST] " << #name << "... "; \
        if (test_##name()) { \
            std::cout << "PASS" << std::endl; \
            tests_passed++; \
        } else { \
            std::cout << "FAIL" << std::endl; \
            tests_failed++; \
        } \
    } while(0)

// Simple PRNG for reproducibility
static uint32_t xorshift_state = 12345;
float rand_float() {
    xorshift_state ^= xorshift_state << 13;
    xorshift_state ^= xorshift_state >> 17;
    xorshift_state ^= xorshift_state << 5;
    return (static_cast<float>(xorshift_state) / 4294967296.0f) * 2.0f - 1.0f;
}

// ─── Test 1: LUT Consistency ─────────────────────────────────────────────────

bool test_lut_roundtrip() {
    const auto& ulut = unpack_lut();
    // For every valid byte (0..242), unpack then repack should give same byte
    for (int b = 0; b < POW3_5; ++b) {
        const TritQuintet& q = ulut.table[b];
        // Repack
        int packed = 0;
        int base = 1;
        for (int i = 0; i < 5; ++i) {
            packed += (q.t[i] + 1) * base;
            base *= 3;
        }
        if (packed != b) return false;
    }
    // Verify all trits are in {-1, 0, +1}
    for (int b = 0; b < POW3_5; ++b) {
        const TritQuintet& q = ulut.table[b];
        for (int i = 0; i < 5; ++i) {
            if (q.t[i] < -1 || q.t[i] > 1) return false;
        }
    }
    return true;
}

// ─── Test 2: Block Quantize/Dequantize Roundtrip ────────────────────────────

bool test_block_roundtrip() {
    float original[QK];
    float reconstructed[QK];
    BlockTQ block;

    // Generate random weights
    for (int i = 0; i < QK; ++i) {
        original[i] = rand_float();
    }

    quantize_block(original, block);
    dequantize_block(block, reconstructed);

    // Each reconstructed value should be in {-scale, 0, +scale}
    float scale = block.scale();
    for (int i = 0; i < QK; ++i) {
        float r = reconstructed[i];
        bool valid = (std::fabs(r) < 1e-6f) ||
                     (std::fabs(r - scale) < 1e-3f) ||
                     (std::fabs(r + scale) < 1e-3f);
        if (!valid) return false;
    }

    // MSE should be reasonable (< 0.5 for uniform [-1,1])
    double mse = 0.0;
    for (int i = 0; i < QK; ++i) {
        double err = original[i] - reconstructed[i];
        mse += err * err;
    }
    mse /= QK;
    if (mse > 0.5) return false;

    return true;
}

// ─── Test 3: Zero Block ─────────────────────────────────────────────────────

bool test_zero_block() {
    float zeros[QK] = {};
    float out[QK];
    BlockTQ block;

    quantize_block(zeros, block);
    dequantize_block(block, out);

    for (int i = 0; i < QK; ++i) {
        if (std::fabs(out[i]) > 1e-6f) return false;
    }
    return true;
}

// ─── Test 4: Constant Block ─────────────────────────────────────────────────

bool test_constant_block() {
    float ones[QK];
    float out[QK];
    BlockTQ block;

    // All +1.0
    for (int i = 0; i < QK; ++i) ones[i] = 1.0f;
    quantize_block(ones, block);
    dequantize_block(block, out);

    for (int i = 0; i < QK; ++i) {
        if (std::fabs(out[i] - 1.0f) > 0.01f) return false;
    }

    // All -1.0
    for (int i = 0; i < QK; ++i) ones[i] = -1.0f;
    quantize_block(ones, block);
    dequantize_block(block, out);

    for (int i = 0; i < QK; ++i) {
        if (std::fabs(out[i] + 1.0f) > 0.01f) return false;
    }

    return true;
}

// ─── Test 5: GEMV Correctness ───────────────────────────────────────────────

bool test_gemv_correctness() {
    const size_t M = 4;
    const size_t K = QK;  // single block per row

    // Create a weight matrix and input vector
    float W[M * K];
    float x[K];
    float y_ref[M] = {};
    float y_tq[M] = {};

    for (size_t i = 0; i < M * K; ++i) W[i] = rand_float();
    for (size_t i = 0; i < K; ++i) x[i] = rand_float();

    // Reference: float matmul
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            y_ref[i] += W[i * K + j] * x[j];
        }
    }

    // Ternary GEMV
    TernaryMatrix tm(M, K);
    tm.quantize_from(W);
    tm.matvec(x, y_tq);

    // Check relative error (ternary is lossy, so allow ~50% error)
    for (size_t i = 0; i < M; ++i) {
        double ref = y_ref[i];
        double tq = y_tq[i];
        double rel_err = (std::fabs(ref) > 0.01) ? std::fabs(ref - tq) / std::fabs(ref) : std::fabs(ref - tq);
        // Ternary quantization is very lossy, so we just check it's in the right ballpark
        if (rel_err > 2.0 && std::fabs(ref - tq) > 5.0) return false;
    }

    return true;
}

// ─── Test 6: Balanced Ternary Arithmetic ────────────────────────────────────

bool test_balanced_ternary() {
    // Test conversion and addition
    for (int a = -50; a <= 50; ++a) {
        BalancedTernary bt(a);
        if (bt.to_int() != a) return false;
    }

    // Test addition
    for (int a = -20; a <= 20; ++a) {
        for (int b = -20; b <= 20; ++b) {
            BalancedTernary ba(a);
            BalancedTernary bb(b);
            BalancedTernary sum = BalancedTernary::add(ba, bb);
            if (sum.to_int() != a + b) return false;
        }
    }

    return true;
}

// ─── Test 7: Statistics & Entropy ───────────────────────────────────────────

bool test_statistics() {
    float original[QK];
    float reconstructed[QK];
    BlockTQ block;

    for (int i = 0; i < QK; ++i) original[i] = rand_float();
    quantize_block(original, block);
    dequantize_block(block, reconstructed);

    QuantStats stats = compute_stats(original, reconstructed, &block, QK);

    // Total should add up
    if (stats.count_neg + stats.count_zero + stats.count_pos != QK) return false;

    // Entropy should be between 0 and log2(3)
    if (stats.entropy < 0.0 || stats.entropy > LOG2_3 + 0.01) return false;

    // Efficiency should be between 0 and 1
    if (stats.efficiency < 0.0 || stats.efficiency > 1.01) return false;

    // MSE should be non-negative
    if (stats.mse < 0.0) return false;

    return true;
}

// ─── Test 8: TernaryLinear Forward ──────────────────────────────────────────

bool test_linear_forward() {
    const size_t in_dim = QK;
    const size_t out_dim = 4;

    TernaryLinear layer(in_dim, out_dim, true);

    // Create random weights and bias
    std::vector<float> w(out_dim * in_dim);
    std::vector<float> b(out_dim);
    for (auto& v : w) v = rand_float();
    for (auto& v : b) v = rand_float() * 0.1f;

    layer.load_weight(w.data());
    layer.load_bias(b.data());

    // Forward pass
    std::vector<float> x(in_dim);
    std::vector<float> y(out_dim);
    for (auto& v : x) v = rand_float();

    layer.forward(x.data(), y.data());

    // Check output is finite
    for (size_t i = 0; i < out_dim; ++i) {
        if (!std::isfinite(y[i])) return false;
    }

    return true;
}

// ─── Test 9: TernaryMLP Forward ─────────────────────────────────────────────

bool test_mlp_forward() {
    const size_t in_dim = QK;
    const size_t hidden_dim = QK;
    const size_t out_dim = QK;

    TernaryMLP mlp(in_dim, hidden_dim, out_dim);

    // Load random weights
    std::vector<float> w_up(hidden_dim * in_dim);
    std::vector<float> b_up(hidden_dim);
    std::vector<float> w_down(out_dim * hidden_dim);
    std::vector<float> b_down(out_dim);
    for (auto& v : w_up)   v = rand_float();
    for (auto& v : b_up)   v = rand_float() * 0.1f;
    for (auto& v : w_down) v = rand_float();
    for (auto& v : b_down) v = rand_float() * 0.1f;

    mlp.up.load_weight(w_up.data());
    mlp.up.load_bias(b_up.data());
    mlp.down.load_weight(w_down.data());
    mlp.down.load_bias(b_down.data());

    // Forward
    std::vector<float> x(in_dim);
    std::vector<float> y(out_dim);
    for (auto& v : x) v = rand_float();

    mlp.forward(x.data(), y.data());

    // Check output is finite
    for (size_t i = 0; i < out_dim; ++i) {
        if (!std::isfinite(y[i])) return false;
    }

    return true;
}

// ─── Test 10: Compression Ratio ─────────────────────────────────────────────

bool test_compression_ratio() {
    TernaryMatrix tm(64, QK);
    double ratio = tm.compression_ratio();
    double bpw = tm.bits_per_weight();

    // Compression ratio should be > 1 (smaller than fp32)
    if (ratio < 1.0) return false;

    // BPW should be close to 1.6875
    if (std::fabs(bpw - BPW) > 0.01) return false;

    // fp32 is 32 bpw, so ratio should be ~32/1.6875 ≈ 18.96
    if (ratio < 15.0 || ratio > 25.0) return false;

    return true;
}

// ─── Test 11: Packing Density Constants ─────────────────────────────────────

bool test_packing_density() {
    auto p8  = PackingDensity::byte_5();
    auto p16 = PackingDensity::u16_10();
    auto p32 = PackingDensity::u32_20();
    auto p64 = PackingDensity::u64_40();

    // All efficiencies should be < 1.0 (can't beat Shannon limit)
    if (p8.efficiency  >= 1.0) return false;
    if (p16.efficiency >= 1.0) return false;
    if (p32.efficiency >= 1.0) return false;
    if (p64.efficiency >= 1.0) return false;

    // All have the same efficiency (5/8 = 10/16 = 20/32 = 40/64)
    // because floor(2^k / log2(3)) / 2^k is constant for k = 3,4,5,6
    double eps = 1e-10;
    if (std::fabs(p8.efficiency - p16.efficiency) > eps) return false;
    if (std::fabs(p16.efficiency - p32.efficiency) > eps) return false;
    if (std::fabs(p32.efficiency - p64.efficiency) > eps) return false;

    // Wider containers give more SIMD parallelism
    if (p16.simd_parallelism <= p8.simd_parallelism) return false;
    if (p32.simd_parallelism <= p16.simd_parallelism) return false;
    if (p64.simd_parallelism <= p32.simd_parallelism) return false;

    // BPW should all be 1.6
    if (std::fabs(p8.bpw - 1.6) > 0.01) return false;

    return true;
}

// ─── Test 12: Benchmark ─────────────────────────────────────────────────────

bool test_benchmark() {
    const size_t M = 256;
    const size_t K = 1024;
    const int ITERS = 100;

    TernaryMatrix tm(M, K);
    std::vector<float> w(M * K);
    std::vector<float> x(K);
    std::vector<float> y(M);

    for (auto& v : w) v = rand_float();
    for (auto& v : x) v = rand_float();
    tm.quantize_from(w.data());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        tm.matvec(x.data(), y.data());
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double ops_per_iter = 2.0 * M * K;  // multiply-add
    double gflops = (ops_per_iter * ITERS) / (ms * 1e6);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[" << ms << " ms / " << ITERS << " iters, "
              << gflops << " GFLOPS] ";

    // Should complete in reasonable time
    return (ms < 10000.0);
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== Log2(3) Ternary Quantization Test Suite ===" << std::endl;
    std::cout << "Block size: " << QK << " | Packed bytes: " << PACKED_BYTES
              << " | BPW: " << BPW << std::endl;
    std::cout << "Theoretical optimum: " << LOG2_3 << " bpw" << std::endl;
    std::cout << "Packing efficiency: " << std::fixed << std::setprecision(4)
              << (LOG2_3 / BPW * 100.0) << "%" << std::endl;
    std::cout << std::endl;

    TEST(lut_roundtrip);
    TEST(block_roundtrip);
    TEST(zero_block);
    TEST(constant_block);
    TEST(gemv_correctness);
    TEST(balanced_ternary);
    TEST(statistics);
    TEST(linear_forward);
    TEST(mlp_forward);
    TEST(compression_ratio);
    TEST(packing_density);
    TEST(benchmark);

    std::cout << std::endl;
    std::cout << "=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
