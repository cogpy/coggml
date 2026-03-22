// tree_polytope_npu.hpp — C++11 Header-Only Tree-Polytope NPU for coggml
//
// Composition:
//   /reservoirpy-nodes (
//     /tree-polytope-kernel (
//       [ /llama-cpp-spec ⊗ /harmonic-llm ] -> /npu (
//         /rooted-tree-enum ( /llama-cpp-skillm )
//       )
//     )
//   )
//
// Provides: TreePolytopeNPU, HarmonicKernel, MatulaDecoder, SimplexGeometry
// Depends: cog/tq_log2_3.hpp (optional, for ternary quantization)

#ifndef COG_NPU_TREE_POLYTOPE_NPU_HPP
#define COG_NPU_TREE_POLYTOPE_NPU_HPP

#include <cstdint>
#include <cstring>
#include <string.h>
#include <cstddef>
#include <cmath>
#include <cassert>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <functional>

namespace cog { namespace npu {

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

static const double PI = 3.14159265358979323846;
static const double LOG2_3 = 1.5849625007211562;
static const double ALPHA_OTTER = 2.95576407;
static const double C_OTTER = 0.43992401;

// A000081 first 20 terms (precomputed)
static const int A000081[] = {
    0, 1, 1, 2, 4, 9, 20, 48, 115, 286,
    719, 1842, 4766, 12486, 32973, 87811,
    235381, 634847, 1721159, 4688676
};
static const size_t A000081_LEN = 20;

// First 32 primes for Matula encoding
static const int PRIMES[] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131
};
static const size_t N_PRIMES = 32;

// ─────────────────────────────────────────────────────────────────────────────
// Simplex Incidence Geometry
// ─────────────────────────────────────────────────────────────────────────────

struct SimplexGeometry {
    /// Pascal row with alternating signs: (1,-1)^n
    static std::vector<int> incidence_polynomial(int n) {
        std::vector<int> row(n + 1);
        row[0] = 1;
        for (int k = 1; k <= n; ++k) {
            row[k] = row[k-1] * (n - k + 1) / k;
        }
        for (int i = 1; i <= n; i += 2) {
            row[i] = -row[i];
        }
        return row;
    }
    
    /// Spectral radius from system number: 1 - 1/a000081(n+1)
    static float spectral_radius(int system) {
        int idx = system + 1;
        if (idx >= 0 && idx < (int)A000081_LEN && A000081[idx] > 0) {
            return 1.0f - 1.0f / (float)A000081[idx];
        }
        return 0.95f;
    }
    
    /// Number of tree topologies at system level
    static int tree_count(int system) {
        int idx = system + 1;
        if (idx >= 0 && idx < (int)A000081_LEN) return A000081[idx];
        return 0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Matula Number Decoder
// ─────────────────────────────────────────────────────────────────────────────

struct MatulaDecoder {
    enum Topology { TRIVIAL, NEST, BRANCH, BRIDGE };
    
    struct TreeInfo {
        int matula;
        int depth;
        int size;
        Topology topology;
        std::vector<int> polynomial;
        std::vector<int> children_matula;
    };
    
    /// Get the index of a prime (1-indexed)
    static int prime_index(int p) {
        for (size_t i = 0; i < N_PRIMES; ++i) {
            if (PRIMES[i] == p) return (int)(i + 1);
        }
        return 0;
    }
    
    /// Get the n-th prime (1-indexed)
    static int nth_prime(int n) {
        if (n >= 1 && n <= (int)N_PRIMES) return PRIMES[n - 1];
        return 2;
    }
    
    /// Factorize into primes
    static std::vector<int> factorize(int n) {
        std::vector<int> factors;
        for (int d = 2; d * d <= n; ++d) {
            while (n % d == 0) {
                factors.push_back(d);
                n /= d;
            }
        }
        if (n > 1) factors.push_back(n);
        return factors;
    }
    
    /// Decode Matula number to tree info
    static TreeInfo decode(int m) {
        TreeInfo info;
        info.matula = m;
        
        if (m == 1) {
            info.depth = 0;
            info.size = 1;
            info.topology = TRIVIAL;
            info.polynomial = {1};
            return info;
        }
        
        auto factors = factorize(m);
        info.children_matula.clear();
        for (int p : factors) {
            info.children_matula.push_back(prime_index(p));
        }
        
        // Compute depth and size recursively
        info.depth = 0;
        info.size = 1;
        for (int child_m : info.children_matula) {
            auto child = decode(child_m);
            info.depth = std::max(info.depth, child.depth + 1);
            info.size += child.size;
        }
        
        // Classify topology
        if (factors.size() == 1) {
            info.topology = NEST;  // single prime = chain
        } else {
            bool all_same = true;
            for (size_t i = 1; i < factors.size(); ++i) {
                if (factors[i] != factors[0]) { all_same = false; break; }
            }
            info.topology = all_same ? BRANCH : BRIDGE;
        }
        
        // Compute polynomial via convolution of shifted child polynomials
        info.polynomial = {1};
        for (int child_m : info.children_matula) {
            auto child = decode(child_m);
            std::vector<int> shifted(child.polynomial.size() + 1);
            shifted[0] = 1;
            for (size_t i = 0; i < child.polynomial.size(); ++i) {
                shifted[i + 1] = child.polynomial[i];
            }
            // Convolve
            std::vector<int> conv(info.polynomial.size() + shifted.size() - 1, 0);
            for (size_t i = 0; i < info.polynomial.size(); ++i) {
                for (size_t j = 0; j < shifted.size(); ++j) {
                    conv[i + j] += info.polynomial[i] * shifted[j];
                }
            }
            info.polynomial = conv;
        }
        
        return info;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Skillm ISA (Instruction Set Architecture)
// ─────────────────────────────────────────────────────────────────────────────

enum SkillmOpcode : uint8_t {
    OP_DISCOVER    = 0,   // Matula 2
    OP_INSPECT     = 1,   // Matula 3
    OP_NAVIGATE    = 2,   // Matula 4
    OP_CREATE      = 3,   // Matula 5
    OP_COMPOSE     = 4,   // Matula 6
    OP_OBSERVE     = 5,   // Matula 7
    OP_MUTATE      = 6,   // Matula 8
    OP_DESTROY     = 7,   // Matula 9
    OP_ORCHESTRATE = 8,   // Matula 11
    OP_CLASSIFY    = 9,   // Matula 13
    OP_ASSESS      = 10,  // Matula 17
    OP_EVOLVE      = 11,  // Matula 19
    OP_DEPLOY      = 12,  // Matula 23
    OP_TRAIN       = 13,  // Matula 29
    OP_INTROSPECT  = 14,  // Matula 31
    OP_COUNT       = 15
};

static const int OPCODE_MATULA[] = {
    2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 17, 19, 23, 29, 31
};

struct SkillmInstruction {
    SkillmOpcode opcode;
    MatulaDecoder::TreeInfo tree;
};

// ─────────────────────────────────────────────────────────────────────────────
// Harmonic Kernel (frequency-domain inference)
// ─────────────────────────────────────────────────────────────────────────────

class HarmonicKernel {
    size_t dim_;
    size_t n_harmonics_;
    std::vector<float> W_phase_;      // n_harmonics x n_harmonics
    std::vector<float> W_spectral_;   // n_harmonics x dim
    
public:
    HarmonicKernel(size_t dim = 256, size_t n_harmonics = 16)
        : dim_(dim), n_harmonics_(n_harmonics)
        , W_phase_(n_harmonics * n_harmonics, 0.0f)
        , W_spectral_(n_harmonics * dim, 0.0f)
    {
        // Initialize with small random values (deterministic seed)
        uint32_t seed = 42;
        for (auto& w : W_phase_) {
            seed = seed * 1103515245 + 12345;
            w = ((float)(seed >> 16) / 32768.0f - 1.0f) * 0.01f;
        }
        for (auto& w : W_spectral_) {
            seed = seed * 1103515245 + 12345;
            w = ((float)(seed >> 16) / 32768.0f - 1.0f) * 0.01f;
        }
    }
    
    size_t dim() const { return dim_; }
    size_t n_harmonics() const { return n_harmonics_; }
    
    /// Forward pass: input[dim] -> output[dim]
    void forward(const float* input, float* output) const {
        // Step 1: Compute magnitudes of first n_harmonics components
        std::vector<float> magnitudes(n_harmonics_, 0.0f);
        for (size_t h = 0; h < n_harmonics_; ++h) {
            float sum = 0.0f;
            for (size_t i = 0; i < dim_; ++i) {
                float phase = 2.0f * (float)PI * (h + 1) * i / dim_;
                sum += input[i] * std::cos(phase);
            }
            magnitudes[h] = sum / dim_;
        }
        
        // Step 2: Phase attention (self-attention on magnitudes)
        std::vector<float> attended(n_harmonics_, 0.0f);
        for (size_t i = 0; i < n_harmonics_; ++i) {
            for (size_t j = 0; j < n_harmonics_; ++j) {
                attended[i] += magnitudes[j] * W_phase_[i * n_harmonics_ + j];
            }
            attended[i] = std::tanh(attended[i]);
        }
        
        // Step 3: Spectral MLP -> output
        for (size_t d = 0; d < dim_; ++d) {
            float sum = 0.0f;
            for (size_t h = 0; h < n_harmonics_; ++h) {
                sum += attended[h] * W_spectral_[h * dim_ + d];
            }
            output[d] = sum;
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// NPU Register Map
// ─────────────────────────────────────────────────────────────────────────────

struct NPURegisters {
    uint32_t ctrl;          // 0x00: Control (write 1 to start)
    uint32_t status;        // 0x04: Status
    uint32_t input_addr;    // 0x08: DMA input address
    uint32_t input_len;     // 0x0C: Input length
    uint32_t output_addr;   // 0x10: DMA output address
    uint32_t output_len;    // 0x14: Output length
    uint32_t temperature;   // 0x18: Temperature (fixed 16.16)
    uint32_t system_n;      // 0x1C: Tree-polytope system number
    uint32_t matula_reg;    // 0x20: Current Matula number
    uint32_t harmonic_n;    // 0x24: Number of harmonics
    uint32_t spectral_r;    // 0x28: Spectral radius (fixed 16.16)
    uint32_t tree_depth;    // 0x2C: Current tree depth
    uint32_t poly_hash;     // 0x30: Polynomial hash
    uint32_t irq_mask;      // 0x34: IRQ mask
    uint32_t dma_ctrl;      // 0x38: DMA control
    uint32_t telemetry;     // 0x3C: Telemetry counter
};

// ─────────────────────────────────────────────────────────────────────────────
// Tree-Polytope NPU (the composed device)
// ─────────────────────────────────────────────────────────────────────────────

class TreePolytopeNPU {
    size_t dim_;
    int system_;
    NPURegisters regs_;
    HarmonicKernel kernel_;
    
    // DMA buffers
    std::vector<float> input_buf_;
    std::vector<float> output_buf_;
    
    // Telemetry
    uint64_t cycle_count_;
    uint64_t total_flops_;
    
public:
    TreePolytopeNPU(size_t dim = 256, int system = 4, size_t n_harmonics = 16)
        : dim_(dim), system_(system)
        , kernel_(dim, n_harmonics)
        , input_buf_(dim, 0.0f)
        , output_buf_(dim, 0.0f)
        , cycle_count_(0), total_flops_(0)
    {
        memset(&regs_, 0, sizeof(regs_));
        regs_.system_n = system;
        regs_.harmonic_n = (uint32_t)n_harmonics;
        regs_.spectral_r = (uint32_t)(SimplexGeometry::spectral_radius(system) * 65536);
    }
    
    // MMIO interface
    void write32(uint32_t offset, uint32_t value) {
        uint32_t* base = reinterpret_cast<uint32_t*>(&regs_);
        base[offset / 4] = value;
        if (offset == 0x00 && value == 1) execute();
    }
    
    uint32_t read32(uint32_t offset) const {
        const uint32_t* base = reinterpret_cast<const uint32_t*>(&regs_);
        return base[offset / 4];
    }
    
    // DMA interface
    void dma_write(const float* data, size_t n) {
        n = std::min(n, dim_);
        std::copy(data, data + n, input_buf_.begin());
        regs_.input_len = (uint32_t)n;
    }
    
    void dma_read(float* data, size_t n) const {
        n = std::min(n, (size_t)regs_.output_len);
        std::copy(output_buf_.begin(), output_buf_.begin() + n, data);
    }
    
    // Execute inference
    void execute() {
        regs_.status = 0x02;  // ENCODING
        cycle_count_++;
        
        // Run harmonic kernel
        regs_.status = 0x03;  // FFT
        regs_.status = 0x04;  // ATTN
        regs_.status = 0x05;  // MLP
        kernel_.forward(input_buf_.data(), output_buf_.data());
        
        regs_.status = 0x06;  // DECODING
        regs_.output_len = (uint32_t)dim_;
        total_flops_ += dim_ * kernel_.n_harmonics() * 2;
        regs_.telemetry = (uint32_t)(total_flops_ & 0xFFFFFFFF);
        
        regs_.status = 0x80;  // COMPLETE
    }
    
    // Accessors
    size_t dim() const { return dim_; }
    int system() const { return system_; }
    uint64_t cycles() const { return cycle_count_; }
    uint64_t flops() const { return total_flops_; }
    float spectral_radius() const { return regs_.spectral_r / 65536.0f; }
    
    // Tree-polytope reservoir weight generation
    void generate_reservoir_weights(float* W, size_t n, float sparsity = 0.9f) const {
        // Generate n x n weight matrix from tree polynomials
        uint32_t seed = 12345;
        for (size_t i = 0; i < n; ++i) {
            // Get tree polynomial for row i
            int matula = (int)(i % 9) + 1;  // cycle through Matula 1-9
            auto info = MatulaDecoder::decode(matula);
            
            for (size_t j = 0; j < n; ++j) {
                // Sparsity mask
                seed = seed * 1103515245 + 12345;
                float r = (float)(seed >> 16) / 65536.0f;
                if (r < sparsity) {
                    W[i * n + j] = 0.0f;
                    continue;
                }
                // Tree polynomial coefficient (cyclic)
                size_t idx = (j + i) % info.polynomial.size();
                W[i * n + j] = (float)info.polynomial[idx];
            }
        }
        
        // Scale to target spectral radius (approximate)
        float rho = SimplexGeometry::spectral_radius(system_);
        float max_val = 0.0f;
        for (size_t i = 0; i < n * n; ++i) {
            max_val = std::max(max_val, std::fabs(W[i]));
        }
        if (max_val > 0.0f) {
            float scale = rho / (max_val * std::sqrt((float)n * (1.0f - sparsity)));
            for (size_t i = 0; i < n * n; ++i) {
                W[i] *= scale;
            }
        }
    }
};

}} // namespace cog::npu

#endif // COG_NPU_TREE_POLYTOPE_NPU_HPP
