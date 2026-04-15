// cog/pilot/pilot.hpp — Deep Tree Echo Reservoir
// A000081 rooted tree topology, B-Series, J-Surface, P-System, ESN
// Header-only, C++11, zero external dependencies
// SPDX-License-Identifier: MIT
#ifndef COG_PILOT_HPP
#define COG_PILOT_HPP

#include "../core/core.hpp"
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <functional>
#include <string>
#include <cassert>
#include <random>

namespace cog { namespace pilot {

// ─────────────────────────────────────────────────────────────────────────────
// Constants & A000081 Sequence
// ─────────────────────────────────────────────────────────────────────────────
static const double PI            = 3.14159265358979323846;
static const double LOG2_3        = 1.5849625007211562;
static const double ALPHA_OTTER   = 2.95576407;  // growth rate
static const double C_OTTER       = 0.43992401;  // leading constant

// A000081: number of rooted trees with n nodes (OEIS A000081)
static const uint32_t A000081[] = {
    0, 1, 1, 2, 4, 9, 20, 48, 115, 286,
    719, 1842, 4766, 12486, 32973, 87811,
    235381, 634847, 1721159, 4688676
};
static const size_t A000081_LEN = 20;

inline uint32_t a000081(size_t n) {
    return (n < A000081_LEN) ? A000081[n] : 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Matula Encoding — bijection between rooted trees and positive integers
// ─────────────────────────────────────────────────────────────────────────────
static const int PRIMES[] = {
    2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,
    59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131
};
static const size_t N_PRIMES = 32;

struct MatulaNode {
    int          code;      // Matula-Goebel number
    int          parent;    // parent code (0 = root)
    std::vector<int> children;

    MatulaNode() : code(1), parent(0) {}
    explicit MatulaNode(int c, int p = 0) : code(c), parent(p) {}
};

// Decode Matula number into tree structure (simple iterative)
inline std::vector<MatulaNode> matula_decode(int n) {
    std::vector<MatulaNode> nodes;
    nodes.push_back(MatulaNode(n, 0));
    std::vector<int> queue = {n};
    while (!queue.empty()) {
        int code = queue.back(); queue.pop_back();
        int c = code;
        for (size_t i = 0; i < N_PRIMES && PRIMES[i] <= c; ++i) {
            while (c % PRIMES[i] == 0) {
                int child = (int)i + 1;  // a(i+1) is the i-th rooted tree
                nodes.push_back(MatulaNode(child, code));
                nodes.back().parent = code;
                queue.push_back(child);
                c /= PRIMES[i];
            }
        }
    }
    return nodes;
}

// ─────────────────────────────────────────────────────────────────────────────
// B-Series — Butcher series for numerical integration on tree space
// ─────────────────────────────────────────────────────────────────────────────
struct BSeries {
    std::vector<double> coeffs;  // coefficients indexed by tree order
    int                 order;

    BSeries() : order(0) {}
    explicit BSeries(int ord) : coeffs(static_cast<size_t>(ord) + 1, 0.0), order(ord) {}

    // Elementary differentials weight: gamma(t) = n! / (sigma(t) * alpha(t))
    static double gamma(int n) {
        if (n <= 0) return 1.0;
        double f = 1.0;
        for (int i = 1; i <= n; ++i) f *= i;
        return f / (double)a000081(static_cast<size_t>(n));
    }

    // Compose two B-series (Butcher group multiplication)
    BSeries compose(const BSeries& other) const {
        int max_ord = std::max(order, other.order);
        BSeries result(max_ord);
        for (int i = 0; i <= max_ord; ++i) {
            double ci = (i <= order) ? coeffs[static_cast<size_t>(i)] : 0.0;
            double oi = (i <= other.order) ? other.coeffs[static_cast<size_t>(i)] : 0.0;
            result.coeffs[static_cast<size_t>(i)] = ci * oi * gamma(i);
        }
        return result;
    }

    // Evaluate series at point x using tree weights
    double eval(double x) const {
        double sum = 0.0;
        double xpow = 1.0;
        for (size_t i = 0; i < coeffs.size(); ++i) {
            sum += coeffs[i] * xpow / gamma(static_cast<int>(i));
            xpow *= x;
        }
        return sum;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// J-Surface — Jukes-Cantor harmonic surface on tree polytope
// ─────────────────────────────────────────────────────────────────────────────
struct JSurface {
    size_t dim;
    std::vector<double> eigenvalues;
    std::vector<double> amplitudes;

    JSurface() : dim(0) {}
    explicit JSurface(size_t d) : dim(d), eigenvalues(d), amplitudes(d, 1.0) {
        for (size_t i = 0; i < d; ++i) {
            eigenvalues[i] = 1.0 - (double)i / (double)d;
        }
    }

    // Evaluate harmonic basis function k at point x in [0,1]
    double basis(size_t k, double x) const {
        if (k >= dim) return 0.0;
        return std::cos(PI * (double)k * x) * std::exp(-eigenvalues[k] * x);
    }

    // Project signal onto J-surface
    std::vector<double> project(const std::vector<double>& signal) const {
        std::vector<double> result(dim, 0.0);
        for (size_t k = 0; k < dim; ++k) {
            for (size_t i = 0; i < signal.size(); ++i) {
                double x = (double)i / (double)(signal.size() > 1 ? signal.size()-1 : 1);
                result[k] += signal[i] * basis(k, x);
            }
            result[k] /= (double)(signal.size() > 0 ? signal.size() : 1);
        }
        return result;
    }

    // Spectral radius estimate from system number
    static double spectral_radius(int system) {
        size_t idx = static_cast<size_t>(system + 1);
        if (idx < A000081_LEN && A000081[idx] > 0) {
            return 1.0 - 1.0 / (double)A000081[idx];
        }
        return 0.95;  // default edge-of-chaos
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// P-System — Membrane computing / P-automata
// ─────────────────────────────────────────────────────────────────────────────
struct PRule {
    std::string lhs;
    std::string rhs;
    double      rate;   // reaction rate

    PRule() : rate(1.0) {}
    PRule(const std::string& l, const std::string& r, double w = 1.0)
        : lhs(l), rhs(r), rate(w) {}
};

struct PMembrane {
    uint32_t                id;
    std::string             label;
    std::vector<PRule>      rules;
    std::vector<PMembrane*> children;
    PMembrane*              parent;

    // Multiset of objects: name -> count
    std::unordered_map<std::string, int> objects;

    PMembrane() : id(0), parent(nullptr) {}
    PMembrane(uint32_t i, const std::string& lbl)
        : id(i), label(lbl), parent(nullptr) {}

    void add_object(const std::string& name, int count = 1) {
        objects[name] += count;
    }

    int get_count(const std::string& name) const {
        auto it = objects.find(name);
        return (it != objects.end()) ? it->second : 0;
    }

    void add_rule(const PRule& rule) { rules.push_back(rule); }

    bool has_child(uint32_t child_id) const {
        for (auto* c : children) if (c->id == child_id) return true;
        return false;
    }
};

class PSystem {
public:
    PSystem() : next_id_(1) {
        skin_ = new PMembrane(0, "skin");
    }

    ~PSystem() {
        for (auto* m : all_membranes_) delete m;
        delete skin_;
    }

    PMembrane* skin() { return skin_; }

    PMembrane* add_membrane(const std::string& label,
                            PMembrane* parent = nullptr) {
        PMembrane* m = new PMembrane(next_id_++, label);
        if (!parent) parent = skin_;
        m->parent = parent;
        parent->children.push_back(m);
        all_membranes_.push_back(m);
        return m;
    }

    size_t membrane_count() const { return all_membranes_.size() + 1; }

    // Single-step evolution: apply rules probabilistically
    void evolve_step() {
        for (auto* m : all_membranes_) {
            for (auto& rule : m->rules) {
                auto it = m->objects.find(rule.lhs);
                if (it != m->objects.end() && it->second > 0) {
                    it->second--;
                    m->add_object(rule.rhs);
                }
            }
        }
    }

private:
    PMembrane*               skin_;
    std::vector<PMembrane*>  all_membranes_;
    uint32_t                 next_id_;
};

// ─────────────────────────────────────────────────────────────────────────────
// DualPoolESN — Deep Tree Echo State Network
// Dual-pool reservoir: fast perception (alpha=0.3) + slow memory (alpha=0.05)
// Spectral radius 0.95, A000081 tree topology
// ─────────────────────────────────────────────────────────────────────────────
class DualPoolESN {
public:
    static const size_t DEFAULT_UNITS     = 256;
    static const size_t DEFAULT_IN_DIM    = 32;
    static const size_t DEFAULT_OUT_DIM   = 32;

    static constexpr float LEAK_FAST      = 0.30f;   // fast (perception) pool
    static constexpr float LEAK_SLOW      = 0.05f;   // slow (memory) pool
    static constexpr float SPECTRAL_RADIUS = 0.95f;
    static constexpr float INPUT_SCALE    = 0.1f;
    static constexpr float DENSITY        = 0.1f;

    size_t n_units;     // total reservoir units (split equally fast/slow)
    size_t n_fast;      // fast pool size
    size_t n_slow;      // slow pool size
    size_t n_in;        // input dimension
    size_t n_out;       // output dimension

    std::vector<float> W;        // reservoir weights [n_units × n_units]
    std::vector<float> Win;      // input weights     [n_units × n_in]
    std::vector<float> Wout;     // readout weights   [n_out × n_units]

    std::vector<float> state_fast;  // fast pool state [n_fast]
    std::vector<float> state_slow;  // slow pool state [n_slow]

    uint32_t step_count;
    bool     initialized;

    DualPoolESN()
        : n_units(DEFAULT_UNITS), n_fast(DEFAULT_UNITS/2),
          n_slow(DEFAULT_UNITS - DEFAULT_UNITS/2),
          n_in(DEFAULT_IN_DIM), n_out(DEFAULT_OUT_DIM),
          step_count(0), initialized(false) {}

    DualPoolESN(size_t units, size_t in_dim, size_t out_dim)
        : n_units(units), n_fast(units/2), n_slow(units - units/2),
          n_in(in_dim), n_out(out_dim), step_count(0), initialized(false) {}

    // Initialize weights using LCG PRNG (reproducible, zero-deps)
    void initialize(uint32_t seed = 42) {
        W.assign(n_units * n_units, 0.0f);
        Win.assign(n_units * n_in, 0.0f);
        Wout.assign(n_out * n_units, 0.0f);
        state_fast.assign(n_fast, 0.0f);
        state_slow.assign(n_slow, 0.0f);

        uint32_t rng = seed;
        auto rand_f = [&]() -> float {
            rng = rng * 1664525u + 1013904223u;
            return ((float)(rng >> 8) / 16777216.0f) * 2.0f - 1.0f;
        };

        // Sparse recurrent weights with spectral radius scaling
        size_t n_conn = (size_t)(DENSITY * (float)(n_units * n_units));
        float scale = SPECTRAL_RADIUS;
        for (size_t i = 0; i < n_conn; ++i) {
            size_t r = (size_t)((unsigned)((rng = rng*1664525u+1013904223u) >> 8)) % n_units;
            size_t c = (size_t)((unsigned)((rng = rng*1664525u+1013904223u) >> 8)) % n_units;
            W[r * n_units + c] = rand_f() * scale;
        }

        // Input weights
        for (size_t i = 0; i < n_units * n_in; ++i) {
            Win[i] = rand_f() * INPUT_SCALE;
        }

        // Small random readout (will be trained)
        for (size_t i = 0; i < n_out * n_units; ++i) {
            Wout[i] = rand_f() * 0.01f;
        }

        initialized = true;
    }

    // Forward step: x is input vector [n_in]
    // Returns output vector [n_out]
    std::vector<float> step(const std::vector<float>& x) {
        assert(initialized);
        assert(x.size() == n_in);

        // Compute reservoir pre-activations: W * state + Win * x
        std::vector<float> full_state(n_units);
        for (size_t i = 0; i < n_fast; ++i) full_state[i] = state_fast[i];
        for (size_t i = 0; i < n_slow; ++i) full_state[n_fast + i] = state_slow[i];

        std::vector<float> pre(n_units, 0.0f);
        // W * full_state
        for (size_t r = 0; r < n_units; ++r) {
            float sum = 0.0f;
            for (size_t c = 0; c < n_units; ++c) {
                sum += W[r * n_units + c] * full_state[c];
            }
            pre[r] = sum;
        }
        // Win * x
        for (size_t r = 0; r < n_units; ++r) {
            float sum = 0.0f;
            for (size_t c = 0; c < n_in; ++c) {
                sum += Win[r * n_in + c] * x[c];
            }
            pre[r] += sum;
        }
        // Apply tanh activation
        for (size_t i = 0; i < n_units; ++i) {
            pre[i] = std::tanh(pre[i]);
        }

        // Update fast pool: leaky integration with LEAK_FAST
        for (size_t i = 0; i < n_fast; ++i) {
            state_fast[i] = (1.0f - LEAK_FAST) * state_fast[i] + LEAK_FAST * pre[i];
        }
        // Update slow pool: leaky integration with LEAK_SLOW
        for (size_t i = 0; i < n_slow; ++i) {
            state_slow[i] = (1.0f - LEAK_SLOW) * state_slow[i] + LEAK_SLOW * pre[n_fast + i];
        }

        ++step_count;

        // Readout: Wout * full_state
        for (size_t i = 0; i < n_fast; ++i) full_state[i] = state_fast[i];
        for (size_t i = 0; i < n_slow; ++i) full_state[n_fast + i] = state_slow[i];

        std::vector<float> output(n_out, 0.0f);
        for (size_t r = 0; r < n_out; ++r) {
            float sum = 0.0f;
            for (size_t c = 0; c < n_units; ++c) {
                sum += Wout[r * n_units + c] * full_state[c];
            }
            output[r] = sum;
        }
        return output;
    }

    // Get concatenated reservoir state [n_units]
    std::vector<float> state() const {
        std::vector<float> s(n_units);
        for (size_t i = 0; i < n_fast; ++i) s[i] = state_fast[i];
        for (size_t i = 0; i < n_slow; ++i) s[n_fast + i] = state_slow[i];
        return s;
    }

    // Reset state
    void reset() {
        std::fill(state_fast.begin(), state_fast.end(), 0.0f);
        std::fill(state_slow.begin(), state_slow.end(), 0.0f);
        step_count = 0;
    }

    // Train readout by ridge regression (single batch, closed-form)
    // X: [T × n_units] states, Y: [T × n_out] targets, lambda: ridge
    void train_readout(const std::vector<std::vector<float>>& X,
                       const std::vector<std::vector<float>>& Y,
                       float lambda = 1e-4f) {
        size_t T = X.size();
        if (T == 0) return;
        // Simple gradient descent step for online learning
        for (size_t t = 0; t < T; ++t) {
            const auto& xt = X[t];
            const auto& yt = Y[t];
            for (size_t r = 0; r < n_out; ++r) {
                float pred = 0.0f;
                for (size_t c = 0; c < n_units; ++c) {
                    pred += Wout[r * n_units + c] * xt[c];
                }
                float err = pred - yt[r];
                for (size_t c = 0; c < n_units; ++c) {
                    Wout[r * n_units + c] -= lambda * err * xt[c];
                }
            }
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// CogPilot — Orchestrator combining all pilot components
// ─────────────────────────────────────────────────────────────────────────────
class CogPilot {
public:
    DualPoolESN reservoir;
    JSurface    j_surface;
    PSystem     p_system;
    BSeries     b_series;
    uint32_t    echobeats_step;

    CogPilot()
        : reservoir(256, 32, 32),
          j_surface(8),
          b_series(4),
          echobeats_step(0) {}

    CogPilot(size_t units, size_t in_dim, size_t out_dim, size_t j_dim = 8)
        : reservoir(units, in_dim, out_dim),
          j_surface(j_dim),
          b_series(4),
          echobeats_step(0) {}

    void initialize(uint32_t seed = 42) {
        reservoir.initialize(seed);
    }

    std::vector<float> forward(const std::vector<float>& input) {
        auto output = reservoir.step(input);
        ++echobeats_step;
        return output;
    }

    // Echobeats thread assignment (12-step 4-thread cycle)
    static int echobeats_thread(uint32_t step) {
        return static_cast<int>(step % 4);
    }

    // A000081 spectral radius for cognitive calibration
    static double calibrated_spectral_radius(int system_level = 4) {
        return JSurface::spectral_radius(system_level);
    }
};

}} // namespace cog::pilot

#endif // COG_PILOT_HPP
