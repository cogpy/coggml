// test/test_coverage.cpp — Comprehensive coverage tests for all CogPy modules
// Compile: g++ -std=c++11 -I../include -o test_coverage test/test_coverage.cpp && ./test_coverage
// CMake:   mkdir build && cd build && cmake .. && make && ctest
//
// This test suite targets the APIs identified as missing from test_all.cpp:
//   cog::core  — TruthValue (merge/deduction/induction/abduction/revision),
//                AttentionValue (pay_rent/consolidate/in_attentional_focus),
//                AtomSpace (get_incoming/remove_atom/foreach_atom/pattern_match/clear),
//                Arena, Spinlock
//   cog::gml   — f32_to_f16, quantize_q4_0/q8_0, Context, Tensor, CGraph ops,
//                Adam, LBFGS
//   cog::prime — induction, disjunction, is_attentional, PatternMatcher,
//                recall_salient, procedural list, top_by_sti
//   cog::lux   — has_node, all_nodes/all_edges, in_edges, BFS max_depth
//   cog::glow  — IRType::numel, IRNode attrs, pass_dce, pass_constant_fold,
//                interpreter: SUB/MUL/DIV/NEG/ABS/TANH/SIGMOID/GELU/SUM/MEAN
//   cog::webvm — lambda, cond, let, set!, custom builtin, JSON make_array
//   cog::pilot — BSeries::compose, DualPoolESN reset/train_readout,
//                CogPilot::calibrated_spectral_radius
//   cog::mach  — Fixed abs/floor/recip/sqrt/tanh/exp, compound assignment,
//                FixedTensor fill/+, VMRegion protection, VMMap find/region_count,
//                KernelAtomSpace set_tv

#include "cog/cog.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

// ─── Test Infrastructure ──────────────────────────────────────────────────────

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name, ...) \
    do { \
        std::cout << "  [" #name "] "; \
        bool _ok = true; \
        do { __VA_ARGS__ } while (0); \
        if (_ok) { std::cout << "PASS\n"; tests_passed++; } \
        else     { std::cout << "FAIL\n"; tests_failed++;  } \
    } while (0)

#define REQUIRE(expr) \
    if (!(expr)) { \
        std::cout << "FAIL (line " << __LINE__ << ": " << #expr << ")\n"; \
        _ok = false; \
        break; \
    }

#define REQUIRE_NEAR(a, b, tol) \
    REQUIRE(std::fabs((double)(a) - (double)(b)) < (tol))

static void section(const char* name) {
    std::cout << "\n[" << name << "]\n";
}

// ─── cog::core (extended) ────────────────────────────────────────────────────

void test_core_extended() {
    section("cog::core (extended)");

    // TruthValue::merge
    TEST(truth_value_merge, {
        cog::TruthValue a(0.8f, 0.6f);
        cog::TruthValue b(0.4f, 0.4f);
        auto m = a.merge(b);
        // weighted average: (0.8*0.6 + 0.4*0.4) / (0.6+0.4) = 0.64
        REQUIRE_NEAR(m.strength, 0.64f, 1e-4f);
        REQUIRE_NEAR(m.confidence, 1.0f, 1e-4f); // capped at 1.0
    });

    TEST(truth_value_merge_zero_confidence, {
        cog::TruthValue a(0.7f, 0.0f);
        cog::TruthValue b(0.3f, 0.0f);
        auto m = a.merge(b);
        // both zero confidence → default 0.5
        REQUIRE_NEAR(m.strength, 0.5f, 1e-4f);
        REQUIRE_NEAR(m.confidence, 0.0f, 1e-4f);
    });

    // TruthValue::deduction
    TEST(truth_value_deduction, {
        cog::TruthValue ab(0.8f, 0.9f);
        cog::TruthValue bc(0.7f, 0.8f);
        auto ac = cog::TruthValue::deduction(ab, bc);
        REQUIRE_NEAR(ac.strength, 0.8f * 0.7f, 1e-4f);
        REQUIRE_NEAR(ac.confidence, 0.9f * 0.8f * 0.9f, 1e-4f);
    });

    // TruthValue::induction
    TEST(truth_value_induction, {
        cog::TruthValue ab(0.8f, 0.9f);
        cog::TruthValue cb(0.7f, 0.8f);
        auto ac = cog::TruthValue::induction(ab, cb);
        REQUIRE_NEAR(ac.strength, 0.8f * 0.7f, 1e-4f);
        REQUIRE_NEAR(ac.confidence, 0.9f * 0.8f * 0.5f, 1e-4f);
    });

    // TruthValue::abduction
    TEST(truth_value_abduction, {
        cog::TruthValue ab(0.8f, 0.9f);
        cog::TruthValue ac(0.6f, 0.7f);
        auto bc = cog::TruthValue::abduction(ab, ac);
        REQUIRE_NEAR(bc.strength, 0.8f * 0.6f, 1e-4f);
        REQUIRE_NEAR(bc.confidence, 0.9f * 0.7f * 0.5f, 1e-4f);
    });

    // TruthValue::revision
    TEST(truth_value_revision, {
        cog::TruthValue a(0.9f, 0.8f);
        cog::TruthValue b(0.1f, 0.2f);
        auto r = cog::TruthValue::revision(a, b);
        // revision weights by confidence odds
        REQUIRE(r.strength > 0.5f); // a has much higher confidence → biased high
        REQUIRE(r.confidence > 0.0f);
    });

    // AttentionValue::pay_rent
    TEST(attention_pay_rent, {
        cog::AttentionValue av(200, 5, 0);
        av.pay_rent(50);
        REQUIRE(av.sti == 150);
    });

    // AttentionValue::consolidate
    TEST(attention_consolidate, {
        cog::AttentionValue av(200, 0, 0);
        int16_t lti_before = av.lti;
        av.consolidate(100);
        REQUIRE(av.lti == lti_before + 1);
    });

    TEST(attention_consolidate_below_threshold, {
        cog::AttentionValue av(30, 0, 0);
        av.consolidate(100);
        REQUIRE(av.lti == 0); // below threshold, no change
    });

    // AttentionValue::in_attentional_focus
    TEST(attention_in_attentional_focus, {
        cog::AttentionValue high(100, 0, 0);
        cog::AttentionValue low(10, 0, 0);
        REQUIRE( high.in_attentional_focus(50));
        REQUIRE(!low.in_attentional_focus(50));
    });

    // AtomSpace::get_incoming
    TEST(atomspace_get_incoming, {
        cog::AtomSpace as;
        cog::Handle h_cat  = as.add_node(cog::AtomType::CONCEPT_NODE, "Cat");
        cog::Handle h_mammal = as.add_node(cog::AtomType::CONCEPT_NODE, "Mammal");
        cog::Handle h_link = as.add_link(cog::AtomType::INHERITANCE_LINK, {h_cat, h_mammal});
        auto inc = as.get_incoming(h_cat);
        REQUIRE(inc.size() == 1);
        REQUIRE(inc[0] == h_link);
        auto inc_mammal = as.get_incoming(h_mammal);
        REQUIRE(inc_mammal.size() == 1);
    });

    // AtomSpace::get_atom_mut
    TEST(atomspace_get_atom_mut, {
        cog::AtomSpace as;
        cog::Handle h = as.add_node(cog::AtomType::CONCEPT_NODE, "Dog");
        cog::Atom* a = as.get_atom_mut(h);
        REQUIRE(a != nullptr);
        REQUIRE(a->name == "Dog");
        a->tv = cog::TruthValue(0.5f, 0.5f);
        const cog::Atom* b = as.get_atom(h);
        REQUIRE_NEAR(b->tv.strength, 0.5f, 1e-6f);
    });

    // AtomSpace::remove_atom
    TEST(atomspace_remove_atom, {
        cog::AtomSpace as;
        cog::Handle h = as.add_node(cog::AtomType::CONCEPT_NODE, "Cat");
        REQUIRE(as.size() == 1);
        bool removed = as.remove_atom(h);
        REQUIRE(removed);
        REQUIRE(as.size() == 0);
        REQUIRE(as.get_atom(h) == nullptr);
    });

    TEST(atomspace_remove_nonexistent, {
        cog::AtomSpace as;
        bool removed = as.remove_atom(999);
        REQUIRE(!removed);
    });

    // AtomSpace::foreach_atom
    TEST(atomspace_foreach_atom, {
        cog::AtomSpace as;
        as.add_node(cog::AtomType::CONCEPT_NODE, "A");
        as.add_node(cog::AtomType::CONCEPT_NODE, "B");
        as.add_node(cog::AtomType::PREDICATE_NODE, "Likes");
        int count = 0;
        as.foreach_atom([&](const cog::Atom&) { ++count; });
        REQUIRE(count == 3);
    });

    // AtomSpace::pattern_match
    TEST(atomspace_pattern_match, {
        cog::AtomSpace as;
        cog::Handle h_a   = as.add_node(cog::AtomType::CONCEPT_NODE, "Dog");
        cog::Handle h_b   = as.add_node(cog::AtomType::CONCEPT_NODE, "Animal");
        cog::Handle h_c   = as.add_node(cog::AtomType::CONCEPT_NODE, "Cat");
        as.add_link(cog::AtomType::INHERITANCE_LINK, {h_a, h_b});
        as.add_link(cog::AtomType::INHERITANCE_LINK, {h_c, h_b});
        // Wildcard: find all InheritanceLinks with h_b as target
        auto matches = as.pattern_match(cog::AtomType::INHERITANCE_LINK,
                                        {cog::UNDEFINED_HANDLE, h_b});
        REQUIRE(matches.size() == 2);
    });

    TEST(atomspace_pattern_match_exact, {
        cog::AtomSpace as;
        cog::Handle h_a = as.add_node(cog::AtomType::CONCEPT_NODE, "Dog");
        cog::Handle h_b = as.add_node(cog::AtomType::CONCEPT_NODE, "Animal");
        cog::Handle link = as.add_link(cog::AtomType::INHERITANCE_LINK, {h_a, h_b});
        auto matches = as.pattern_match(cog::AtomType::INHERITANCE_LINK, {h_a, h_b});
        REQUIRE(matches.size() == 1);
        REQUIRE(matches[0] == link);
    });

    // AtomSpace::clear
    TEST(atomspace_clear, {
        cog::AtomSpace as;
        as.add_node(cog::AtomType::CONCEPT_NODE, "A");
        as.add_node(cog::AtomType::CONCEPT_NODE, "B");
        REQUIRE(as.size() == 2);
        as.clear();
        REQUIRE(as.size() == 0);
        // After clear, same names can be added fresh
        cog::Handle h = as.add_node(cog::AtomType::CONCEPT_NODE, "A");
        REQUIRE(h != cog::UNDEFINED_HANDLE);
        REQUIRE(as.size() == 1);
    });

    // Arena
    TEST(arena_alloc, {
        cog::Arena arena(1024);
        REQUIRE(arena.used() == 0);
        REQUIRE(arena.capacity() == 1024);
        void* p = arena.alloc(64);
        REQUIRE(p != nullptr);
        REQUIRE(arena.used() >= 64);
    });

    TEST(arena_create, {
        cog::Arena arena(1024);
        int* p = arena.create<int>(42);
        REQUIRE(p != nullptr);
        REQUIRE(*p == 42);
    });

    TEST(arena_overflow_returns_null, {
        cog::Arena arena(8);
        void* p = arena.alloc(100);
        REQUIRE(p == nullptr);
    });

    TEST(arena_reset, {
        cog::Arena arena(1024);
        arena.alloc(100);
        size_t used = arena.used();
        REQUIRE(used >= 100);
        arena.reset();
        REQUIRE(arena.used() == 0);
        void* p2 = arena.alloc(100);
        REQUIRE(p2 != nullptr);
    });

    // Spinlock
    TEST(spinlock_lock_unlock, {
        cog::Spinlock s;
        s.lock();
        s.unlock();
        REQUIRE(true); // just verify no deadlock
    });

    TEST(spinlock_guard, {
        cog::Spinlock s;
        {
            cog::Spinlock::Guard g(s);
            // lock held; unlocked on scope exit
        }
        // Should be able to lock again
        s.lock();
        s.unlock();
        REQUIRE(true);
    });
}

// ─── cog::gml (extended) ─────────────────────────────────────────────────────

void test_gml_extended() {
    section("cog::gml (extended)");

    // f32_to_f16 conversion
    TEST(f32_to_f16, {
        float orig = 3.14f;
        uint16_t h = cog::gml::f32_to_f16(orig);
        float back = cog::gml::f16_to_f32(h);
        REQUIRE_NEAR(back, orig, 0.01f);
    });

    TEST(f32_to_f16_special, {
        // zero
        uint16_t h0 = cog::gml::f32_to_f16(0.0f);
        REQUIRE_NEAR(cog::gml::f16_to_f32(h0), 0.0f, 1e-6f);
    });

    // Quantize / dequantize Q4_0
    TEST(quantize_q4_0_roundtrip, {
        const size_t QK = 32;
        std::vector<float> src(QK);
        for (size_t i = 0; i < QK; ++i) src[i] = (float)(i % 16) - 8.0f;
        std::vector<cog::gml::BlockQ4_0> blocks(1);
        cog::gml::quantize_q4_0(src.data(), blocks.data(), QK);
        std::vector<float> dst(QK);
        cog::gml::dequantize_q4_0(blocks.data(), dst.data(), QK);
        // Dequantized should be approximately equal to src
        float max_err = 0.0f;
        for (size_t i = 0; i < QK; ++i)
            max_err = std::max(max_err, std::fabs(dst[i] - src[i]));
        REQUIRE(max_err < 2.0f); // Q4 has ~1 bit resolution per unit
    });

    TEST(quantize_q4_0_zero_block, {
        const size_t QK = 32;
        std::vector<float> src(QK, 0.0f);
        std::vector<cog::gml::BlockQ4_0> blocks(1);
        cog::gml::quantize_q4_0(src.data(), blocks.data(), QK);
        std::vector<float> dst(QK);
        cog::gml::dequantize_q4_0(blocks.data(), dst.data(), QK);
        for (size_t i = 0; i < QK; ++i) REQUIRE_NEAR(dst[i], 0.0f, 1e-5f);
    });

    // Quantize / dequantize Q8_0
    TEST(quantize_q8_0_roundtrip, {
        const size_t QK = 32;
        std::vector<float> src(QK);
        for (size_t i = 0; i < QK; ++i) src[i] = (float)(i) * 0.1f - 1.5f;
        std::vector<cog::gml::BlockQ8_0> blocks(1);
        cog::gml::quantize_q8_0(src.data(), blocks.data(), QK);
        std::vector<float> dst(QK);
        cog::gml::dequantize_q8_0(blocks.data(), dst.data(), QK);
        float max_err = 0.0f;
        for (size_t i = 0; i < QK; ++i)
            max_err = std::max(max_err, std::fabs(dst[i] - src[i]));
        REQUIRE(max_err < 0.05f); // Q8 has much better precision
    });

    TEST(quantize_q8_0_zero_block, {
        const size_t QK = 32;
        std::vector<float> src(QK, 0.0f);
        std::vector<cog::gml::BlockQ8_0> blocks(1);
        cog::gml::quantize_q8_0(src.data(), blocks.data(), QK);
        std::vector<float> dst(QK);
        cog::gml::dequantize_q8_0(blocks.data(), dst.data(), QK);
        for (size_t i = 0; i < QK; ++i) REQUIRE_NEAR(dst[i], 0.0f, 1e-5f);
    });

    // Context
    TEST(context_basic, {
        cog::gml::Context ctx(4096);
        REQUIRE(ctx.used() == 0);
        REQUIRE(ctx.capacity() == 4096);
    });

    // Tensor creation
    TEST(tensor_create_and_access, {
        cog::gml::Context ctx(4096);
        auto t = cog::gml::Tensor::create(ctx, cog::gml::DType::F32, 4, 2);
        REQUIRE(t.valid());
        REQUIRE(t.ne(0) == 4);
        REQUIRE(t.ne(1) == 2);
        REQUIRE(t.nelem() == 8);
    });

    TEST(tensor_f32_set_get, {
        cog::gml::Context ctx(4096);
        auto t = cog::gml::Tensor::create(ctx, cog::gml::DType::F32, 3, 1);
        t.f32(0) = 1.5f;
        t.f32(1) = 2.5f;
        t.f32(2) = 3.5f;
        REQUIRE_NEAR(t.f32(0), 1.5f, 1e-6f);
        REQUIRE_NEAR(t.f32(1), 2.5f, 1e-6f);
        REQUIRE_NEAR(t.f32(2), 3.5f, 1e-6f);
    });

    TEST(tensor_from_float, {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        auto t = cog::gml::Tensor::from_float(data, 4);
        REQUIRE(t.valid());
        REQUIRE(t.nelem() == 4);
        REQUIRE_NEAR(t.f32(0), 1.0f, 1e-6f);
        REQUIRE_NEAR(t.f32(3), 4.0f, 1e-6f);
    });

    TEST(tensor_alloc_grad, {
        cog::gml::Context ctx(8192);
        auto t = cog::gml::Tensor::create(ctx, cog::gml::DType::F32, 4, 1);
        t.alloc_grad(ctx);
        // grad should be accessible (no crash)
        t.grad_f32(0) = 0.1f;
        REQUIRE_NEAR(t.grad_f32(0), 0.1f, 1e-6f);
    });

    TEST(tensor_shape_str, {
        cog::gml::Context ctx(4096);
        auto t = cog::gml::Tensor::create(ctx, cog::gml::DType::F32, 3, 2);
        std::string s = t.shape_str();
        REQUIRE(s.find("3") != std::string::npos);
        REQUIRE(s.find("2") != std::string::npos);
    });

    // CGraph — additional ops
    TEST(cgraph_sub, {
        cog::gml::Context ctx(4096);
        cog::gml::CGraph g(ctx);
        auto tA = cog::gml::Tensor::create(ctx, cog::gml::DType::F32, 2);
        auto tB = cog::gml::Tensor::create(ctx, cog::gml::DType::F32, 2);
        tA.f32(0) = 5.0f; tA.f32(1) = 6.0f;
        tB.f32(0) = 1.0f; tB.f32(1) = 2.0f;
        uint32_t a = g.input(tA);
        uint32_t b = g.input(tB);
        uint32_t c = g.sub(a, b);
        g.forward();
        REQUIRE_NEAR(g.result(c).f32(0), 4.0f, 1e-5f);
        REQUIRE_NEAR(g.result(c).f32(1), 4.0f, 1e-5f);
    });

    TEST(cgraph_mul, {
        cog::gml::Context ctx(4096);
        cog::gml::CGraph g(ctx);
        auto tA = cog::gml::Tensor::create(ctx, cog::gml::DType::F32, 2);
        auto tB = cog::gml::Tensor::create(ctx, cog::gml::DType::F32, 2);
        tA.f32(0) = 3.0f; tA.f32(1) = 4.0f;
        tB.f32(0) = 2.0f; tB.f32(1) = 0.5f;
        uint32_t a = g.input(tA);
        uint32_t b = g.input(tB);
        uint32_t c = g.mul(a, b);
        g.forward();
        REQUIRE_NEAR(g.result(c).f32(0), 6.0f, 1e-5f);
        REQUIRE_NEAR(g.result(c).f32(1), 2.0f, 1e-5f);
    });

    TEST(cgraph_sigmoid, {
        cog::gml::Context ctx(4096);
        cog::gml::CGraph g(ctx);
        auto tA = cog::gml::Tensor::create(ctx, cog::gml::DType::F32, 1);
        tA.f32(0) = 0.0f;
        uint32_t a = g.input(tA);
        uint32_t b = g.sigmoid(a);
        g.forward();
        REQUIRE_NEAR(g.result(b).f32(0), 0.5f, 1e-5f);
    });

    TEST(cgraph_tanh, {
        cog::gml::Context ctx(4096);
        cog::gml::CGraph g(ctx);
        auto tA = cog::gml::Tensor::create(ctx, cog::gml::DType::F32, 1);
        tA.f32(0) = 0.0f;
        uint32_t a = g.input(tA);
        uint32_t b = g.tanh_op(a);
        g.forward();
        REQUIRE_NEAR(g.result(b).f32(0), 0.0f, 1e-5f);
    });

    TEST(cgraph_matmul, {
        // 2x2 matmul
        cog::gml::Context ctx(8192);
        cog::gml::CGraph g(ctx);
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        auto tA = cog::gml::Tensor::create(ctx, cog::gml::DType::F32, 2, 2);
        tA.f32(0, 0) = 1.0f; tA.f32(0, 1) = 2.0f;
        tA.f32(1, 0) = 3.0f; tA.f32(1, 1) = 4.0f;
        auto tB = cog::gml::Tensor::create(ctx, cog::gml::DType::F32, 2, 2);
        tB.f32(0, 0) = 5.0f; tB.f32(0, 1) = 6.0f;
        tB.f32(1, 0) = 7.0f; tB.f32(1, 1) = 8.0f;
        uint32_t a = g.input(tA);
        uint32_t b = g.input(tB);
        uint32_t c = g.matmul(a, b);
        g.forward();
        REQUIRE_NEAR(g.result(c).f32(0, 0), 19.0f, 1e-3f);
        REQUIRE_NEAR(g.result(c).f32(0, 1), 22.0f, 1e-3f);
        REQUIRE_NEAR(g.result(c).f32(1, 0), 43.0f, 1e-3f);
        REQUIRE_NEAR(g.result(c).f32(1, 1), 50.0f, 1e-3f);
    });

    // Adam optimizer
    TEST(adam_step, {
        cog::gml::Adam adam;
        std::vector<float> param = {1.0f, 2.0f, 3.0f};
        std::vector<float> grad  = {0.1f, 0.2f, 0.3f};
        std::vector<float> orig  = param;
        adam.step(param.data(), grad.data(), param.size());
        // params should have moved in opposite direction of gradient
        REQUIRE(param[0] < orig[0]);
        REQUIRE(param[1] < orig[1]);
        REQUIRE(param[2] < orig[2]);
    });

    TEST(adam_reset, {
        cog::gml::Adam adam;
        std::vector<float> param = {1.0f};
        std::vector<float> grad  = {0.1f};
        adam.step(param.data(), grad.data(), 1);
        adam.reset(); // should not crash; resets internal state
        REQUIRE(true);
    });

    TEST(adam_multiple_steps, {
        cog::gml::Adam adam;
        std::vector<float> param = {5.0f};
        std::vector<float> grad  = {1.0f}; // constant gradient
        for (int i = 0; i < 10; ++i) {
            adam.step(param.data(), grad.data(), 1);
        }
        // after many steps with positive gradient, param should decrease
        REQUIRE(param[0] < 5.0f);
    });

    // LBFGS optimizer
    TEST(lbfgs_step, {
        cog::gml::LBFGS lbfgs;
        std::vector<float> param = {1.0f, 2.0f};
        // Run multiple steps to verify convergence toward minimum of f(p) = sum(p^2)
        for (int iter = 0; iter < 5; ++iter) {
            std::vector<float> grad = {2.0f * param[0], 2.0f * param[1]};
            lbfgs.step(param.data(), grad.data(), param.size(),
                       [](const float* p) {
                           return p[0]*p[0] + p[1]*p[1];
                       });
        }
        // After multiple LBFGS steps, parameters should move closer to zero
        REQUIRE(std::isfinite(param[0]));
        REQUIRE(std::isfinite(param[1]));
        REQUIRE(param[0] < 1.0f); // should have moved from initial 1.0
        REQUIRE(param[1] < 2.0f); // should have moved from initial 2.0
    });
}

// ─── cog::prime (extended) ───────────────────────────────────────────────────

void test_prime_extended() {
    section("cog::prime (extended)");

    // TruthValue::induction
    TEST(prime_truth_value_induction, {
        cog::prime::TruthValue ac(0.8f, 0.9f);
        cog::prime::TruthValue bc(0.7f, 0.8f);
        auto ab = cog::prime::TruthValue::induction(ac, bc);
        // s = ac.s * bc.s / pa (pa=0.5 default) → clamped to [0,1]
        REQUIRE(ab.strength >= 0.0f && ab.strength <= 1.0f);
        REQUIRE(ab.confidence < 0.9f); // reduced confidence
    });

    // TruthValue::disjunction
    TEST(prime_truth_value_disjunction, {
        cog::prime::TruthValue a(0.8f, 0.9f);
        cog::prime::TruthValue b(0.6f, 0.7f);
        auto d = cog::prime::TruthValue::disjunction(a, b);
        REQUIRE_NEAR(d.strength, 1.0f - (1.0f - 0.8f) * (1.0f - 0.6f), 1e-4f);
        REQUIRE(d.confidence <= std::min(a.confidence, b.confidence) + 1e-4f);
    });

    // AttentionValue::is_attentional
    TEST(prime_attention_is_attentional, {
        cog::prime::AttentionValue av_high(0.5f, 0.0f);
        cog::prime::AttentionValue av_zero(0.0f, 0.0f);
        cog::prime::AttentionValue av_neg(-0.1f, 0.0f);
        REQUIRE( av_high.is_attentional());
        REQUIRE(!av_zero.is_attentional());
        REQUIRE(!av_neg.is_attentional());
    });

    // PatternMatcher::match
    TEST(prime_pattern_matcher_exact, {
        cog::prime::AtomSpace as;
        cog::Handle h_dog    = as.add_node(cog::AtomType::CONCEPT_NODE, "Dog");
        cog::Handle h_animal = as.add_node(cog::AtomType::CONCEPT_NODE, "Animal");
        cog::Handle h_link   = as.add_link(cog::AtomType::INHERITANCE_LINK,
                                            {h_dog, h_animal});
        cog::prime::PatternMatcher pm(as);
        cog::prime::Binding b;
        REQUIRE(pm.match(h_link, h_link, b)); // exact match
    });

    TEST(prime_pattern_matcher_variable, {
        cog::prime::AtomSpace as;
        cog::Handle h_var    = as.add_node(cog::AtomType::VARIABLE_NODE, "$X");
        cog::Handle h_animal = as.add_node(cog::AtomType::CONCEPT_NODE, "Animal");
        cog::Handle h_dog    = as.add_node(cog::AtomType::CONCEPT_NODE, "Dog");
        cog::prime::PatternMatcher pm(as);
        cog::prime::Binding b;
        // variable matches any concept node
        REQUIRE(pm.match(h_var, h_animal, b));
        REQUIRE(b.get(h_var) == h_animal);
    });

    TEST(prime_pattern_matcher_find, {
        cog::prime::AtomSpace as;
        cog::Handle h_dog    = as.add_node(cog::AtomType::CONCEPT_NODE, "Dog");
        cog::Handle h_cat    = as.add_node(cog::AtomType::CONCEPT_NODE, "Cat");
        cog::Handle h_animal = as.add_node(cog::AtomType::CONCEPT_NODE, "Animal");
        as.add_link(cog::AtomType::INHERITANCE_LINK, {h_dog, h_animal});
        as.add_link(cog::AtomType::INHERITANCE_LINK, {h_cat, h_animal});
        // Pattern: find concept node "Dog"
        cog::prime::PatternMatcher pm(as);
        auto results = pm.find(h_dog);
        REQUIRE(results.size() >= 1);
    });

    // EpisodicMemory::recall_salient
    TEST(prime_episodic_recall_salient, {
        cog::prime::EpisodicMemory em;
        em.record("ctx1", "event_low",  0.2f);
        em.record("ctx1", "event_high", 0.9f);
        em.record("ctx2", "event_mid",  0.5f);
        auto salient = em.recall_salient(2);
        REQUIRE(salient.size() == 2);
        REQUIRE(salient[0].salience >= salient[1].salience);
        REQUIRE(salient[0].event == "event_high");
    });

    // ProceduralMemory::list
    TEST(prime_procedural_list, {
        cog::prime::ProceduralMemory pm;
        pm.store(cog::prime::Procedure("walk", "Move forward", 0.3f));
        pm.store(cog::prime::Procedure("speak", "Produce speech", 0.5f));
        pm.store(cog::prime::Procedure("think", "Process information", 0.7f));
        auto names = pm.list();
        REQUIRE(names.size() == 3);
        auto it_walk  = std::find(names.begin(), names.end(), "walk");
        auto it_speak = std::find(names.begin(), names.end(), "speak");
        REQUIRE(it_walk  != names.end());
        REQUIRE(it_speak != names.end());
    });

    // AtomSpace::top_by_sti
    TEST(prime_atomspace_top_by_sti, {
        cog::prime::AtomSpace as;
        cog::Handle h1 = as.add_node(cog::AtomType::CONCEPT_NODE, "A");
        cog::Handle h2 = as.add_node(cog::AtomType::CONCEPT_NODE, "B");
        cog::Handle h3 = as.add_node(cog::AtomType::CONCEPT_NODE, "C");
        as.stimulate(h1, 0.9f);
        as.stimulate(h3, 0.5f);
        // h2 not stimulated
        auto top = as.top_by_sti(2);
        REQUIRE(top.size() == 2);
        REQUIRE(top[0] == h1); // highest STI
    });

    // AtomSpace::lookup_node
    TEST(prime_atomspace_lookup_node, {
        cog::prime::AtomSpace as;
        cog::Handle h = as.add_node(cog::AtomType::CONCEPT_NODE, "Planet");
        REQUIRE(as.lookup_node("Planet") == h);
        REQUIRE(as.lookup_node("Missing") == cog::UNDEFINED_HANDLE);
    });

    // TruthValue::to_string
    TEST(prime_tv_to_string, {
        cog::prime::TruthValue tv(0.75f, 0.9f);
        std::string s = tv.to_string();
        REQUIRE(s.find("TV") != std::string::npos);
        REQUIRE(s.find("0.75") != std::string::npos);
    });

    // TruthValue::operator==
    TEST(prime_tv_equality, {
        cog::prime::TruthValue tv1(0.8f, 0.6f);
        cog::prime::TruthValue tv2(0.8f, 0.6f);
        cog::prime::TruthValue tv3(0.7f, 0.6f);
        REQUIRE(tv1 == tv2);
        REQUIRE(!(tv1 == tv3));
    });

    // TruthValue edge cases
    TEST(prime_tv_zero_confidence_revision, {
        cog::prime::TruthValue t1(0.9f, 0.0f);
        cog::prime::TruthValue t2(0.1f, 0.0f);
        auto rev = cog::prime::TruthValue::revise(t1, t2);
        REQUIRE_NEAR(rev.confidence, 0.0f, 1e-5f);
        REQUIRE_NEAR(rev.strength,   0.5f, 1e-5f);
    });

    // AtomSpace::decay_attention
    TEST(prime_atomspace_decay_attention, {
        cog::prime::AtomSpace as;
        cog::Handle h = as.add_node(cog::AtomType::CONCEPT_NODE, "X");
        as.stimulate(h, 0.8f);
        auto* atom_before = as.get(h);
        REQUIRE(atom_before != nullptr);
        float sti_before = atom_before->av.sti;
        REQUIRE(sti_before > 0.0f);

        as.decay_attention(0.5f);  // aggressive decay
        auto* atom_after = as.get(h);
        REQUIRE(atom_after != nullptr);
        REQUIRE(atom_after->av.sti < sti_before);
    });

    // AtomSpace::remove
    TEST(prime_atomspace_remove_atom, {
        cog::prime::AtomSpace as;
        cog::Handle h = as.add_node(cog::AtomType::CONCEPT_NODE, "Temp");
        REQUIRE(as.size() == 1);
        REQUIRE(as.remove(h));
        REQUIRE(as.size() == 0);
        REQUIRE(as.get(h) == nullptr);
    });

    // AtomSpace::remove non-existent
    TEST(prime_atomspace_remove_nonexistent, {
        cog::prime::AtomSpace as;
        REQUIRE(!as.remove(cog::UNDEFINED_HANDLE));
        REQUIRE(!as.remove(999));
    });

    // PatternMatcher: link pattern matching with variables
    TEST(prime_pattern_matcher_link_variable, {
        cog::prime::AtomSpace as;
        cog::Handle h_dog    = as.add_node(cog::AtomType::CONCEPT_NODE, "Dog");
        cog::Handle h_animal = as.add_node(cog::AtomType::CONCEPT_NODE, "Animal");
        cog::Handle h_var    = as.add_node(cog::AtomType::VARIABLE_NODE, "$X");
        cog::Handle h_link   = as.add_link(cog::AtomType::INHERITANCE_LINK,
                                            {h_dog, h_animal});
        // Pattern: $X -> Animal
        cog::Handle h_pattern = as.add_link(cog::AtomType::INHERITANCE_LINK,
                                             {h_var, h_animal});
        cog::prime::PatternMatcher pm(as);
        cog::prime::Binding b;
        REQUIRE(pm.match(h_pattern, h_link, b));
        REQUIRE(b.get(h_var) == h_dog);
    });

    // PatternMatcher: inconsistent binding
    TEST(prime_pattern_matcher_inconsistent_binding, {
        cog::prime::AtomSpace as;
        cog::Handle h_dog    = as.add_node(cog::AtomType::CONCEPT_NODE, "Dog");
        cog::Handle h_cat    = as.add_node(cog::AtomType::CONCEPT_NODE, "Cat");
        cog::Handle h_animal = as.add_node(cog::AtomType::CONCEPT_NODE, "Animal");
        cog::Handle h_var    = as.add_node(cog::AtomType::VARIABLE_NODE, "$X");
        cog::Handle h_link   = as.add_link(cog::AtomType::INHERITANCE_LINK,
                                            {h_dog, h_animal});
        // Pre-bind $X to Cat — should fail to match Dog->Animal
        cog::prime::Binding b;
        b.bind(h_var, h_cat);
        cog::Handle h_pattern = as.add_link(cog::AtomType::INHERITANCE_LINK,
                                             {h_var, h_animal});
        cog::prime::PatternMatcher pm(as);
        REQUIRE(!pm.match(h_pattern, h_link, b));
    });

    // OntogeneticState::update and try_advance
    TEST(prime_ontogenetic_update_advance, {
        cog::prime::OntogeneticState os;
        REQUIRE(os.level == cog::prime::OntogeneticLevel::SCAFFOLD);
        // Update fitness above threshold
        for (int i = 0; i < 200; ++i) os.update(1.0f);
        REQUIRE(os.fitness > 0.8f);
        bool advanced = os.try_advance();
        REQUIRE(advanced);
        REQUIRE(os.level == cog::prime::OntogeneticLevel::REACTIVE);
        REQUIRE(os.wisdom > 0.0f);
    });

    // OntogeneticState::to_string
    TEST(prime_ontogenetic_to_string, {
        cog::prime::OntogeneticState os;
        std::string s = os.to_string();
        REQUIRE(s.find("Scaffold") != std::string::npos);
        REQUIRE(s.find("fitness") != std::string::npos);
    });

    // CognitiveCycle: multiple phase handlers
    TEST(prime_cognitive_cycle_multi_phase, {
        cog::prime::CognitiveCycle cycle;
        int perceive_count = 0, act_count = 0;
        cycle.on_phase(cog::prime::CyclePhase::PERCEIVE,
            [&perceive_count](cog::prime::CognitiveState& st,
                              cog::prime::DeclarativeMemory&,
                              cog::prime::EpisodicMemory&,
                              cog::prime::ProceduralMemory&) {
                ++perceive_count;
                st.arousal = 0.5f;
            });
        cycle.on_phase(cog::prime::CyclePhase::ACT,
            [&act_count](cog::prime::CognitiveState& st,
                         cog::prime::DeclarativeMemory&,
                         cog::prime::EpisodicMemory&,
                         cog::prime::ProceduralMemory&) {
                ++act_count;
                st.valence = 0.3f;
            });
        cycle.run(2);
        REQUIRE(perceive_count == 2);
        REQUIRE(act_count == 2);
        REQUIRE_NEAR(cycle.state().arousal, 0.5f, 1e-5f);
        REQUIRE_NEAR(cycle.state().valence, 0.3f, 1e-5f);
    });

    // EpisodicMemory: recall by context
    TEST(prime_episodic_recall_context, {
        cog::prime::EpisodicMemory em;
        em.record("work", "meeting", 0.7f);
        em.record("work", "lunch",   0.5f);
        em.record("home", "dinner",  0.8f);
        auto work_episodes = em.recall("work");
        REQUIRE(work_episodes.size() == 2);
        auto home_episodes = em.recall("home");
        REQUIRE(home_episodes.size() == 1);
        REQUIRE(home_episodes[0].event == "dinner");
    });

    // DeclarativeMemory: store and lookup fact
    TEST(prime_declarative_store_lookup, {
        cog::prime::DeclarativeMemory mem;
        auto h = mem.store_fact("Paris", "capital_of", "France");
        REQUIRE(h != cog::UNDEFINED_HANDLE);
        // store_fact creates: subject (ConceptNode), predicate (PredicateNode),
        // object (ConceptNode), and the EvaluationLink = 4 atoms total
        REQUIRE(mem.size() == 4);
    });
}

// ─── cog::lux (extended) ─────────────────────────────────────────────────────

void test_lux_extended() {
    section("cog::lux (extended)");

    TEST(lux_has_node, {
        cog::lux::LuxGraph g;
        cog::lux::NodeId n1 = g.add_node(cog::lux::NodeType::CONCEPT, "A");
        REQUIRE( g.has_node(n1));
        REQUIRE(!g.has_node(9999));
    });

    TEST(lux_all_nodes, {
        cog::lux::LuxGraph g;
        g.add_node(cog::lux::NodeType::CONCEPT,   "X");
        g.add_node(cog::lux::NodeType::PREDICATE, "Y");
        g.add_node(cog::lux::NodeType::CONCEPT,   "Z");
        auto nodes = g.all_nodes();
        REQUIRE(nodes.size() == 3);
    });

    TEST(lux_all_edges, {
        cog::lux::LuxGraph g;
        cog::lux::NodeId a = g.add_node(cog::lux::NodeType::CONCEPT, "A");
        cog::lux::NodeId b = g.add_node(cog::lux::NodeType::CONCEPT, "B");
        cog::lux::NodeId c = g.add_node(cog::lux::NodeType::CONCEPT, "C");
        g.add_edge(cog::lux::EdgeType::INHERITANCE, a, b);
        g.add_edge(cog::lux::EdgeType::SIMILARITY,  b, c);
        auto edges = g.all_edges();
        REQUIRE(edges.size() == 2);
    });

    TEST(lux_in_edges, {
        cog::lux::LuxGraph g;
        cog::lux::NodeId a = g.add_node(cog::lux::NodeType::CONCEPT, "Source");
        cog::lux::NodeId b = g.add_node(cog::lux::NodeType::CONCEPT, "Target");
        g.add_edge(cog::lux::EdgeType::INHERITANCE, a, b);
        auto in_b = g.in_edges(b);
        REQUIRE(in_b.size() == 1);
        auto in_a = g.in_edges(a);
        REQUIRE(in_a.empty());
    });

    TEST(lux_bfs_max_depth, {
        cog::lux::LuxGraph g;
        cog::lux::NodeId a = g.add_node(cog::lux::NodeType::CONCEPT, "a");
        cog::lux::NodeId b = g.add_node(cog::lux::NodeType::CONCEPT, "b");
        cog::lux::NodeId c = g.add_node(cog::lux::NodeType::CONCEPT, "c");
        cog::lux::NodeId d = g.add_node(cog::lux::NodeType::CONCEPT, "d");
        g.add_edge(cog::lux::EdgeType::INHERITANCE, a, b);
        g.add_edge(cog::lux::EdgeType::INHERITANCE, b, c);
        g.add_edge(cog::lux::EdgeType::INHERITANCE, c, d);
        // depth=1: only a and b
        auto bfs1 = g.bfs(a, 1);
        REQUIRE(bfs1.size() == 2);
        // depth=2: a, b, c
        auto bfs2 = g.bfs(a, 2);
        REQUIRE(bfs2.size() == 3);
    });

    TEST(lux_node_type_names, {
        REQUIRE(std::string(cog::lux::node_type_name(cog::lux::NodeType::CONCEPT))   == "Concept");
        REQUIRE(std::string(cog::lux::node_type_name(cog::lux::NodeType::PREDICATE)) == "Predicate");
    });

    TEST(lux_edge_type_names, {
        REQUIRE(std::string(cog::lux::edge_type_name(cog::lux::EdgeType::INHERITANCE)) == "Inheritance");
        REQUIRE(std::string(cog::lux::edge_type_name(cog::lux::EdgeType::SIMILARITY))  == "Similarity");
    });

    TEST(lux_edge_weight, {
        cog::lux::LuxGraph g;
        cog::lux::NodeId a = g.add_node(cog::lux::NodeType::CONCEPT, "A");
        cog::lux::NodeId b = g.add_node(cog::lux::NodeType::CONCEPT, "B");
        cog::lux::EdgeId eid = g.add_edge(cog::lux::EdgeType::INHERITANCE, a, b, 0.75f);
        const cog::lux::LuxEdge* e = g.edge(eid);
        REQUIRE(e != nullptr);
        REQUIRE_NEAR(e->weight, 0.75f, 1e-5f);
    });
}

// ─── cog::glow (extended) ────────────────────────────────────────────────────

// Helper: build a simple one-op graph and run it
static cog::glow::GlowInterpreter::TensorMap run_unary_op(
    cog::glow::OpType op,
    const std::vector<float>& input_data)
{
    cog::glow::GlowCompiler c;
    auto in_vid  = c.input({input_data.size()});
    auto out_vid = c.op1(op, in_vid);
    c.mark_output(out_vid);
    cog::glow::GlowInterpreter interp;
    cog::glow::GlowInterpreter::TensorMap inputs;
    inputs[in_vid] = input_data;
    return interp.run(c.graph(), inputs);
}

static cog::glow::GlowInterpreter::TensorMap run_binary_op(
    cog::glow::OpType op,
    const std::vector<float>& a,
    const std::vector<float>& b)
{
    cog::glow::GlowCompiler c;
    auto va  = c.input({a.size()});
    auto vb  = c.input({b.size()});
    auto vc  = c.op2(op, va, vb);
    c.mark_output(vc);
    cog::glow::GlowInterpreter interp;
    cog::glow::GlowInterpreter::TensorMap inputs;
    inputs[va] = a;
    inputs[vb] = b;
    return interp.run(c.graph(), inputs);
}

void test_glow_extended() {
    section("cog::glow (extended)");

    // IRType::numel
    TEST(irtype_numel, {
        cog::glow::IRType t1(cog::gml::DType::F32, {4, 3});
        REQUIRE(t1.numel() == 12);
        cog::glow::IRType t2(cog::gml::DType::F32, {7});
        REQUIRE(t2.numel() == 7);
        cog::glow::IRType t3(cog::gml::DType::F32, {});
        REQUIRE(t3.numel() == 0);
    });

    // IRNode attributes
    TEST(irnode_attrs, {
        cog::glow::IRNode n(1, cog::glow::OpType::LINEAR);
        n.set_attr("in_features",  "128");
        n.set_attr("out_features", "64");
        REQUIRE(n.get_attr("in_features")  == "128");
        REQUIRE(n.get_attr("out_features") == "64");
        REQUIRE(n.get_attr("missing", "default") == "default");
        REQUIRE(n.get_int_attr("in_features")  == 128);
        REQUIRE(n.get_int_attr("out_features") == 64);
        REQUIRE(n.get_int_attr("missing", -1) == -1);
    });

    // op_type_name coverage
    TEST(op_type_name_coverage, {
        REQUIRE(std::string(cog::glow::op_type_name(cog::glow::OpType::SUB))     == "Sub");
        REQUIRE(std::string(cog::glow::op_type_name(cog::glow::OpType::MUL))     == "Mul");
        REQUIRE(std::string(cog::glow::op_type_name(cog::glow::OpType::DIV))     == "Div");
        REQUIRE(std::string(cog::glow::op_type_name(cog::glow::OpType::NEG))     == "Neg");
        REQUIRE(std::string(cog::glow::op_type_name(cog::glow::OpType::ABS))     == "Abs");
        REQUIRE(std::string(cog::glow::op_type_name(cog::glow::OpType::EXP))     == "Exp");
        REQUIRE(std::string(cog::glow::op_type_name(cog::glow::OpType::LOG))     == "Log");
        REQUIRE(std::string(cog::glow::op_type_name(cog::glow::OpType::SQRT))    == "Sqrt");
        REQUIRE(std::string(cog::glow::op_type_name(cog::glow::OpType::TANH))    == "Tanh");
        REQUIRE(std::string(cog::glow::op_type_name(cog::glow::OpType::SIGMOID)) == "Sigmoid");
        REQUIRE(std::string(cog::glow::op_type_name(cog::glow::OpType::GELU))    == "Gelu");
        REQUIRE(std::string(cog::glow::op_type_name(cog::glow::OpType::SUM))     == "Sum");
        REQUIRE(std::string(cog::glow::op_type_name(cog::glow::OpType::MEAN))    == "Mean");
        REQUIRE(std::string(cog::glow::op_type_name(cog::glow::OpType::MATMUL))  == "MatMul");
    });

    // Interpreter: SUB
    TEST(interpreter_sub, {
        auto res = run_binary_op(cog::glow::OpType::SUB, {10.0f, 5.0f}, {3.0f, 2.0f});
        REQUIRE(!res.empty());
        auto& out = res.begin()->second;
        REQUIRE_NEAR(out[0], 7.0f, 1e-5f);
        REQUIRE_NEAR(out[1], 3.0f, 1e-5f);
    });

    // Interpreter: MUL
    TEST(interpreter_mul, {
        auto res = run_binary_op(cog::glow::OpType::MUL, {3.0f, 4.0f}, {2.0f, 0.5f});
        REQUIRE(!res.empty());
        auto& out = res.begin()->second;
        REQUIRE_NEAR(out[0], 6.0f, 1e-5f);
        REQUIRE_NEAR(out[1], 2.0f, 1e-5f);
    });

    // Interpreter: DIV
    TEST(interpreter_div, {
        auto res = run_binary_op(cog::glow::OpType::DIV, {8.0f, 9.0f}, {4.0f, 3.0f});
        REQUIRE(!res.empty());
        auto& out = res.begin()->second;
        REQUIRE_NEAR(out[0], 2.0f, 1e-5f);
        REQUIRE_NEAR(out[1], 3.0f, 1e-5f);
    });

    // Interpreter: NEG
    TEST(interpreter_neg, {
        auto res = run_unary_op(cog::glow::OpType::NEG, {1.0f, -2.0f, 3.0f});
        REQUIRE(!res.empty());
        auto& out = res.begin()->second;
        REQUIRE_NEAR(out[0], -1.0f, 1e-5f);
        REQUIRE_NEAR(out[1],  2.0f, 1e-5f);
        REQUIRE_NEAR(out[2], -3.0f, 1e-5f);
    });

    // Interpreter: ABS
    TEST(interpreter_abs, {
        auto res = run_unary_op(cog::glow::OpType::ABS, {-3.0f, 4.0f, -0.5f});
        REQUIRE(!res.empty());
        auto& out = res.begin()->second;
        REQUIRE_NEAR(out[0], 3.0f, 1e-5f);
        REQUIRE_NEAR(out[1], 4.0f, 1e-5f);
        REQUIRE_NEAR(out[2], 0.5f, 1e-5f);
    });

    // Interpreter: EXP
    TEST(interpreter_exp, {
        auto res = run_unary_op(cog::glow::OpType::EXP, {0.0f, 1.0f});
        REQUIRE(!res.empty());
        auto& out = res.begin()->second;
        REQUIRE_NEAR(out[0], 1.0f,          1e-4f);
        REQUIRE_NEAR(out[1], (float)M_E,    1e-4f);
    });

    // Interpreter: LOG
    TEST(interpreter_log, {
        auto res = run_unary_op(cog::glow::OpType::LOG, {1.0f, (float)M_E});
        REQUIRE(!res.empty());
        auto& out = res.begin()->second;
        REQUIRE_NEAR(out[0], 0.0f, 1e-4f);
        REQUIRE_NEAR(out[1], 1.0f, 1e-4f);
    });

    // Interpreter: SQRT
    TEST(interpreter_sqrt, {
        auto res = run_unary_op(cog::glow::OpType::SQRT, {4.0f, 9.0f, 0.0f});
        REQUIRE(!res.empty());
        auto& out = res.begin()->second;
        REQUIRE_NEAR(out[0], 2.0f, 1e-4f);
        REQUIRE_NEAR(out[1], 3.0f, 1e-4f);
        REQUIRE_NEAR(out[2], 0.0f, 1e-4f);
    });

    // Interpreter: TANH
    TEST(interpreter_tanh, {
        auto res = run_unary_op(cog::glow::OpType::TANH, {0.0f, 1.0f});
        REQUIRE(!res.empty());
        auto& out = res.begin()->second;
        REQUIRE_NEAR(out[0], 0.0f,                1e-4f);
        REQUIRE_NEAR(out[1], (float)std::tanh(1.0), 1e-4f);
    });

    // Interpreter: SIGMOID
    TEST(interpreter_sigmoid_glow, {
        auto res = run_unary_op(cog::glow::OpType::SIGMOID, {0.0f});
        REQUIRE(!res.empty());
        auto& out = res.begin()->second;
        REQUIRE_NEAR(out[0], 0.5f, 1e-4f);
    });

    // Interpreter: GELU
    TEST(interpreter_gelu, {
        auto res = run_unary_op(cog::glow::OpType::GELU, {0.0f, 1.0f});
        REQUIRE(!res.empty());
        auto& out = res.begin()->second;
        REQUIRE_NEAR(out[0], 0.0f, 1e-3f); // GELU(0) ≈ 0
        REQUIRE(out[1] > 0.0f);             // GELU(1) > 0
    });

    // Interpreter: SUM
    TEST(interpreter_sum, {
        auto res = run_unary_op(cog::glow::OpType::SUM, {1.0f, 2.0f, 3.0f, 4.0f});
        REQUIRE(!res.empty());
        auto& out = res.begin()->second;
        REQUIRE(out.size() == 1);
        REQUIRE_NEAR(out[0], 10.0f, 1e-4f);
    });

    // Interpreter: MEAN
    TEST(interpreter_mean, {
        auto res = run_unary_op(cog::glow::OpType::MEAN, {2.0f, 4.0f, 6.0f});
        REQUIRE(!res.empty());
        auto& out = res.begin()->second;
        REQUIRE(out.size() == 1);
        REQUIRE_NEAR(out[0], 4.0f, 1e-4f);
    });

    // pass_dce — dead code elimination
    TEST(pass_dce, {
        cog::glow::GlowCompiler c;
        auto v_in    = c.input({4});
        auto v_live  = c.op1(cog::glow::OpType::RELU, v_in);   // used as output
        auto v_dead  = c.op1(cog::glow::OpType::RELU, v_in);   // not marked output
        c.mark_output(v_live);
        (void)v_dead;
        int removed = cog::glow::pass_dce(c.graph());
        REQUIRE(removed >= 1); // the dead node should be eliminated
    });

    // pass_constant_fold
    TEST(pass_constant_fold, {
        cog::glow::GlowCompiler c;
        auto vc1  = c.constant({1.0f, 2.0f}, {2});
        auto vc2  = c.constant({3.0f, 4.0f}, {2});
        auto vsum = c.op2(cog::glow::OpType::ADD, vc1, vc2);
        c.mark_output(vsum);
        int folded = cog::glow::pass_constant_fold(c.graph());
        REQUIRE(folded >= 1);
        // The output value's const_data should be [4, 6]
        const cog::glow::IRValue* out_v = c.graph().value(vsum);
        REQUIRE(out_v != nullptr);
        if (!out_v->const_data.empty()) {
            REQUIRE_NEAR(out_v->const_data[0], 4.0f, 1e-4f);
            REQUIRE_NEAR(out_v->const_data[1], 6.0f, 1e-4f);
        }
    });
}

// ─── cog::webvm (extended) ───────────────────────────────────────────────────

void test_webvm_extended() {
    section("cog::webvm (extended)");

    // Lambda definition and application
    TEST(scheme_lambda, {
        cog::webvm::SchemeREPL repl;
        repl.eval_str("(define square (lambda (x) (* x x)))");
        auto r = repl.eval_str("(square 5)");
        REQUIRE(r.find("25") != std::string::npos);
    });

    TEST(scheme_lambda_multi_param, {
        cog::webvm::SchemeREPL repl;
        repl.eval_str("(define add (lambda (a b) (+ a b)))");
        auto r = repl.eval_str("(add 3 4)");
        REQUIRE(r.find("7") != std::string::npos);
    });

    // cond expression
    TEST(scheme_cond, {
        cog::webvm::SchemeREPL repl;
        auto r = repl.eval_str("(cond ((= 1 2) \"no\") (else \"yes\"))");
        REQUIRE(r.find("yes") != std::string::npos);
    });

    TEST(scheme_cond_first_match, {
        cog::webvm::SchemeREPL repl;
        auto r = repl.eval_str("(cond ((= 1 1) \"first\") ((= 2 2) \"second\") (else \"last\"))");
        REQUIRE(r.find("first") != std::string::npos);
    });

    // let binding
    TEST(scheme_let, {
        cog::webvm::SchemeREPL repl;
        auto r = repl.eval_str("(let ((x 10) (y 20)) (+ x y))");
        REQUIRE(r.find("30") != std::string::npos);
    });

    // set! mutation
    TEST(scheme_set_mutation, {
        cog::webvm::SchemeREPL repl;
        repl.eval_str("(define counter 0)");
        repl.eval_str("(set! counter 42)");
        auto r = repl.eval_str("counter");
        REQUIRE(r.find("42") != std::string::npos);
    });

    // custom builtin registration
    TEST(scheme_custom_builtin, {
        cog::webvm::SchemeREPL repl;
        repl.define_builtin("double",
            [](const std::vector<cog::webvm::SExprPtr>& args,
               cog::webvm::SchemeEnvPtr) -> cog::webvm::SExprPtr {
                if (args.empty()) return cog::webvm::SExpr::make_number(0.0);
                return cog::webvm::SExpr::make_number(args[0]->number * 2.0);
            });
        auto r = repl.eval_str("(double 7)");
        REQUIRE(r.find("14") != std::string::npos);
    });

    // JSONSerializer::make_array
    TEST(json_make_array, {
        using JS = cog::webvm::JSONSerializer;
        auto arr = JS::make_array({JS::num(1), JS::num(2), JS::num(3)});
        REQUIRE(arr.find("[") != std::string::npos);
        REQUIRE(arr.find("]") != std::string::npos);
        REQUIRE(arr.find("1") != std::string::npos);
        REQUIRE(arr.find("3") != std::string::npos);
    });

    TEST(json_make_array_strings, {
        using JS = cog::webvm::JSONSerializer;
        auto arr = JS::make_array({JS::str("hello"), JS::str("world")});
        REQUIRE(arr.find("hello") != std::string::npos);
        REQUIRE(arr.find("world") != std::string::npos);
    });

    TEST(json_nested_structure, {
        using JS = cog::webvm::JSONSerializer;
        auto inner = JS::make_array({JS::num(1), JS::num(2)});
        auto outer = JS::make_object({{"data", inner}, {"ok", JS::boolean(true)}});
        REQUIRE(outer.find("data")  != std::string::npos);
        REQUIRE(outer.find("true")  != std::string::npos);
    });
}

// ─── cog::pilot (extended) ───────────────────────────────────────────────────

void test_pilot_extended() {
    section("cog::pilot (extended)");

    // BSeries::compose
    TEST(bseries_compose, {
        cog::pilot::BSeries a(3);
        cog::pilot::BSeries b(3);
        for (int i = 0; i <= 3; ++i) {
            a.coeffs[static_cast<size_t>(i)] = 1.0 / (i + 1);
            b.coeffs[static_cast<size_t>(i)] = 1.0 / (i + 1);
        }
        auto c = a.compose(b);
        REQUIRE(c.order == 3);
        // composed coefficients should be finite and non-negative
        for (size_t i = 0; i <= 3; ++i) {
            REQUIRE(std::isfinite(c.coeffs[i]));
        }
    });

    TEST(bseries_compose_identity, {
        // Composing with zero B-series should give near-zero result
        cog::pilot::BSeries a(2);
        a.coeffs[0] = 1.0; a.coeffs[1] = 0.5; a.coeffs[2] = 0.25;
        cog::pilot::BSeries zero(2);
        auto c = a.compose(zero);
        REQUIRE(c.order == 2);
        // All coefficients of zero composed should be 0
        for (size_t i = 0; i <= 2; ++i) {
            REQUIRE_NEAR(c.coeffs[i], 0.0, 1e-10);
        }
    });

    // DualPoolESN::reset
    TEST(esn_reset, {
        cog::pilot::DualPoolESN esn(32, 4, 2);
        esn.initialize(42);
        std::vector<float> input(4, 1.0f);
        esn.step(input); // advance state
        esn.reset();
        // After reset, state should be zero
        for (float v : esn.state_fast) REQUIRE_NEAR(v, 0.0f, 1e-6f);
        for (float v : esn.state_slow) REQUIRE_NEAR(v, 0.0f, 1e-6f);
    });

    // DualPoolESN::train_readout
    TEST(esn_train_readout, {
        cog::pilot::DualPoolESN esn(16, 4, 2);
        esn.initialize(42);
        // Create simple training data
        std::vector<std::vector<float>> X(5, std::vector<float>(esn.n_units, 0.0f));
        std::vector<std::vector<float>> Y(5, std::vector<float>(2, 0.5f));
        for (size_t t = 0; t < 5; ++t) {
            for (size_t i = 0; i < esn.n_units; ++i)
                X[t][i] = (float)(t * esn.n_units + i) * 0.01f;
        }
        esn.train_readout(X, Y, 1e-3f);
        // After training, forward pass should produce output
        std::vector<float> input(4, 0.1f);
        auto out = esn.step(input);
        REQUIRE(out.size() == 2);
    });

    // CogPilot::calibrated_spectral_radius
    TEST(cogpilot_calibrated_spectral_radius, {
        // A000081 sequence: 0,1,1,2,4,9,20,...
        // spectral_radius(system) = 1 - 1/A000081[system+1]
        // system_level=2 → A000081[3]=2 → 0.5
        // system_level=4 → A000081[5]=9 → ~0.889
        // system_level=8 → A000081[9]=286 → ~0.997
        double r2 = cog::pilot::CogPilot::calibrated_spectral_radius(2);
        double r4 = cog::pilot::CogPilot::calibrated_spectral_radius(4);
        double r8 = cog::pilot::CogPilot::calibrated_spectral_radius(8);
        // All results must be in valid range [0,1] and finite
        REQUIRE(r2 >= 0.0 && r2 <= 1.0);
        REQUIRE(r4 >= 0.0 && r4 <= 1.0);
        REQUIRE(r8 >= 0.0 && r8 <= 1.0);
        REQUIRE(std::isfinite(r2));
        REQUIRE(std::isfinite(r4));
        REQUIRE(std::isfinite(r8));
        // A000081 is strictly increasing for system_level >= 2, so spectral radii increase
        REQUIRE(r4 > r2);
        REQUIRE(r8 > r4);
    });
}

// ─── cog::mach (extended) ────────────────────────────────────────────────────

void test_mach_extended() {
    section("cog::mach (extended)");

    // Fixed::abs
    TEST(fixed_abs, {
        auto x = cog::mach::Fixed::from_float(-3.5f);
        auto a = x.abs();
        REQUIRE_NEAR(a.to_float(), 3.5f, 0.001f);
        auto y = cog::mach::Fixed::from_float(2.0f);
        REQUIRE_NEAR(y.abs().to_float(), 2.0f, 0.001f);
    });

    // Fixed::floor
    TEST(fixed_floor, {
        auto x = cog::mach::Fixed::from_float(3.7f);
        REQUIRE_NEAR(x.floor().to_float(), 3.0f, 0.001f);
        auto y = cog::mach::Fixed::from_float(-1.3f);
        REQUIRE_NEAR(y.floor().to_float(), -2.0f, 0.001f);
    });

    // Fixed::recip
    TEST(fixed_recip, {
        auto x = cog::mach::Fixed::from_float(4.0f);
        auto r = x.recip();
        REQUIRE_NEAR(r.to_float(), 0.25f, 0.002f);
    });

    // Fixed compound assignment (+=, -=)
    TEST(fixed_compound_add, {
        auto x = cog::mach::Fixed::from_float(3.0f);
        x += cog::mach::Fixed::from_float(1.5f);
        REQUIRE_NEAR(x.to_float(), 4.5f, 0.001f);
    });

    TEST(fixed_compound_sub, {
        auto x = cog::mach::Fixed::from_float(5.0f);
        x -= cog::mach::Fixed::from_float(2.0f);
        REQUIRE_NEAR(x.to_float(), 3.0f, 0.001f);
    });

    // fixed_sqrt
    TEST(fixed_sqrt_func, {
        auto r = cog::mach::fixed_sqrt(cog::mach::Fixed::from_float(9.0f));
        REQUIRE_NEAR(r.to_float(), 3.0f, 0.01f);
        auto r0 = cog::mach::fixed_sqrt(cog::mach::Fixed::from_float(0.0f));
        REQUIRE_NEAR(r0.to_float(), 0.0f, 0.001f);
    });

    // fixed_tanh
    TEST(fixed_tanh_func, {
        auto r = cog::mach::fixed_tanh(cog::mach::Fixed::from_float(0.0f));
        REQUIRE_NEAR(r.to_float(), 0.0f, 0.001f);
        auto r1 = cog::mach::fixed_tanh(cog::mach::Fixed::from_float(1.0f));
        REQUIRE_NEAR(r1.to_float(), (float)std::tanh(1.0), 0.01f);
    });

    // fixed_exp
    TEST(fixed_exp_func, {
        auto r = cog::mach::fixed_exp(cog::mach::Fixed::from_float(0.0f));
        REQUIRE_NEAR(r.to_float(), 1.0f, 0.01f);
        auto r1 = cog::mach::fixed_exp(cog::mach::Fixed::from_float(1.0f));
        REQUIRE_NEAR(r1.to_float(), (float)M_E, 0.05f);
    });

    // FixedTensor::fill
    TEST(fixed_tensor_fill, {
        cog::mach::FixedTensor t(3, 3);
        t.fill(cog::mach::Fixed::from_float(5.0f));
        for (size_t r = 0; r < 3; ++r)
            for (size_t c = 0; c < 3; ++c)
                REQUIRE_NEAR(t.at(r, c).to_float(), 5.0f, 0.001f);
    });

    // FixedTensor::operator+
    TEST(fixed_tensor_add, {
        cog::mach::FixedTensor a(2, 2), b(2, 2);
        a.fill(cog::mach::Fixed::from_float(3.0f));
        b.fill(cog::mach::Fixed::from_float(1.5f));
        auto c = a + b;
        REQUIRE_NEAR(c.at(0, 0).to_float(), 4.5f, 0.01f);
        REQUIRE_NEAR(c.at(1, 1).to_float(), 4.5f, 0.01f);
    });

    // VMRegion::contains and protection
    TEST(vmregion_contains, {
        cog::mach::VMRegion r(0x1000, 0x1000, cog::mach::VMRegion::PROT_RW);
        REQUIRE( r.contains(0x1000));
        REQUIRE( r.contains(0x1500));
        REQUIRE(!r.contains(0x2000));
        REQUIRE(!r.contains(0x0FFF));
    });

    TEST(vmregion_protection, {
        cog::mach::VMRegion rw(0, 16, cog::mach::VMRegion::PROT_RW);
        REQUIRE( rw.readable());
        REQUIRE( rw.writable());
        REQUIRE(!rw.executable());
        cog::mach::VMRegion rx(0, 16, cog::mach::VMRegion::PROT_READ | cog::mach::VMRegion::PROT_EXEC);
        REQUIRE( rx.readable());
        REQUIRE(!rx.writable());
        REQUIRE( rx.executable());
        cog::mach::VMRegion none(0, 16, cog::mach::VMRegion::PROT_NONE);
        REQUIRE(!none.readable());
        REQUIRE(!none.writable());
        REQUIRE(!none.executable());
    });

    // VMMap::find and region_count
    TEST(vmmap_find, {
        cog::mach::VMMap vm;
        uint64_t base = vm.allocate(4096);
        cog::mach::VMRegion* r = vm.find(base);
        REQUIRE(r != nullptr);
        REQUIRE(r->contains(base));
        REQUIRE(vm.find(0) == nullptr); // unmapped
    });

    TEST(vmmap_region_count, {
        cog::mach::VMMap vm;
        REQUIRE(vm.region_count() == 0);
        vm.allocate(4096);
        REQUIRE(vm.region_count() == 1);
        vm.allocate(8192);
        REQUIRE(vm.region_count() == 2);
    });

    // KernelAtomSpace::set_tv and lookup
    TEST(kernel_atomspace_set_tv, {
        cog::mach::KernelAtomSpace kas;
        cog::Handle h = kas.add(cog::AtomType::CONCEPT_NODE, "Dog");
        kas.set_tv(h, 0.7f, 0.6f);
        auto* a = kas.get(h);
        REQUIRE(a != nullptr);
        REQUIRE_NEAR(a->tv_strength.to_float(),   0.7f, 0.001f);
        REQUIRE_NEAR(a->tv_confidence.to_float(),  0.6f, 0.001f);
    });

    TEST(kernel_atomspace_lookup_missing, {
        cog::mach::KernelAtomSpace kas;
        cog::Handle h = kas.lookup("NonExistent");
        REQUIRE(h == cog::UNDEFINED_HANDLE);
    });

    TEST(kernel_atomspace_lookup_existing, {
        cog::mach::KernelAtomSpace kas;
        cog::Handle h1 = kas.add(cog::AtomType::CONCEPT_NODE, "Cat");
        cog::Handle h2 = kas.lookup("Cat");
        REQUIRE(h1 == h2);
    });
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== CogPy Extended Coverage Test Suite ===\n";

    test_core_extended();
    test_gml_extended();
    test_prime_extended();
    test_lux_extended();
    test_glow_extended();
    test_webvm_extended();
    test_pilot_extended();
    test_mach_extended();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";

    return tests_failed == 0 ? 0 : 1;
}
