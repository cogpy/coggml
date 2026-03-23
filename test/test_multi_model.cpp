// test_multi_model.cpp — Comprehensive test suite for the unified multi-model DTE inference engine
//
// Tests: MultiModelEngine, DualPoolReservoir, AARState, AutognosisMonitor,
//        EchobeatsConfig, ModelSpec, PipelineStep
//
// Build: g++ -std=c++11 -I ../include -o test_multi_model test_multi_model.cpp && ./test_multi_model

#include <cog/inference/multi_model.hpp>
#include <iostream>
#include <cmath>
#include <cassert>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { std::cout << "  " << name << "... "; } while(0)
#define PASS() do { std::cout << "PASS" << std::endl; tests_passed++; } while(0)
#define FAIL(msg) do { std::cout << "FAIL: " << msg << std::endl; tests_failed++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)
#define ASSERT_NEAR(a, b, eps, msg) ASSERT(std::abs((a)-(b)) < (eps), msg)

using namespace cog::inference;

// ─────────────────────────────────────────────────────────────────────────────
// Test: Verb Enumeration
// ─────────────────────────────────────────────────────────────────────────────
void test_verb_names() {
    TEST("Verb names are correct");
    ASSERT(std::string(VERB_NAMES[0]) == "DISCOVER", "DISCOVER name");
    ASSERT(std::string(VERB_NAMES[6]) == "COMPOSE", "COMPOSE name");
    ASSERT(std::string(VERB_NAMES[9]) == "CLASSIFY", "CLASSIFY name");
    ASSERT(static_cast<int>(Verb::CLASSIFY) == 9, "CLASSIFY value");
    PASS();
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Model Registry
// ─────────────────────────────────────────────────────────────────────────────
void test_model_specs() {
    TEST("Model specs are correctly defined");
    ASSERT(MODEL_SPECS[0].id == ModelId::LUCY_DTE, "Lucy ID");
    ASSERT(std::string(MODEL_SPECS[0].hf_repo) == "drzo/lucy-dte", "Lucy repo");
    ASSERT(MODEL_SPECS[0].format == ModelFormat::GGUF, "Lucy format");
    ASSERT(MODEL_SPECS[0].params == 1700000000ULL, "Lucy params");
    ASSERT(MODEL_SPECS[0].context_length == 128000, "Lucy context");
    PASS();
}

void test_model_names() {
    TEST("Model names and roles are correct");
    ASSERT(std::string(MODEL_NAMES[0]) == "lucy-dte", "Lucy name");
    ASSERT(std::string(MODEL_NAMES[1]) == "echoself", "EchoSelf name");
    ASSERT(std::string(MODEL_NAMES[2]) == "unicosys-hypergraph", "Unicosys name");
    ASSERT(std::string(MODEL_NAMES[3]) == "Blocknut", "Blocknut name");
    ASSERT(std::string(MODEL_ROLES[0]) == "voice", "Lucy role");
    ASSERT(std::string(MODEL_ROLES[3]) == "identity", "Blocknut role");
    PASS();
}

void test_echoself_spec() {
    TEST("EchoSelf spec is correct (GPT-2 24M)");
    const auto& spec = MODEL_SPECS[1];
    ASSERT(spec.id == ModelId::ECHOSELF, "ID");
    ASSERT(std::string(spec.architecture) == "gpt2", "arch");
    ASSERT(spec.params == 24000000ULL, "params");
    ASSERT(spec.context_length == 1024, "context");
    ASSERT(spec.format == ModelFormat::SAFETENSORS, "format");
    PASS();
}

void test_unicosys_spec() {
    TEST("Unicosys spec is correct (GNN 36M)");
    const auto& spec = MODEL_SPECS[2];
    ASSERT(spec.id == ModelId::UNICOSYS, "ID");
    ASSERT(std::string(spec.architecture) == "unicosys_hypergraph", "arch");
    ASSERT(spec.params == 36000000ULL, "params");
    ASSERT(std::string(spec.primary_module) == "cog::lux", "primary module");
    PASS();
}

void test_blocknut_spec() {
    TEST("Blocknut spec is correct (DTE-MLP)");
    const auto& spec = MODEL_SPECS[3];
    ASSERT(spec.id == ModelId::BLOCKNUT, "ID");
    ASSERT(std::string(spec.architecture) == "dte_mlp", "arch");
    ASSERT(spec.format == ModelFormat::DTEM, "format");
    ASSERT(std::string(spec.quantization) == "TQ_LOG2_3", "quantization");
    PASS();
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Echobeats Configuration
// ─────────────────────────────────────────────────────────────────────────────
void test_echobeats_cycle() {
    TEST("Echobeats 12-step 4-thread cycle");
    ASSERT(EchobeatsConfig::CYCLE_LENGTH == 12, "cycle length");
    ASSERT(EchobeatsConfig::THREAD_COUNT == 4, "thread count");
    ASSERT(EchobeatsConfig::PHASE_OFFSET == 3, "phase offset");
    PASS();
}

void test_echobeats_thread_mapping() {
    TEST("Echobeats thread-to-model mapping");
    ASSERT(EchobeatsConfig::thread_model(0) == ModelId::LUCY_DTE, "thread 0 = Lucy");
    ASSERT(EchobeatsConfig::thread_model(1) == ModelId::ECHOSELF, "thread 1 = EchoSelf");
    ASSERT(EchobeatsConfig::thread_model(2) == ModelId::UNICOSYS, "thread 2 = Unicosys");
    ASSERT(EchobeatsConfig::thread_model(3) == ModelId::BLOCKNUT, "thread 3 = Blocknut");
    PASS();
}

void test_echobeats_step_assignment() {
    TEST("Echobeats step assignment (phased 3 apart)");
    auto steps0 = EchobeatsConfig::thread_steps(0);
    auto steps1 = EchobeatsConfig::thread_steps(1);
    auto steps2 = EchobeatsConfig::thread_steps(2);
    auto steps3 = EchobeatsConfig::thread_steps(3);
    ASSERT(steps0[0] == 0 && steps0[1] == 4 && steps0[2] == 8, "thread 0 steps");
    ASSERT(steps1[0] == 1 && steps1[1] == 5 && steps1[2] == 9, "thread 1 steps");
    ASSERT(steps2[0] == 2 && steps2[1] == 6 && steps2[2] == 10, "thread 2 steps");
    ASSERT(steps3[0] == 3 && steps3[1] == 7 && steps3[2] == 11, "thread 3 steps");
    PASS();
}

void test_echobeats_dyadic_pairs() {
    TEST("Echobeats 6 dyadic pairs P(i,j)");
    auto d0 = EchobeatsConfig::dyad(0);
    auto d5 = EchobeatsConfig::dyad(5);
    ASSERT(d0.a == 0 && d0.b == 1, "P(0,1)");
    ASSERT(d5.a == 2 && d5.b == 3, "P(2,3)");
    ASSERT(EchobeatsConfig::NUM_DYADS == 6, "6 dyads");
    PASS();
}

void test_echobeats_triads() {
    TEST("Echobeats complementary triads MP1/MP2");
    auto mp1_0 = EchobeatsConfig::mp1_triad(0);
    auto mp2_0 = EchobeatsConfig::mp2_triad(0);
    ASSERT(mp1_0.a == 0 && mp1_0.b == 1 && mp1_0.c == 2, "MP1[0] = {0,1,2}");
    ASSERT(mp2_0.a == 0 && mp2_0.b == 2 && mp2_0.c == 3, "MP2[0] = {0,2,3}");
    PASS();
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Dual-Pool Reservoir
// ─────────────────────────────────────────────────────────────────────────────
void test_reservoir_init() {
    TEST("DualPoolReservoir initialization");
    ReservoirConfig cfg;
    cfg.fast_pool_size = 32;
    cfg.slow_pool_size = 16;
    cfg.fast_leak_rate = 0.3f;
    cfg.slow_leak_rate = 0.05f;
    cfg.spectral_radius = 0.95f;
    cfg.input_dim = 24;
    cfg.readout_dim = 5;
    
    DualPoolReservoir r(cfg);
    ASSERT(r.fast_state.size() == 32, "fast state size");
    ASSERT(r.slow_state.size() == 16, "slow state size");
    ASSERT(r.readout_weights.size() == (32 + 16) * 5, "readout weights size");
    PASS();
}

void test_reservoir_step() {
    TEST("DualPoolReservoir step produces output");
    ReservoirConfig cfg;
    cfg.fast_pool_size = 16;
    cfg.slow_pool_size = 8;
    cfg.fast_leak_rate = 0.3f;
    cfg.slow_leak_rate = 0.05f;
    cfg.spectral_radius = 0.95f;
    cfg.input_dim = 5;
    cfg.readout_dim = 3;
    
    DualPoolReservoir r(cfg);
    std::vector<float> input = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    auto state = r.step(input);
    ASSERT(state.size() == 24, "combined state size = fast + slow");
    
    // State should be non-zero after stepping
    float energy = 0.0f;
    for (float s : state) energy += s * s;
    ASSERT(energy > 0.0f, "state has energy after step");
    PASS();
}

void test_reservoir_echo_property() {
    TEST("Reservoir echo state property (signal decays)");
    ReservoirConfig cfg;
    cfg.fast_pool_size = 16;
    cfg.slow_pool_size = 8;
    cfg.fast_leak_rate = 0.3f;
    cfg.slow_leak_rate = 0.05f;
    cfg.spectral_radius = 0.95f;
    cfg.input_dim = 5;
    cfg.readout_dim = 3;
    
    DualPoolReservoir r(cfg);
    
    // Drive with strong input
    std::vector<float> strong = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    for (int i = 0; i < 10; ++i) r.step(strong);
    
    float energy_driven = 0.0f;
    for (float s : r.fast_state) energy_driven += s * s;
    
    // Now step with zero input — energy should decay
    std::vector<float> zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 50; ++i) r.step(zero);
    
    float energy_decayed = 0.0f;
    for (float s : r.fast_state) energy_decayed += s * s;
    
    ASSERT(energy_decayed < energy_driven, "energy decays (echo state property)");
    PASS();
}

void test_reservoir_readout() {
    TEST("Reservoir readout produces correct dimensions");
    ReservoirConfig cfg;
    cfg.fast_pool_size = 16;
    cfg.slow_pool_size = 8;
    cfg.fast_leak_rate = 0.3f;
    cfg.slow_leak_rate = 0.05f;
    cfg.spectral_radius = 0.95f;
    cfg.input_dim = 5;
    cfg.readout_dim = 3;
    
    DualPoolReservoir r(cfg);
    std::vector<float> input = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    r.step(input);
    
    auto readout = r.readout();
    ASSERT(readout.size() == 3, "readout dimension");
    PASS();
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: AAR Identity State
// ─────────────────────────────────────────────────────────────────────────────
void test_aar_default_state() {
    TEST("AAR default state initialization");
    AARState state;
    ASSERT_NEAR(state.coherence, 0.5f, 0.01f, "coherence default");
    ASSERT_NEAR(state.valence, 0.0f, 0.01f, "valence default");
    ASSERT_NEAR(state.stability, 0.7f, 0.01f, "stability default");
    ASSERT(state.level == 0, "starts at EMBRYONIC");
    PASS();
}

void test_aar_to_vector() {
    TEST("AAR state serializes to 24-dim vector");
    AARState state;
    auto vec = state.to_vector();
    ASSERT(vec.size() == 24, "vector size");
    ASSERT_NEAR(vec[0], 0.5f, 0.01f, "first element = coherence");
    PASS();
}

void test_aar_update_from_readout() {
    TEST("AAR state updates from reservoir readout");
    AARState state;
    float old_alignment = state.alignment;
    
    std::vector<float> readout = {2.0f, 1.5f, 1.0f, 0.5f, 0.0f};
    state.update_from_readout(readout);
    
    // Alignment should have moved toward sigmoid(2.0) ≈ 0.88
    ASSERT(state.alignment != old_alignment, "alignment changed");
    ASSERT(state.alignment > old_alignment, "alignment increased (positive readout)");
    PASS();
}

void test_aar_ontogenetic_levels() {
    TEST("AAR ontogenetic level names");
    // Level 0=EMBRYONIC, 1=INFANT, 2=CHILD, 3=ADOLESCENT, 4=ADULT, 5=ELDER, 6=SAGE
    AARState state;
    ASSERT(state.level == 0, "starts EMBRYONIC");
    state.level = 4;
    ASSERT(state.level == 4, "can set to ADULT");
    PASS();
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Autognosis Monitor
// ─────────────────────────────────────────────────────────────────────────────
void test_autognosis_init() {
    TEST("Autognosis 5-level initialization");
    AutognosisMonitor ag;
    ASSERT(ag.NUM_LEVELS == 5, "5 levels");
    ASSERT_NEAR(ag.levels[0].confidence, 0.90f, 0.01f, "L0 confidence");
    ASSERT_NEAR(ag.levels[4].confidence, 0.50f, 0.01f, "L4 confidence");
    PASS();
}

void test_autognosis_monitoring() {
    TEST("Autognosis monitoring updates levels");
    AutognosisMonitor ag;
    AARState state;
    
    ReservoirConfig cfg;
    cfg.fast_pool_size = 16;
    cfg.slow_pool_size = 8;
    cfg.fast_leak_rate = 0.3f;
    cfg.slow_leak_rate = 0.05f;
    cfg.spectral_radius = 0.95f;
    cfg.input_dim = 5;
    cfg.readout_dim = 3;
    DualPoolReservoir r(cfg);
    
    ag.monitor(state, r);
    
    // L0 should reflect coherence
    ASSERT_NEAR(ag.levels[0].confidence, state.coherence, 0.01f, "L0 = coherence");
    // L2 should reflect metacoherence
    ASSERT_NEAR(ag.levels[2].confidence, state.metacoherence, 0.01f, "L2 = metacoherence");
    PASS();
}

void test_autognosis_convergence() {
    TEST("Autognosis convergence detection");
    AutognosisMonitor ag;
    // Not converged initially
    ASSERT(!ag.has_converged(0.01f), "not converged initially");
    
    // Force high confidence
    for (size_t i = 0; i < ag.NUM_LEVELS; ++i) {
        ag.levels[i].confidence = 0.999f;
    }
    ASSERT(ag.has_converged(0.01f), "converged when all high");
    PASS();
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: MultiModelEngine
// ─────────────────────────────────────────────────────────────────────────────
void test_engine_init() {
    TEST("MultiModelEngine initialization");
    MultiModelEngine engine;
    ASSERT(!engine.model_loaded[0], "Lucy not loaded initially");
    ASSERT(!engine.model_loaded[3], "Blocknut not loaded initially");
    ASSERT(engine.current_step == 0, "step starts at 0");
    ASSERT(engine.total_cycles == 0, "cycles start at 0");
    PASS();
}

void test_engine_create_models() {
    TEST("MultiModelEngine model loading");
    MultiModelEngine engine;
    
    ASSERT(engine.create_model(ModelId::LUCY_DTE), "load Lucy");
    ASSERT(engine.create_model(ModelId::ECHOSELF), "load EchoSelf");
    ASSERT(engine.create_model(ModelId::UNICOSYS), "load Unicosys");
    ASSERT(engine.create_model(ModelId::BLOCKNUT), "load Blocknut");
    
    for (int i = 0; i < 4; ++i) {
        ASSERT(engine.model_loaded[i], "model loaded");
    }
    PASS();
}

void test_engine_classify() {
    TEST("MultiModelEngine model classification");
    ASSERT(std::string(MultiModelEngine::classify_model(ModelId::LUCY_DTE)) == "decoder-only-transformer", "Lucy type");
    ASSERT(std::string(MultiModelEngine::classify_model(ModelId::ECHOSELF)) == "causal-lm-gpt2", "EchoSelf type");
    ASSERT(std::string(MultiModelEngine::classify_model(ModelId::UNICOSYS)) == "graph-attention-network", "Unicosys type");
    ASSERT(std::string(MultiModelEngine::classify_model(ModelId::BLOCKNUT)) == "dense-mlp-identity", "Blocknut type");
    PASS();
}

void test_engine_inspect() {
    TEST("MultiModelEngine model inspection");
    MultiModelEngine engine;
    const auto& lucy = engine.inspect_model(ModelId::LUCY_DTE);
    ASSERT(std::string(lucy.hf_repo) == "drzo/lucy-dte", "Lucy repo");
    ASSERT(lucy.context_length == 128000, "Lucy context");
    
    const auto& echo = engine.inspect_model(ModelId::ECHOSELF);
    ASSERT(echo.params == 24000000ULL, "EchoSelf params");
    PASS();
}

void test_engine_echobeats_step() {
    TEST("MultiModelEngine Echobeats step");
    MultiModelEngine engine;
    engine.create_model(ModelId::LUCY_DTE);
    
    std::vector<float> input(24, 0.1f);
    auto readout = engine.compose_echobeats_step(input);
    
    ASSERT(readout.size() == 5, "readout dimension");
    ASSERT(engine.identity.echobeats_step == 1, "step incremented");
    PASS();
}

void test_engine_full_cycle() {
    TEST("MultiModelEngine full 12-step cognitive cycle");
    MultiModelEngine engine;
    for (int i = 0; i < 4; ++i) engine.create_model(static_cast<ModelId>(i));
    
    std::vector<float> input(24, 0.2f);
    auto result = engine.run_cycle(input);
    
    ASSERT(result.readout.size() == 5, "readout produced");
    ASSERT(result.cycle_number == 1, "cycle count");
    // After 1 cycle, history is tiny so XP < 100, level stays EMBRYONIC
    ASSERT(result.ontogenetic_level <= 1, "EMBRYONIC or INFANT after 1 cycle");
    ASSERT(result.coherence > 0.0f, "coherence > 0");
    PASS();
}

void test_engine_multiple_cycles() {
    TEST("MultiModelEngine multiple cycles evolve identity");
    MultiModelEngine engine;
    for (int i = 0; i < 4; ++i) engine.create_model(static_cast<ModelId>(i));
    
    std::vector<float> input(24, 0.3f);
    MultiModelEngine::CycleResult last;
    for (int c = 0; c < 100; ++c) {
        last = engine.run_cycle(input);
    }
    
    ASSERT(last.cycle_number == 100, "100 cycles completed");
    ASSERT(engine.identity.history > 0.0f, "history accumulated");
    PASS();
}

void test_engine_destroy() {
    TEST("MultiModelEngine cleanup");
    MultiModelEngine engine;
    for (int i = 0; i < 4; ++i) engine.create_model(static_cast<ModelId>(i));
    
    engine.destroy();
    for (int i = 0; i < 4; ++i) {
        ASSERT(!engine.model_loaded[i], "model unloaded");
    }
    PASS();
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Pipeline Step
// ─────────────────────────────────────────────────────────────────────────────
void test_pipeline_step() {
    TEST("PipelineStep construction");
    PipelineStep step(Verb::COMPOSE, ModelId::LUCY_DTE, "llama_decode");
    ASSERT(step.verb == Verb::COMPOSE, "verb");
    ASSERT(step.model == ModelId::LUCY_DTE, "model");
    ASSERT(step.api_name == "llama_decode", "api");
    PASS();
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Integration — Full Pipeline
// ─────────────────────────────────────────────────────────────────────────────
void test_full_pipeline_integration() {
    TEST("Full pipeline: ORCHESTRATE → CREATE → COMPOSE → OBSERVE → DESTROY");
    MultiModelEngine engine;
    
    // ORCHESTRATE
    ASSERT(engine.orchestrate_init(), "backend init");
    
    // CREATE all models
    for (int i = 0; i < 4; ++i) {
        ASSERT(engine.create_model(static_cast<ModelId>(i)), "create model");
    }
    
    // COMPOSE: Run cognitive cycles
    std::vector<float> input(24, 0.5f);
    for (int c = 0; c < 10; ++c) {
        auto result = engine.run_cycle(input);
        ASSERT(result.readout.size() == 5, "readout valid");
    }
    
    // OBSERVE: Check autognosis
    engine.observe_autognosis();
    ASSERT(engine.autognosis.levels[0].confidence >= 0.0f, "L0 valid");
    ASSERT(engine.autognosis.levels[4].confidence >= 0.0f, "L4 valid");
    
    // DESTROY
    engine.destroy();
    ASSERT(!engine.model_loaded[0], "cleaned up");
    
    PASS();
}

void test_reservoir_dual_pool_rates() {
    TEST("Dual pool leak rates: fast > slow");
    MultiModelEngine engine;
    ASSERT(engine.reservoir.config.fast_leak_rate > engine.reservoir.config.slow_leak_rate,
           "fast leak > slow leak");
    ASSERT_NEAR(engine.reservoir.config.fast_leak_rate, 0.3f, 0.01f, "fast = 0.3");
    ASSERT_NEAR(engine.reservoir.config.slow_leak_rate, 0.05f, 0.01f, "slow = 0.05");
    PASS();
}

void test_spectral_radius() {
    TEST("Spectral radius at edge-of-chaos (0.95)");
    MultiModelEngine engine;
    ASSERT_NEAR(engine.reservoir.config.spectral_radius, 0.95f, 0.01f, "ρ = 0.95");
    PASS();
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Multi-Model DTE Inference Engine — Test Suite              ║" << std::endl;
    std::cout << "║  /optimal-cognitive-grip ( /llama-cpp-skillm { 4 models } ) ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Model Registry:" << std::endl;
    test_verb_names();
    test_model_specs();
    test_model_names();
    test_echoself_spec();
    test_unicosys_spec();
    test_blocknut_spec();
    
    std::cout << std::endl << "Echobeats Configuration:" << std::endl;
    test_echobeats_cycle();
    test_echobeats_thread_mapping();
    test_echobeats_step_assignment();
    test_echobeats_dyadic_pairs();
    test_echobeats_triads();
    
    std::cout << std::endl << "Dual-Pool Reservoir:" << std::endl;
    test_reservoir_init();
    test_reservoir_step();
    test_reservoir_echo_property();
    test_reservoir_readout();
    test_reservoir_dual_pool_rates();
    test_spectral_radius();
    
    std::cout << std::endl << "AAR Identity State:" << std::endl;
    test_aar_default_state();
    test_aar_to_vector();
    test_aar_update_from_readout();
    test_aar_ontogenetic_levels();
    
    std::cout << std::endl << "Autognosis Monitor:" << std::endl;
    test_autognosis_init();
    test_autognosis_monitoring();
    test_autognosis_convergence();
    
    std::cout << std::endl << "MultiModelEngine:" << std::endl;
    test_engine_init();
    test_engine_create_models();
    test_engine_classify();
    test_engine_inspect();
    test_engine_echobeats_step();
    test_engine_full_cycle();
    test_engine_multiple_cycles();
    test_engine_destroy();
    
    std::cout << std::endl << "Pipeline:" << std::endl;
    test_pipeline_step();
    
    std::cout << std::endl << "Integration:" << std::endl;
    test_full_pipeline_integration();
    
    std::cout << std::endl;
    std::cout << "════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "Results: " << tests_passed << " passed, " << tests_failed << " failed" << std::endl;
    std::cout << "════════════════════════════════════════════════════════════════" << std::endl;
    
    return tests_failed > 0 ? 1 : 0;
}
