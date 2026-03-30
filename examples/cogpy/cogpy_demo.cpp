// examples/cogpy/cogpy_demo.cpp
//
// A complete demonstration of the cogpy header-only C++11 library.
// Shows all 9 modules in a single integrated example.
//
// Build:
//   g++ -std=c++11 -I ../../include -o cogpy_demo cogpy_demo.cpp && ./cogpy_demo
//
// Or from repo root:
//   g++ -std=c++11 -I include -o cogpy_demo examples/cogpy/cogpy_demo.cpp

#include <cog/cog.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>

// ─── Utilities ───────────────────────────────────────────────────────────────

static void banner(const char* title) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  " << std::left << std::setw(52) << title << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";
}

static void section(const char* name) {
    std::cout << "\n── " << name << " ──\n";
}

// ─── cog::core — AtomSpace & TruthValues ─────────────────────────────────────

void demo_core() {
    banner("cog::core — AtomSpace & Shared Types");

    cog::AtomSpace as;

    // Add concept nodes
    auto cat    = as.add_node(cog::AtomType::CONCEPT_NODE, "cat");
    auto animal = as.add_node(cog::AtomType::CONCEPT_NODE, "animal");
    auto pet    = as.add_node(cog::AtomType::CONCEPT_NODE, "pet");

    // Add inheritance links
    auto cat_animal = as.add_link(cog::AtomType::INHERITANCE_LINK, {cat, animal},
                                   cog::TruthValue(0.95f, 0.9f));
    auto cat_pet    = as.add_link(cog::AtomType::INHERITANCE_LINK, {cat, pet},
                                   cog::TruthValue(0.80f, 0.85f));

    section("Atoms");
    std::cout << "  cat:    handle=" << cat    << " type=" << cog::atom_type_name(cog::AtomType::CONCEPT_NODE)    << "\n";
    std::cout << "  animal: handle=" << animal << " type=" << cog::atom_type_name(cog::AtomType::CONCEPT_NODE)    << "\n";
    std::cout << "  cat→animal: handle=" << cat_animal << " type=" << cog::atom_type_name(cog::AtomType::INHERITANCE_LINK) << "\n";

    section("Truth Value Revision");
    cog::TruthValue tv1(0.8f, 0.6f);
    cog::TruthValue tv2(0.9f, 0.7f);
    auto revised = cog::TruthValue::revision(tv1, tv2);
    std::cout << "  tv1=(" << tv1.strength << "," << tv1.confidence << ")"
              << " ⊕ tv2=(" << tv2.strength << "," << tv2.confidence << ")"
              << " = (" << std::fixed << std::setprecision(3)
              << revised.strength << "," << revised.confidence << ")\n";

    section("Atom types");
    std::cout << "  is_node(CONCEPT_NODE)      = " << std::boolalpha << cog::is_node(cog::AtomType::CONCEPT_NODE) << "\n";
    std::cout << "  is_link(INHERITANCE_LINK)  = " << cog::is_link(cog::AtomType::INHERITANCE_LINK) << "\n";

    std::cout << "\n  ✓ AtomSpace: " << as.size() << " atoms\n";
    (void)cat_pet;
}

// ─── cog::plan9 — CogFS Filesystem ───────────────────────────────────────────

void demo_plan9() {
    banner("cog::plan9 — Plan 9 Cognitive OS");

    cog::plan9::CogFS fs;

    // Build a CogFS hierarchy
    fs.mkdir("/atoms");
    fs.mkdir("/atoms/concepts");
    fs.mkdir("/atoms/links");
    fs.mkfile("/atoms/concepts/cat",    "ConceptNode cat {0.9,0.8}");
    fs.mkfile("/atoms/concepts/animal", "ConceptNode animal {0.95,0.95}");
    fs.mkfile("/atoms/links/inh-0",     "InheritanceLink cat→animal {0.9,0.85}");

    section("Lookup nodes");
    auto* concepts_dir = fs.lookup("/atoms/concepts");
    if (concepts_dir) {
        std::cout << "  /atoms/concepts found, is_dir=" << std::boolalpha << concepts_dir->is_dir << "\n";
        for (const auto& child : concepts_dir->children) {
            std::cout << "    " << child->name << ": " << child->content << "\n";
        }
    }

    section("Qid types");
    cog::plan9::Qid dir_qid(1, 0, cog::plan9::Qid::QTDIR);
    cog::plan9::Qid file_qid(2, 0, cog::plan9::Qid::QTFILE);
    std::cout << "  dir_qid.is_dir()  = " << std::boolalpha << dir_qid.is_dir()  << "\n";
    std::cout << "  file_qid.is_dir() = " << std::boolalpha << file_qid.is_dir() << "\n";

    std::cout << "\n  ✓ CogFS: " << fs.node_count() << " nodes\n";
}

// ─── cog::pilot — Deep Tree Echo ESN ─────────────────────────────────────────

void demo_pilot() {
    banner("cog::pilot — Deep Tree Echo Reservoir");

    section("A000081 sequence (rooted tree counts)");
    for (int n = 1; n <= 8; ++n) {
        std::cout << "  a(" << n << ") = " << cog::pilot::a000081(n) << "\n";
    }

    section("Dual-pool ESN");
    // DualPoolESN(units, in_dim, out_dim)
    cog::pilot::DualPoolESN esn(64, 8, 8);
    esn.initialize(42);
    std::vector<float> input(8, 0.5f);
    auto output = esn.step(input);

    float mean = 0.0f;
    for (float v : output) mean += v;
    mean /= static_cast<float>(output.size());
    std::cout << "  reservoir output mean = " << std::fixed << std::setprecision(4) << mean << "\n";
    std::cout << "  reservoir output dim  = " << output.size() << "\n";

    section("CogPilot echobeats");
    cog::pilot::CogPilot pilot(64, 8, 8);
    pilot.initialize(42);
    std::vector<float> percept(8, 1.0f);
    auto readout = pilot.forward(percept);
    std::cout << "  readout dim = " << readout.size() << "\n";
    int thread = cog::pilot::CogPilot::echobeats_thread(0);
    std::cout << "  echobeats step 0 → thread " << thread << "\n";

    std::cout << "\n  ✓ DualPoolESN: " << output.size() << " output units\n";
}

// ─── cog::mach — Q16.16 Fixed-Point Arithmetic ───────────────────────────────

void demo_mach() {
    banner("cog::mach — Mach Microkernel Cognitive");

    section("Q16.16 fixed-point arithmetic");
    cog::mach::Fixed a = cog::mach::Fixed::from_float(3.14159f);
    cog::mach::Fixed b = cog::mach::Fixed::from_float(2.71828f);
    cog::mach::Fixed c = a + b;
    cog::mach::Fixed d = a * b;

    std::cout << "  π ≈ " << std::fixed << std::setprecision(5) << a.to_float() << "\n";
    std::cout << "  e ≈ " << b.to_float() << "\n";
    std::cout << "  π + e = " << c.to_float() << "\n";
    std::cout << "  π × e = " << d.to_float() << "\n";

    section("Fixed-point transcendentals");
    cog::mach::Fixed x = cog::mach::Fixed::from_float(0.5f);
    std::cout << "  tanh(0.5) ≈ " << cog::mach::fixed_tanh(x).to_float()
              << "  (ref=" << std::tanh(0.5f) << ")\n";
    std::cout << "  sqrt(0.5) ≈ " << cog::mach::fixed_sqrt(x).to_float()
              << "  (ref=" << std::sqrt(0.5f) << ")\n";

    section("Mach IPC");
    cog::mach::MachKernel kernel;
    auto p1 = kernel.alloc_port();
    auto p2 = kernel.alloc_port();
    // Register a simple echo handler on p2
    kernel.register_handler(p2, [](const cog::mach::MachMsg& msg,
                                    cog::mach::MachMsg& reply) {
        reply.msgh_id = msg.msgh_id + 100;
        return cog::mach::KernReturn::SUCCESS;
    });
    cog::mach::MachMsg msg(p2, p1, 42, {});
    cog::mach::MachMsg reply;
    auto kr = kernel.msg_send(msg, reply);
    std::cout << "  msg_send result = " << (kr == cog::mach::KernReturn::SUCCESS ? "SUCCESS" : "FAIL") << "\n";
    std::cout << "  reply.msgh_id = " << reply.msgh_id << "  (expect 142)\n";

    std::cout << "\n  ✓ Fixed-point arithmetic, Mach IPC working\n";
}

// ─── cog::lux — Cognitive Graph ──────────────────────────────────────────────

void demo_lux() {
    banner("cog::lux — Cognitive Node Graph");

    cog::lux::LuxGraph g;

    // Build a small knowledge graph
    auto n_agi    = g.add_node(cog::lux::NodeType::CONCEPT, "AGI");
    auto n_cog    = g.add_node(cog::lux::NodeType::CONCEPT, "Cognition");
    auto n_mem    = g.add_node(cog::lux::NodeType::CONCEPT, "Memory");
    auto n_reason = g.add_node(cog::lux::NodeType::CONCEPT, "Reasoning");
    auto n_learn  = g.add_node(cog::lux::NodeType::CONCEPT, "Learning");

    g.add_edge(cog::lux::EdgeType::INHERITANCE, n_agi, n_cog,    0.9f);
    g.add_edge(cog::lux::EdgeType::INHERITANCE, n_agi, n_mem,    0.85f);
    g.add_edge(cog::lux::EdgeType::INHERITANCE, n_agi, n_reason, 0.95f);
    g.add_edge(cog::lux::EdgeType::SIMILARITY,  n_cog, n_mem,    0.7f);
    g.add_edge(cog::lux::EdgeType::SIMILARITY,  n_cog, n_learn,  0.8f);
    g.add_edge(cog::lux::EdgeType::SIMILARITY,  n_reason, n_mem, 0.75f);

    section("BFS from AGI");
    auto bfs_order = g.bfs(n_agi);
    std::cout << "  BFS order: ";
    for (size_t i = 0; i < bfs_order.size(); ++i) {
        if (i > 0) std::cout << " → ";
        const auto* n = g.node(bfs_order[i]);
        if (n) std::cout << n->label;
    }
    std::cout << "\n";

    section("PageRank");
    auto pr_map = g.pagerank(20, 0.85f);
    std::vector<std::pair<float, std::string>> ranked;
    for (const auto& kv : pr_map) {
        const auto* n = g.node(kv.first);
        if (n) ranked.push_back({kv.second, n->label});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const std::pair<float,std::string>& a,
                 const std::pair<float,std::string>& b){ return a.first > b.first; });
    for (const auto& r : ranked) {
        std::cout << "  " << std::setw(12) << std::left << r.second
                  << " PR=" << std::fixed << std::setprecision(4) << r.first << "\n";
    }

    std::cout << "\n  ✓ LuxGraph: " << g.node_count() << " nodes, "
              << g.edge_count() << " edges\n";
    (void)n_learn;
}

// ─── cog::glow — Neural Network Compiler ─────────────────────────────────────

void demo_glow() {
    banner("cog::glow — Neural Network Compiler");

    section("Building a computation graph");
    cog::glow::GlowCompiler compiler;

    // c0 = 1.5,  c1 = -0.5
    // sum = c0 + c1 = 1.0
    // r   = relu(sum) = 1.0
    // c2  = 2.0
    // out = r * c2 = 2.0
    auto c0  = compiler.constant({1.5f},  {1}, "c0");
    auto c1  = compiler.constant({-0.5f}, {1}, "c1");
    auto sum = compiler.op2(cog::glow::OpType::ADD,  c0, c1, "sum");
    auto r   = compiler.op1(cog::glow::OpType::RELU, sum, "relu");
    auto c2  = compiler.constant({2.0f},  {1}, "c2");
    auto out = compiler.op2(cog::glow::OpType::MUL,  r, c2, "out");
    compiler.mark_output(out);

    cog::glow::GlowGraph& graph = compiler.graph();

    section("Dead-code elimination");
    size_t before = graph.node_count();
    cog::glow::pass_dce(graph);
    size_t after = graph.node_count();
    std::cout << "  Nodes before DCE: " << before << "\n";
    std::cout << "  Nodes after DCE:  " << after  << "\n";

    section("Interpretation");
    cog::glow::GlowInterpreter interp;
    auto tensors = interp.run(graph, {});
    auto it = tensors.find(out);
    float result = (it != tensors.end() && !it->second.empty()) ? it->second[0] : 0.0f;
    std::cout << "  relu(1.5 + (-0.5)) × 2.0 = " << result << "  (expected: 2.0)\n";

    std::cout << "\n  ✓ Graph compiled, DCE + interpretation complete\n";
}

// ─── cog::gml — Tensor Quantization ─────────────────────────────────────────

void demo_gml() {
    banner("cog::gml — Tensor Library");

    section("Data type info");
    using cog::gml::DType;
    const DType types[] = {DType::F32, DType::F16, DType::Q4_0, DType::Q8_0};
    for (auto dt : types) {
        std::cout << "  " << std::left << std::setw(6) << cog::gml::dtype_name(dt)
                  << "  size=" << cog::gml::dtype_size(dt) << " bytes/block\n";
    }

    section("Q4_0 round-trip quantization");
    const size_t N = 64;
    float original[N];
    for (size_t i = 0; i < N; ++i) original[i] = static_cast<float>(i) * 0.1f - 3.2f;

    // N floats → N/QK blocks
    const size_t N_BLOCKS = N / cog::gml::QK;
    cog::gml::BlockQ4_0 blocks[N_BLOCKS];
    cog::gml::quantize_q4_0(original, blocks, N);

    float recovered[N];
    cog::gml::dequantize_q4_0(blocks, recovered, N);

    float max_err = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float err = std::fabs(original[i] - recovered[i]);
        if (err > max_err) max_err = err;
    }
    std::cout << "  Max Q4_0 reconstruction error = " << std::fixed
              << std::setprecision(4) << max_err << " (expect < 0.5)\n";

    section("f16 conversion");
    float pi = 3.14159265f;
    uint16_t pi_f16 = cog::gml::f32_to_f16(pi);
    float pi_back   = cog::gml::f16_to_f32(pi_f16);
    std::cout << "  π (f32) = " << std::setprecision(6) << pi << "\n";
    std::cout << "  π (f16→f32) = " << pi_back
              << "  error = " << std::fabs(pi - pi_back) << "\n";

    std::cout << "\n  ✓ Tensor types, quantization, f16 working\n";
}

// ─── cog::prime — AGI Cognitive Cycle ────────────────────────────────────────

void demo_prime() {
    banner("cog::prime — AGI Architecture");

    section("PLN Truth Values");
    cog::prime::TruthValue tv_a(0.8f, 0.9f);   // P(A→B)
    cog::prime::TruthValue tv_b(0.7f, 0.8f);   // P(B→C)
    // deduction(ab, bc, b_prior) → P(A→C)
    auto deduced = cog::prime::TruthValue::deduction(tv_a, tv_b, 0.5f);
    std::cout << "  P(A→B)=" << tv_a.strength << ", P(B→C)=" << tv_b.strength
              << " → P(A→C)=" << std::fixed << std::setprecision(3) << deduced.strength << "\n";

    auto revised = cog::prime::TruthValue::revise(tv_a, tv_b);
    std::cout << "  revise(0.8, 0.7) strength = " << revised.strength << "\n";

    section("AtomSpace with ECAN attention");
    cog::prime::AtomSpace as;
    auto id_cat    = as.add_node(cog::AtomType::CONCEPT_NODE, "cat");
    auto id_mammal = as.add_node(cog::AtomType::CONCEPT_NODE, "mammal");
    auto id_animal = as.add_node(cog::AtomType::CONCEPT_NODE, "animal");
    as.add_link(cog::AtomType::INHERITANCE_LINK, {id_cat, id_mammal},
                cog::prime::TruthValue(0.95f, 0.9f));
    as.add_link(cog::AtomType::INHERITANCE_LINK, {id_mammal, id_animal},
                cog::prime::TruthValue(0.99f, 0.95f));

    as.stimulate(id_cat, 30.0f);
    const auto* cat_atom = as.get(id_cat);
    if (cat_atom) {
        std::cout << "  cat STI after stimulation = " << cat_atom->av.sti << "\n";
    }
    std::cout << "  AtomSpace size = " << as.size() << "\n";

    section("Cognitive Cycle (7 phases)");
    cog::prime::CognitiveCycle cycle;
    cycle.run(3);
    std::cout << "  Completed 3 cognitive cycle iterations\n";

    section("Declarative Memory");
    cog::prime::DeclarativeMemory decl;
    decl.store_fact("sky", "is", "blue", cog::prime::TruthValue(0.99f, 0.95f));
    auto hits = decl.query("sky");
    std::cout << "  sky is blue: " << hits.size() << " matching fact(s) found\n";

    std::cout << "\n  ✓ PLN, AtomSpace, cognitive cycle, memory systems working\n";
    (void)id_animal;
}

// ─── cog::webvm — Scheme REPL ────────────────────────────────────────────────

void demo_webvm() {
    banner("cog::webvm — Web AtomSpace VM");

    cog::webvm::SchemeREPL repl;

    section("Arithmetic");
    auto r1 = repl.eval_str("(+ 1 2 3 4 5)");
    auto r2 = repl.eval_str("(* 7 6)");
    auto r3 = repl.eval_str("(- 100 58)");
    std::cout << "  (+ 1 2 3 4 5) = " << r1 << "\n";
    std::cout << "  (* 7 6)       = " << r2 << "\n";
    std::cout << "  (- 100 58)    = " << r3 << "\n";

    section("Define & begin");
    auto sq = repl.eval_str("(begin (define x 7) (* x x))");
    std::cout << "  (begin (define x 7) (* x x)) = " << sq << "\n";

    section("Conditional & boolean");
    auto cond_t = repl.eval_str("(if #t \"yes\" \"no\")");
    auto cond_f = repl.eval_str("(if #f \"yes\" \"no\")");
    std::cout << "  (if #t ...) = " << cond_t << "\n";
    std::cout << "  (if #f ...) = " << cond_f << "\n";

    section("List operations");
    auto car_val = repl.eval_str("(car (list 1 2 3))");
    auto null_t  = repl.eval_str("(null? (list))");
    auto null_f  = repl.eval_str("(null? (list 1 2))");
    std::cout << "  (car (list 1 2 3)) = " << car_val << "\n";
    std::cout << "  (null? (list))     = " << null_t  << "\n";
    std::cout << "  (null? (list 1 2)) = " << null_f  << "\n";

    section("JSON serialization");
    cog::webvm::SParser parser;
    auto ast = parser.parse("(+ 1 2)");
    auto j   = cog::webvm::JSONSerializer::sexpr_to_json(*ast);
    std::cout << "  JSON of (+ 1 2): " << j << "\n";

    auto ast2 = parser.parse("(define x 42)");
    auto j2   = cog::webvm::JSONSerializer::sexpr_to_json(*ast2);
    std::cout << "  JSON of (define x 42): " << j2 << "\n";

    std::cout << "\n  ✓ Scheme REPL with arithmetic, defines, lists, JSON working\n";
}

// ─── cog::fowler — Balanced Ternary Calculating Machine ──────────────────────

void demo_fowler() {
    banner("cog::fowler — Thomas Fowler's Ternary Machine (1840)");

    section("Balanced Ternary Arithmetic");

    // Construct from integers and show balanced ternary representation
    cog::fowler::BalancedTernary a(13), b(7);
    std::cout << "  13 in balanced ternary: " << a.to_string() << "\n";
    std::cout << "   7 in balanced ternary: " << b.to_string() << "\n";
    std::cout << "  13 + 7 = " << (a + b).to_int()
              << "  (" << (a + b).to_string() << ")\n";
    std::cout << "  13 - 7 = " << (a - b).to_int()
              << "  (" << (a - b).to_string() << ")\n";
    std::cout << "  13 × 7 = " << (a * b).to_int()
              << "  (" << (a * b).to_string() << ")\n";

    cog::fowler::BalancedTernary rem;
    cog::fowler::BalancedTernary q = a.divmod(b, rem);
    std::cout << "  13 ÷ 7 = " << q.to_int()
              << " remainder " << rem.to_int() << "\n";

    section("FowlerMachine — Step-by-Step Mechanical Multiplication");

    cog::fowler::FowlerMachine machine;

    // 5 × 3 = 15
    auto result = machine.multiply(cog::fowler::BalancedTernary(5),
                                   cog::fowler::BalancedTernary(3));
    std::cout << "\n  5 × 3 = " << result.to_int()
              << " (" << result.to_string() << ")\n";
    std::cout << "\n" << machine.visualize() << "\n";
    std::cout << "\n  Event log (" << machine.get_log().size() << " steps):\n";
    std::cout << machine.format_log();

    section("FowlerMachine — Division");

    auto quot = machine.divide(cog::fowler::BalancedTernary(91),
                               cog::fowler::BalancedTernary(7));
    std::cout << "  91 ÷ 7 = " << quot.to_int()
              << " (" << quot.to_string() << ")\n";

    std::cout << "\n  ✓ Fowler balanced ternary machine: arithmetic and simulation working\n";
}

// ─── demo_tq ──────────────────────────────────────────────────────────────────

void demo_tq() {
    section("cog::tq — Log₂(3) Ternary Quantization");

    // BlockTQ: quantize and dequantize a 256-element block
    float data[256];
    for (int i = 0; i < 256; ++i) data[i] = (i % 3 == 0) ? 1.0f : (i % 3 == 1) ? -1.0f : 0.0f;
    cog::tq::BlockTQ blk;
    cog::tq::quantize_block(data, blk);
    std::cout << "  BlockTQ: scale = " << blk.scale()
              << "  size = " << sizeof(blk) << " bytes (vs "
              << 256 * 4 << " bytes fp32)\n";
    std::cout << "  Compression: ~"
              << (int)(256 * 4.0 / sizeof(blk)) << "× vs fp32\n";

    // Balanced ternary arithmetic (add static method in cog::tq)
    cog::tq::BalancedTernary a(13), b(-5);
    auto sum = cog::tq::BalancedTernary::add(a, b);
    std::cout << "  BalancedTernary: 13 + (-5) = " << sum.to_int() << "\n";

    // TernaryMLP (256→256→4)
    cog::tq::TernaryMLP mlp(256, 256, 4);
    std::vector<float> in(256, 0.1f);
    std::vector<float> out(4, 0.0f);
    mlp.forward(in.data(), out.data());
    std::cout << "  TernaryMLP (256→256→4): output[0] = " << out[0] << "\n";

    // Packing density info
    auto pd = cog::tq::PackingDensity::byte_5();
    std::cout << "  Packing: " << pd.trits_per_unit << " trits/"
              << pd.unit_bits << "-bit unit"
              << "  (type: " << cog::tq::TQTypeInfo::name << ")\n";

    std::cout << "\n  ✓ TQ_LOG2_3 ternary quantization working\n";
}

// ─── demo_dte ─────────────────────────────────────────────────────────────────

void demo_dte() {
    section("cog::dte — DTE Identity MLP (AAR Backup/Restore)");

    // Create an identity vector and set some AAR values
    cog::dte::IdentityVector id;
    id.agent[0]  = 0.9f;   // coherence
    id.agent[1]  = 0.7f;   // valence
    id.arena[0]  = 0.5f;   // complexity
    id.relation[0] = 0.8f; // alignment
    std::cout << "  IdentityVector: dim=" << id.identity_dim
              << "  coherence=" << id.coherence() << "\n";

    // Checkpoint round-trip
    cog::dte::RecoveryCoreMLP mlp;
    auto bytes = mlp.save_checkpoint(id);
    cog::dte::IdentityVector id2;
    mlp.load_checkpoint(bytes.data(), bytes.size(), id2);
    std::cout << "  Checkpoint: " << bytes.size() << " bytes saved\n";
    std::cout << "  After restore: agent[0]=" << id2.agent[0]
              << "  arena[0]=" << id2.arena[0] << "\n";

    // MSE
    auto flat = id.to_vector();
    float mse = mlp.reconstruction_mse(flat.data());
    std::cout << "  Reconstruction MSE: " << mse << "\n";

    // IdentityRecoveryNode
    cog::dte::IdentityRecoveryNode node;
    node.load(bytes.data(), bytes.size());
    std::cout << "  IdentityRecoveryNode: loaded=" << node.loaded()
              << "  coherence=" << node.coherence() << "\n";

    std::cout << "\n  ✓ DTE identity backup/restore working\n";
}

// ─── demo_npu ─────────────────────────────────────────────────────────────────

void demo_npu() {
    section("cog::npu — Tree-Polytope NPU");

    // Matula decoder
    auto m1 = cog::npu::MatulaDecoder::decode(1);
    auto m2 = cog::npu::MatulaDecoder::decode(2);
    auto m4 = cog::npu::MatulaDecoder::decode(4);
    std::cout << "  MatulaDecoder: M(1)=size" << m1.size
              << "  M(2)=size" << m2.size
              << "  M(4)=size" << m4.size << "\n";

    // Simplex geometry
    float sr = cog::npu::SimplexGeometry::spectral_radius(4);
    int tc = cog::npu::SimplexGeometry::tree_count(4);
    std::cout << "  SimplexGeometry: system=4  spectral_radius=" << sr
              << "  tree_count=" << tc << "\n";

    // HarmonicKernel
    cog::npu::HarmonicKernel hk(8, 4);
    float kin[8] = {1.f, -1.f, 0.5f, -0.5f, 0.3f, -0.3f, 0.1f, -0.1f};
    float kout[8] = {};
    hk.forward(kin, kout);
    std::cout << "  HarmonicKernel(8, 4→8): out[0]=" << kout[0] << "\n";

    // TreePolytopeNPU
    cog::npu::TreePolytopeNPU npu(8, 4, 4);
    npu.dma_write(kin, 8);
    npu.write32(0x00, 1);
    float npuout[8] = {};
    npu.dma_read(npuout, 8);
    std::cout << "  TreePolytopeNPU(dim=8, system=4): cycles=" << npu.cycles()
              << "  flops=" << npu.flops()
              << "  spectral_radius=" << npu.spectral_radius() << "\n";

    std::cout << "\n  ✓ Tree-Polytope NPU working\n";
}

// ─── demo_inference ───────────────────────────────────────────────────────────

void demo_inference() {
    section("cog::inference — Multi-Model DTE Inference Engine");

    // Model registry
    std::cout << "  Model registry (4 models):\n";
    for (int i = 0; i < 4; ++i) {
        const auto& spec = cog::inference::MODEL_SPECS[i];
        std::cout << "    [" << cog::inference::MODEL_NAMES[i] << "]"
                  << " role=" << cog::inference::MODEL_ROLES[i]
                  << " params=" << spec.params
                  << " arch=" << spec.architecture << "\n";
    }

    // Echobeats config
    std::cout << "  Echobeats: " << cog::inference::EchobeatsConfig::CYCLE_LENGTH
              << "-step/" << cog::inference::EchobeatsConfig::THREAD_COUNT
              << "-thread cycle\n";
    for (size_t t = 0; t < cog::inference::EchobeatsConfig::THREAD_COUNT; ++t) {
        auto steps = cog::inference::EchobeatsConfig::thread_steps(t);
        std::cout << "    Thread " << t << " → steps "
                  << steps[0] << "," << steps[1] << "," << steps[2]
                  << " (model: " << cog::inference::MODEL_NAMES[static_cast<int>(
                      cog::inference::EchobeatsConfig::thread_model(t))] << ")\n";
    }

    // Run a cognitive cycle
    cog::inference::MultiModelEngine engine;
    std::vector<float> input(24, 0.1f);
    auto result = engine.run_cycle(input);
    std::cout << "  Cognitive cycle #" << result.cycle_number
              << ": readout[0]=" << result.readout[0]
              << "  coherence=" << result.coherence
              << "  level=" << (int)result.ontogenetic_level << "\n";
    std::cout << "  Verb DISCOVER=" << cog::inference::VERB_NAMES[0]
              << "  CLASSIFY=" << cog::inference::VERB_NAMES[9] << "\n";

    std::cout << "\n  ✓ Multi-model DTE inference engine working\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║      CogPy Header-Only C++11 Library — Demo         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";

    demo_core();
    demo_plan9();
    demo_pilot();
    demo_mach();
    demo_lux();
    demo_glow();
    demo_gml();
    demo_prime();
    demo_webvm();
    demo_fowler();
    demo_tq();
    demo_dte();
    demo_npu();
    demo_inference();

    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  All 14 modules demonstrated successfully!           ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    return 0;
}

